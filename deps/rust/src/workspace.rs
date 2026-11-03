//! Caller-owned aligned workspace. Mutable borrows prevent simultaneous reuse.
use crate::typed::{MatrixData, TypedMatrix};
use crate::{guarded, OwnedMatrix};
use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::traits::{math_utils::*, ComplexField};
use faer::{MatRef, Par};
use rustcall_julia_macros::julia;
#[julia]
pub struct OwnedScratch {
    memory: Option<MemBuffer>,
    bytes: usize,
    alignment: usize,
}
#[julia]
impl OwnedScratch {
    #[julia]
    pub fn new() -> Self {
        Self {
            memory: None,
            bytes: 0,
            alignment: 1,
        }
    }
    #[julia]
    pub fn scratch_reserve(&mut self, bytes: usize, alignment: usize) -> Result<(), String> {
        guarded(|| {
            if !alignment.is_power_of_two() {
                return Err("scratch alignment must be a positive power of two".into());
            }
            if self.memory.is_some() && bytes <= self.bytes && alignment <= self.alignment {
                return Ok(());
            }
            let bytes = bytes.max(self.bytes);
            let alignment = alignment.max(self.alignment);
            let mem = MemBuffer::try_new(StackReq::new_aligned::<u8>(bytes, alignment))
                .map_err(|e| format!("scratch allocation: {e:?}"))?;
            self.memory = Some(mem);
            self.bytes = bytes;
            self.alignment = alignment;
            Ok(())
        })
    }
    #[julia]
    pub fn scratch_info(&self, out: *mut usize) -> Result<(), String> {
        guarded(|| {
            if self.memory.is_none() {
                return Err("scratch is closed or uninitialized".into());
            }
            unsafe {
                out.write(self.bytes);
                out.add(1).write(self.alignment);
            }
            Ok(())
        })
    }
    #[julia]
    pub fn scratch_close(&mut self) -> Result<(), String> {
        guarded(|| {
            self.memory = None;
            self.bytes = 0;
            self.alignment = 1;
            Ok(())
        })
    }
}
fn requirement(n: usize, tag: i32) -> Result<StackReq, String> {
    Ok(match tag {
        0 => faer::linalg::pfaffian::pfaffian_scratch::<f32>(n),
        1 => faer::linalg::pfaffian::pfaffian_scratch::<f64>(n),
        2 => faer::linalg::pfaffian::pfaffian_scratch::<faer::c32>(n),
        3 => faer::linalg::pfaffian::pfaffian_scratch::<faer::c64>(n),
        _ => return Err("invalid Pfaffian scalar".into()),
    })
}
#[julia]
pub fn pfaffian_requirements(n: usize, tag: i32, out: *mut usize) -> Result<(), String> {
    guarded(|| {
        let req = requirement(n, tag)?;
        unsafe {
            out.write(req.size_bytes());
            out.add(1).write(req.align_bytes());
        }
        Ok(())
    })
}
fn run<T: ComplexField>(
    a: MatRef<'_, T>,
    scratch: &mut OwnedScratch,
    out: *mut u8,
    team: Option<&crate::teams::OwnedTeam>,
) -> Result<(), String> {
    if a.nrows() != a.ncols() {
        return Err("Pfaffian requires square input".into());
    }
    if !a.is_all_finite() {
        return Err("Pfaffian input must be finite".into());
    }
    for j in 0..a.ncols() {
        for i in j..a.nrows() {
            if a[(i, j)] != neg(&a[(j, i)]) {
                return Err("Pfaffian requires skew-symmetric input".into());
            }
        }
    }
    let req = faer::linalg::pfaffian::pfaffian_scratch::<T>(a.nrows());
    if scratch.bytes < req.size_bytes() || scratch.alignment < req.align_bytes() {
        return Err("scratch capacity/alignment is insufficient; reserve before calling".into());
    }
    let mem = scratch
        .memory
        .as_mut()
        .ok_or("scratch is closed or uninitialized")?;
    let value = if let Some(team) = team {
        team.execute(|par| faer::linalg::pfaffian::pfaffian(a, par, MemStack::new(mem)))?
    } else {
        faer::linalg::pfaffian::pfaffian(a, Par::Seq, MemStack::new(mem))
    }
    .map_err(|e| format!("{e:?}"))?;
    unsafe {
        out.cast::<T>().write(value);
    }
    Ok(())
}
#[julia]
impl OwnedMatrix {
    #[julia]
    pub fn pfaffian_workspace(
        &self,
        scratch: &mut OwnedScratch,
        out: *mut u8,
    ) -> Result<(), String> {
        guarded(|| run(self.matrix.as_ref(), scratch, out, None))
    }
}
#[julia]
impl TypedMatrix {
    #[julia]
    pub fn pfaffian_workspace(
        &self,
        scratch: &mut OwnedScratch,
        out: *mut u8,
    ) -> Result<(), String> {
        guarded(|| match self.data()? {
            MatrixData::F32(a) => run(a.as_ref(), scratch, out, None),
            MatrixData::F64(a) => run(a.as_ref(), scratch, out, None),
            MatrixData::C32(a) => run(a.as_ref(), scratch, out, None),
            MatrixData::C64(a) => run(a.as_ref(), scratch, out, None),
        })
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pfaffian_reuses_exact_buffer() {
        let a = faer::Mat::<f64>::from_fn(4, 4, |i, j| match (i, j) {
            (0, 1) => 2.0,
            (1, 0) => -2.0,
            (2, 3) => 3.0,
            (3, 2) => -3.0,
            _ => 0.0,
        });
        let req = faer::linalg::pfaffian::pfaffian_scratch::<f64>(4);
        let mut scratch = OwnedScratch::new();
        scratch
            .scratch_reserve(req.size_bytes(), req.align_bytes())
            .unwrap();
        let address = scratch.memory.as_mut().unwrap().as_mut_ptr();
        let mut result = 0.0f64;
        for _ in 0..2 {
            run(
                a.as_ref(),
                &mut scratch,
                (&mut result as *mut f64).cast(),
                None,
            )
            .unwrap();
            assert_eq!(result, 6.0);
            assert_eq!(scratch.memory.as_mut().unwrap().as_mut_ptr(), address);
        }
    }
}

#[julia]
impl OwnedMatrix {
    #[julia]
    pub fn pfaffian_workspace_team(
        &self,
        scratch: &mut OwnedScratch,
        team: &crate::teams::OwnedTeam,
        out: *mut u8,
    ) -> Result<(), String> {
        guarded(|| run(self.matrix.as_ref(), scratch, out, Some(team)))
    }
}
#[julia]
impl TypedMatrix {
    #[julia]
    pub fn pfaffian_workspace_team(
        &self,
        scratch: &mut OwnedScratch,
        team: &crate::teams::OwnedTeam,
        out: *mut u8,
    ) -> Result<(), String> {
        guarded(|| match self.data()? {
            MatrixData::F32(a) => run(a.as_ref(), scratch, out, Some(team)),
            MatrixData::F64(a) => run(a.as_ref(), scratch, out, Some(team)),
            MatrixData::C32(a) => run(a.as_ref(), scratch, out, Some(team)),
            MatrixData::C64(a) => run(a.as_ref(), scratch, out, Some(team)),
        })
    }
}

impl OwnedScratch {
    pub(crate) fn stack(&mut self, req: StackReq) -> Result<&mut MemStack, String> {
        if self.bytes < req.size_bytes() || self.alignment < req.align_bytes() {
            return Err("scratch capacity/alignment is insufficient".into());
        }
        Ok(MemStack::new(
            self.memory
                .as_mut()
                .ok_or("scratch is closed or uninitialized")?,
        ))
    }
}

#[cfg(test)]
mod spectral_workspace_tests {
    use super::*;
    use crate::spectral_factors::OwnedSpectral;
    #[test]
    fn saved_spectral_pinv_reuses_buffer_and_preserves_failed_output() {
        for eigen in [false, true] {
            let a = OwnedMatrix {
                matrix: faer::Mat::from_fn(
                    3,
                    3,
                    |i, j| if i == j { [-4.0, 1e-4, 0.0][i] } else { 0.0 },
                ),
            };
            let mut factor = OwnedSpectral::new();
            a.spectral_factor(&mut factor, eigen, true, false).unwrap();
            let mut req = [0usize; 2];
            factor
                .spectral_pinv_requirements(3, 3, 1, eigen, req.as_mut_ptr())
                .unwrap();
            let mut scratch = OwnedScratch::new();
            let mut output = OwnedMatrix {
                matrix: faer::Mat::full(1, 1, 99.0),
            };
            assert!(output
                .spectral_pinv_workspace(&factor, &mut scratch, 3, 3, 1, eigen, 1e-3, 0.0, false)
                .is_err());
            assert_eq!(output.matrix[(0, 0)], 99.0);
            let mut info = [123usize, 456];
            assert!(factor
                .spectral_pinv_requirements(4, 3, 1, eigen, info.as_mut_ptr())
                .is_err());
            assert_eq!(info, [123, 456]);
            scratch.scratch_reserve(req[0], req[1]).unwrap();
            let address = scratch.memory.as_mut().unwrap().as_mut_ptr();
            for _ in 0..2 {
                output
                    .spectral_pinv_workspace(
                        &factor,
                        &mut scratch,
                        3,
                        3,
                        1,
                        eigen,
                        1e-3,
                        0.0,
                        false,
                    )
                    .unwrap();
                assert!((output.matrix[(0, 0)] + 0.25).abs() < 1e-12);
                assert_eq!(output.matrix[(1, 1)], 0.0);
                assert_eq!(scratch.memory.as_mut().unwrap().as_mut_ptr(), address);
            }
        }
    }
}
