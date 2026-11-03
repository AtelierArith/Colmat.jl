//! Validated NPY import with owned aligned native-endian storage.
use crate::typed::{MatrixData, TypedMatrix};
use crate::{guarded, OwnedMatrix};
use faer::{
    io::npy::{FromNpy, Npy},
    traits::ComplexField,
    Mat,
};
use rustcall_julia_macros::julia;
// pest's stack guard discovers OS-thread bounds, which do not describe a
// Julia Task's switched stack. Parse on an actual Rust thread with known bounds.
// Keep the guard enabled, including for untrusted/malformed headers.
fn parser_stack<T: Send>(f: impl FnOnce() -> T + Send) -> Result<T, String> {
    std::thread::scope(|scope| {
        std::thread::Builder::new()
            .name("colmat-npy-header".into())
            .stack_size(2 * 1024 * 1024)
            .spawn_scoped(scope, f)
            .map_err(|e| format!("NPY parser thread: {e}"))?
            .join()
            .map_err(|_| "NPY parser thread panicked".to_string())
    })
}
#[derive(Clone)]
#[repr(C, align(64))]
struct Block([u8; 64]);
#[derive(Clone)]
struct Data {
    blocks: Vec<Block>,
    len: usize,
    m: usize,
    n: usize,
    rank: usize,
    tag: i32,
    fortran: bool,
    version: (u8, u8),
    endian: usize,
}
impl Data {
    fn bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.blocks.as_ptr().cast(), self.len) }
    }
    fn npy(&self) -> Result<Npy<'_>, String> {
        parser_stack(|| Npy::new(self.bytes()))?.map_err(|e| format!("normalized NPY: {e}"))
    }
    fn check(&self, tag: i32, m: usize, n: usize) -> Result<(), String> {
        if self.tag != tag || self.m != m || self.n != n {
            return Err("NPY owner metadata no longer matches the view".into());
        }
        Ok(())
    }
}
fn load(bytes: &[u8]) -> Result<Data, String> {
    let invalid = |s: &str| format!("invalid NPY: {s}");
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err(invalid("missing/truncated magic or version"));
    }
    let version = (bytes[6], bytes[7]);
    if version.1 != 0 || !(1..=3).contains(&version.0) {
        return Err(invalid("unsupported version"));
    }
    let start = if version.0 == 1 { 10 } else { 12 };
    if bytes.len() < start {
        return Err(invalid("truncated header length"));
    }
    let header = if start == 10 {
        u16::from_le_bytes(bytes[8..10].try_into().unwrap()) as usize
    } else {
        u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize
    };
    let prefix = start
        .checked_add(header)
        .ok_or_else(|| invalid("header length overflow"))?;
    if prefix > bytes.len() {
        return Err(invalid("truncated header"));
    }
    let file =
        parser_stack(|| npyz::NpyFile::new(bytes))?.map_err(|e| format!("invalid NPY: {e}"))?;
    let shape = file.shape();
    let rank = shape.len();
    if rank > 2 {
        return Err(invalid(
            "only scalar, vector and matrix shapes are supported",
        ));
    }
    let dim = |i| -> Result<usize, String> {
        let d = shape.get(i).copied().unwrap_or(1);
        let d = usize::try_from(d).map_err(|_| invalid("shape overflow"))?;
        if d > isize::MAX as usize {
            return Err(invalid("shape exceeds addressable dimensions"));
        }
        Ok(d)
    };
    let (m, n) = (dim(0)?, dim(1)?);
    let npyz::DType::Plain(dtype) = file.dtype() else {
        return Err(invalid("structured/object dtypes are unsupported"));
    };
    let (tag, size, component, descr) = match (dtype.type_char(), dtype.size_field()) {
        (npyz::TypeChar::Float, 4) => (0, 4usize, 4usize, "f4"),
        (npyz::TypeChar::Float, 8) => (1, 8, 8, "f8"),
        (npyz::TypeChar::Complex, 8) => (2, 8, 4, "c8"),
        (npyz::TypeChar::Complex, 16) => (3, 16, 8, "c16"),
        _ => {
            return Err(invalid(
                "dtype must be float32/float64/complex64/complex128",
            ))
        }
    };
    let payload = m
        .checked_mul(n)
        .and_then(|k| k.checked_mul(size))
        .ok_or_else(|| invalid("shape/payload overflow"))?;
    if bytes.len() - prefix != payload {
        return Err(invalid("payload length must exactly match shape and dtype"));
    }
    let endian = match dtype.endianness() {
        npyz::Endianness::Little => 1,
        npyz::Endianness::Big => 2,
        npyz::Endianness::Irrelevant => 0,
    };
    let fortran = file.header().order() == npyz::Order::Fortran;
    let shape = match rank {
        0 => "()".into(),
        1 => format!("({m},)"),
        _ => format!("({m}, {n})"),
    };
    let native_little = cfg!(target_endian = "little");
    let marker = if native_little { '<' } else { '>' };
    let mut header = format!(
        "{{'descr': '{marker}{descr}', 'fortran_order': {}, 'shape': {shape}, }}",
        if fortran { "True" } else { "False" }
    );
    let padding = (64 - ((10 + header.len() + 1) % 64)) % 64;
    header.push_str(&" ".repeat(padding));
    header.push('\n');
    let mut canonical = b"\x93NUMPY\x01\x00".to_vec();
    canonical.extend_from_slice(&(header.len() as u16).to_le_bytes());
    canonical.extend_from_slice(header.as_bytes());
    let len = canonical
        .len()
        .checked_add(payload)
        .ok_or_else(|| invalid("buffer size overflow"))?;
    let blocks_len = len
        .checked_add(63)
        .ok_or_else(|| invalid("buffer size overflow"))?
        / 64;
    let mut blocks = vec![Block([0; 64]); blocks_len];
    let out = unsafe { std::slice::from_raw_parts_mut(blocks.as_mut_ptr().cast::<u8>(), len) };
    out[..canonical.len()].copy_from_slice(&canonical);
    out[canonical.len()..].copy_from_slice(&bytes[prefix..]);
    if endian != 0 && (endian == 1) != native_little {
        for word in out[canonical.len()..].chunks_exact_mut(component) {
            word.reverse();
        }
    }
    Ok(Data {
        blocks,
        len,
        m,
        n,
        rank,
        tag,
        fortran,
        version,
        endian,
    })
}
#[julia]
#[derive(Clone)]
pub struct OwnedNpy {
    data: Option<Data>,
}
impl OwnedNpy {
    fn data(&self) -> Result<&Data, String> {
        self.data
            .as_ref()
            .ok_or("NPY owner is not initialized".into())
    }
}
#[julia]
impl OwnedNpy {
    #[julia]
    pub fn new() -> Self {
        Self { data: None }
    }
    #[julia]
    pub fn npy_load(&mut self, p: *const u8, len: usize) -> Result<(), String> {
        guarded(|| {
            let bytes = if len == 0 {
                &[]
            } else {
                unsafe { std::slice::from_raw_parts(p, len) }
            };
            self.data = Some(load(bytes)?);
            Ok(())
        })
    }
    #[julia]
    pub fn npy_duplicate(&self, tag: i32, m: usize, n: usize) -> Result<Self, String> {
        guarded(|| {
            self.data()?.check(tag, m, n)?;
            Ok(self.clone())
        })
    }
    #[julia]
    pub fn npy_info(&self, out: *mut usize) -> Result<(), String> {
        guarded(|| {
            let a = self.data()?;
            let info = [
                a.tag as usize,
                a.m,
                a.n,
                a.rank,
                a.fortran as usize,
                a.version.0 as usize,
                a.version.1 as usize,
                a.endian,
                a.npy()?.is_aligned() as usize,
            ];
            for (i, v) in info.into_iter().enumerate() {
                unsafe {
                    out.add(i).write(v);
                }
            }
            Ok(())
        })
    }
    #[julia]
    pub fn npy_export(&self, out: *mut u8, tag: i32, m: usize, n: usize) -> Result<(), String> {
        guarded(|| {
            let a = self.data()?;
            a.check(tag, m, n)?;
            unsafe fn run<T: FromNpy>(a: &Data, out: *mut u8) -> Result<(), String> {
                let npy = a.npy()?;
                let mat = npy.as_aligned_ref::<T>();
                for j in 0..a.n {
                    for i in 0..a.m {
                        out.cast::<T>().add(i + j * a.m).write(mat[(i, j)]);
                    }
                }
                Ok(())
            }
            unsafe {
                match tag {
                    0 => run::<f32>(a, out),
                    1 => run::<f64>(a, out),
                    2 => run::<faer::c32>(a, out),
                    3 => run::<faer::c64>(a, out),
                    _ => Err("invalid NPY scalar".into()),
                }
            }
        })
    }
    #[julia]
    pub fn npy_element(
        &self,
        i: usize,
        j: usize,
        out: *mut u8,
        tag: i32,
        m: usize,
        n: usize,
    ) -> Result<(), String> {
        guarded(|| {
            let a = self.data()?;
            a.check(tag, m, n)?;
            if i >= m || j >= n {
                return Err("NPY index out of bounds".into());
            }
            unsafe fn run<T: FromNpy>(
                a: &Data,
                i: usize,
                j: usize,
                out: *mut u8,
            ) -> Result<(), String> {
                let npy = a.npy()?;
                out.cast::<T>().write(npy.as_aligned_ref::<T>()[(i, j)]);
                Ok(())
            }
            unsafe {
                match tag {
                    0 => run::<f32>(a, i, j, out),
                    1 => run::<f64>(a, i, j, out),
                    2 => run::<faer::c32>(a, i, j, out),
                    3 => run::<faer::c64>(a, i, j, out),
                    _ => Err("invalid NPY scalar".into()),
                }
            }
        })
    }
}
fn copied<T: FromNpy + ComplexField>(a: &Data) -> Result<Mat<T>, String> {
    Ok(a.npy()?.to_mat::<T>())
}
#[julia]
impl OwnedMatrix {
    #[julia]
    pub fn npy_copy_from(&mut self, source: &OwnedNpy, m: usize, n: usize) -> Result<(), String> {
        guarded(|| {
            let a = source.data()?;
            a.check(1, m, n)?;
            self.matrix = copied::<f64>(a)?;
            Ok(())
        })
    }
}
#[julia]
impl TypedMatrix {
    #[julia]
    pub fn npy_copy_from(
        &mut self,
        source: &OwnedNpy,
        tag: i32,
        m: usize,
        n: usize,
    ) -> Result<(), String> {
        guarded(|| {
            let a = source.data()?;
            a.check(tag, m, n)?;
            self.data = Some(match tag {
                0 => MatrixData::F32(copied(a)?),
                1 => MatrixData::F64(copied(a)?),
                2 => MatrixData::C32(copied(a)?),
                3 => MatrixData::C64(copied(a)?),
                _ => return Err("invalid NPY scalar".into()),
            });
            Ok(())
        })
    }
}
