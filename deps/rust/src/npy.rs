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
/// Validated source metadata. faer's `Npy::new` does not check payload length,
/// rank or dimension overflow, so this always runs before faer sees the bytes.
struct Meta {
    m: usize,
    n: usize,
    rank: usize,
    tag: i32,
    fortran: bool,
    version: (u8, u8),
    endian: usize,
    prefix: usize,
    payload: usize,
    component: usize,
    descr: &'static str,
}
impl Meta {
    fn check(&self, tag: i32, m: usize, n: usize) -> Result<(), String> {
        if self.tag != tag || self.m != m || self.n != n {
            return Err("NPY source metadata does not match the requested matrix".into());
        }
        Ok(())
    }
}
// Must run on the parser_stack thread (npyz header parsing uses pest).
fn validate(bytes: &[u8]) -> Result<Meta, String> {
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
    let file = npyz::NpyFile::new(bytes).map_err(|e| format!("invalid NPY: {e}"))?;
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
    Ok(Meta {
        m,
        n,
        rank,
        tag,
        fortran,
        version,
        endian,
        prefix,
        payload,
        component,
        descr,
    })
}
/// Snapshot for the read-only view. `Npy::as_aligned_ref` asserts a 64-byte
/// aligned buffer and native byte order, so the view keeps this normalized
/// copy; the copying path (`direct`) reads the source bytes in place instead.
fn load(bytes: &[u8]) -> Result<Data, String> {
    let invalid = |s: &str| format!("invalid NPY: {s}");
    let Meta {
        m,
        n,
        rank,
        tag,
        fortran,
        version,
        endian,
        prefix,
        payload,
        component,
        descr,
    } = parser_stack(|| validate(bytes))??;
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
/// Copies source bytes straight into a native matrix through faer's
/// `Npy::to_mat`, which reads unaligned and foreign-endian payloads since
/// tensor4all/faer#215 (fixes tensor4all/faer#210 and tensor4all/faer#211).
fn direct<T: FromNpy + ComplexField>(
    bytes: &[u8],
    tag: i32,
    m: usize,
    n: usize,
) -> Result<Mat<T>, String> {
    let npy = parser_stack(|| -> Result<Npy<'_>, String> {
        validate(bytes)?.check(tag, m, n)?;
        Npy::new(bytes).map_err(|e| format!("invalid NPY: {e}"))
    })??;
    if npy.dtype() != T::DTYPE {
        return Err("NPY dtype does not match the requested scalar".into());
    }
    Ok(npy.to_mat::<T>())
}
fn source<'a>(p: *const u8, len: usize) -> &'a [u8] {
    if len == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(p, len) }
    }
}
/// Writes validated source metadata without copying the payload:
/// [tag, m, n, rank, fortran, version major, version minor, endian].
#[julia]
pub fn npy_source_info(p: *const u8, len: usize, out: &mut [usize]) -> Result<(), String> {
    let out = crate::workspace::fixed_out(out, 8)?;
    guarded(|| {
        let bytes = source(p, len);
        let a = parser_stack(|| validate(bytes))??;
        let info = [
            a.tag as usize,
            a.m,
            a.n,
            a.rank,
            a.fortran as usize,
            a.version.0 as usize,
            a.version.1 as usize,
            a.endian,
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
            self.data = Some(load(source(p, len))?);
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
    pub fn npy_info(&self, out: &mut [usize]) -> Result<(), String> {
        let out = crate::workspace::fixed_out(out, 9)?;
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
    #[julia]
    pub fn npy_load_direct(
        &mut self,
        p: *const u8,
        len: usize,
        m: usize,
        n: usize,
    ) -> Result<(), String> {
        guarded(|| {
            self.matrix = direct::<f64>(source(p, len), 1, m, n)?;
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
    #[julia]
    pub fn npy_load_direct(
        &mut self,
        p: *const u8,
        len: usize,
        tag: i32,
        m: usize,
        n: usize,
    ) -> Result<(), String> {
        guarded(|| {
            let bytes = source(p, len);
            self.data = Some(match tag {
                0 => MatrixData::F32(direct(bytes, tag, m, n)?),
                1 => MatrixData::F64(direct(bytes, tag, m, n)?),
                2 => MatrixData::C32(direct(bytes, tag, m, n)?),
                3 => MatrixData::C64(direct(bytes, tag, m, n)?),
                _ => return Err("invalid NPY scalar".into()),
            });
            Ok(())
        })
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use faer::{c32, c64};
    use std::fmt::Debug;
    use std::panic::{catch_unwind, AssertUnwindSafe};
    trait Fixture: FromNpy + ComplexField + PartialEq + Debug {
        const TAG: i32;
        const DESCR: &'static str;
        const REAL: usize;
        fn value(k: usize) -> Self;
        fn components(&self) -> Vec<f64>;
    }
    impl Fixture for f32 {
        const TAG: i32 = 0;
        const DESCR: &'static str = "f4";
        const REAL: usize = 4;
        fn value(k: usize) -> Self {
            k as f32 + 0.5
        }
        fn components(&self) -> Vec<f64> {
            vec![*self as f64]
        }
    }
    impl Fixture for f64 {
        const TAG: i32 = 1;
        const DESCR: &'static str = "f8";
        const REAL: usize = 8;
        fn value(k: usize) -> Self {
            k as f64 + 0.5
        }
        fn components(&self) -> Vec<f64> {
            vec![*self]
        }
    }
    impl Fixture for c32 {
        const TAG: i32 = 2;
        const DESCR: &'static str = "c8";
        const REAL: usize = 4;
        fn value(k: usize) -> Self {
            c32::new(k as f32 + 0.5, -(k as f32) - 0.25)
        }
        fn components(&self) -> Vec<f64> {
            vec![self.re as f64, self.im as f64]
        }
    }
    impl Fixture for c64 {
        const TAG: i32 = 3;
        const DESCR: &'static str = "c16";
        const REAL: usize = 8;
        fn value(k: usize) -> Self {
            c64::new(k as f64 + 0.5, -(k as f64) - 0.25)
        }
        fn components(&self) -> Vec<f64> {
            vec![self.re, self.im]
        }
    }
    // Independent encoder: explicit component bytes in the requested order.
    fn encode<T: Fixture>(m: usize, n: usize, fortran: bool, big: bool, version: u8) -> Vec<u8> {
        let marker = if big { '>' } else { '<' };
        let mut header = format!(
            "{{'descr': '{marker}{}', 'fortran_order': {}, 'shape': ({m}, {n}), }}",
            T::DESCR,
            if fortran { "True" } else { "False" }
        );
        let start = if version == 1 { 10 } else { 12 };
        header.push_str(&" ".repeat((64 - ((start + header.len() + 1) % 64)) % 64));
        header.push('\n');
        let mut out = vec![0x93, b'N', b'U', b'M', b'P', b'Y', version, 0];
        if version == 1 {
            out.extend_from_slice(&(header.len() as u16).to_le_bytes());
        } else {
            out.extend_from_slice(&(header.len() as u32).to_le_bytes());
        }
        out.extend_from_slice(header.as_bytes());
        let mut push = |x: f64| {
            let mut word = if T::REAL == 4 {
                (x as f32).to_le_bytes().to_vec()
            } else {
                x.to_le_bytes().to_vec()
            };
            if big {
                word.reverse();
            }
            out.extend_from_slice(&word);
        };
        let order: Vec<(usize, usize)> = if fortran {
            (0..n).flat_map(|j| (0..m).map(move |i| (i, j))).collect()
        } else {
            (0..m).flat_map(|i| (0..n).map(move |j| (i, j))).collect()
        };
        for (i, j) in order {
            T::value(i + j * m)
                .components()
                .into_iter()
                .for_each(&mut push);
        }
        out
    }
    // Places a copy of the source at a chosen offset from a 64-byte boundary.
    struct Placed {
        blocks: Vec<Block>,
        offset: usize,
        len: usize,
    }
    impl Placed {
        fn new(file: &[u8], offset: usize) -> Self {
            let mut blocks = vec![Block([0; 64]); (file.len() + offset).div_ceil(64) + 1];
            let all = unsafe {
                std::slice::from_raw_parts_mut(blocks.as_mut_ptr().cast::<u8>(), blocks.len() * 64)
            };
            all[offset..offset + file.len()].copy_from_slice(file);
            Self {
                blocks,
                offset,
                len: file.len(),
            }
        }
        fn bytes(&self) -> &[u8] {
            unsafe {
                std::slice::from_raw_parts(
                    self.blocks.as_ptr().cast::<u8>().add(self.offset),
                    self.len,
                )
            }
        }
    }
    fn direct_case<T: Fixture>() {
        let native_big = cfg!(target_endian = "big");
        for (m, n) in [(2, 3), (3, 1), (1, 1), (0, 3), (3, 0)] {
            for fortran in [false, true] {
                for big in [false, true] {
                    for version in 1..=3 {
                        for offset in [0, 1, 3, 4, 8, 17] {
                            let file = encode::<T>(m, n, fortran, big, version);
                            let placed = Placed::new(&file, offset);
                            let src = placed.bytes();
                            let npy = Npy::new(src).unwrap();
                            assert_eq!(npy.is_aligned(), offset == 0);
                            assert_eq!(npy.is_native_endian(), big == native_big);
                            let mat = direct::<T>(src, T::TAG, m, n).unwrap();
                            assert_eq!(mat.shape(), (m, n));
                            for j in 0..n {
                                for i in 0..m {
                                    assert_eq!(mat[(i, j)], T::value(i + j * m));
                                }
                            }
                            // Stale Julia metadata must never reach to_mat.
                            assert!(direct::<T>(src, T::TAG, m + 1, n).is_err());
                            assert!(direct::<T>(src, (T::TAG + 1) % 4, m, n).is_err());
                        }
                    }
                }
            }
        }
    }
    // tensor4all/faer#210 and #211 are fixed upstream (PR #215, in the pin):
    // to_mat reads unaligned and foreign-endian sources without normalization.
    #[test]
    fn direct_to_mat_reads_foreign_endian_and_unaligned_sources() {
        direct_case::<f32>();
        direct_case::<f64>();
        direct_case::<c32>();
        direct_case::<c64>();
    }
    #[test]
    fn direct_rejects_invalid_sources_before_faer() {
        let file = encode::<f64>(2, 3, false, false, 1);
        let mut trailing = file.clone();
        trailing.push(0);
        assert!(direct::<f64>(&trailing, 1, 2, 3).is_err());
        assert!(direct::<f64>(&file[..file.len() - 1], 1, 2, 3).is_err());
        assert!(direct::<f64>(&file[..9], 1, 2, 3).is_err());
        // Well-formed headers that faer alone would accept or misread.
        let raw = |descr: &str, shape: &str| {
            let mut header =
                format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape}, }}");
            header.push_str(&" ".repeat((64 - ((10 + header.len() + 1) % 64)) % 64));
            header.push('\n');
            let mut out = b"\x93NUMPY\x01\x00".to_vec();
            out.extend_from_slice(&(header.len() as u16).to_le_bytes());
            out.extend_from_slice(header.as_bytes());
            out.extend_from_slice(&[0; 48]);
            out
        };
        assert!(direct::<f64>(&raw("<f8", "(2, 3)"), 1, 2, 3).is_ok());
        assert!(direct::<f64>(&raw("<f8", "(2, 3, 1)"), 1, 2, 3).is_err());
        assert!(direct::<f64>(&raw("<i8", "(2, 3)"), 1, 2, 3).is_err());
        assert!(direct::<f64>(&raw("<f8", "(1, 6)"), 1, 2, 3).is_err());
    }
    // The owned view still needs normalized storage: as_aligned_ref requires a
    // 64-byte aligned buffer and native byte order by design.
    #[test]
    fn aligned_view_requires_normalized_storage() {
        let native_big = cfg!(target_endian = "big");
        for (offset, big) in [(0, !native_big), (1, native_big), (8, !native_big)] {
            let file = encode::<f64>(2, 3, false, big, 1);
            let placed = Placed::new(&file, offset);
            let src = placed.bytes();
            let npy = Npy::new(src).unwrap();
            assert!(
                catch_unwind(AssertUnwindSafe(|| npy.as_aligned_ref::<f64>()[(0, 0)])).is_err()
            );
            let data = load(src).unwrap();
            let view = data.npy().unwrap();
            assert!(view.is_aligned() && view.is_native_endian());
            assert_eq!(view.as_aligned_ref::<f64>()[(1, 2)], f64::value(5));
        }
    }
    #[test]
    fn direct_path_skips_normalized_buffer_allocations() {
        let file = encode::<c64>(16, 16, false, !cfg!(target_endian = "big"), 1);
        let placed = Placed::new(&file, 1);
        let src = placed.bytes();
        let normalized = crate::allocation_probe::count(|| {
            let data = load(src).unwrap();
            let _ = copied::<c64>(&data).unwrap();
        });
        let direct = crate::allocation_probe::count(|| {
            let _ = direct::<c64>(src, 3, 16, 16).unwrap();
        });
        eprintln!("npy allocations: normalized+to_mat={normalized}, direct={direct}");
        assert!(direct < normalized);
    }
}
