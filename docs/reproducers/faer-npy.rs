use faer::io::npy::Npy;
#[repr(align(64))]
struct Aligned([u8; 512]);
fn main() {
    let unaligned = std::env::args().any(|s| s == "unaligned");
    let big = if unaligned {
        cfg!(target_endian = "big")
    } else {
        cfg!(target_endian = "little")
    };
    let marker = if big { '>' } else { '<' };
    let mut header =
        format!("{{'descr': '{marker}f8', 'fortran_order': False, 'shape': (1, 1), }}");
    let padding = (64 - ((10 + header.len() + 1) % 64)) % 64;
    header.push_str(&" ".repeat(padding));
    header.push('\n');
    let mut file = b"\x93NUMPY\x01\x00".to_vec();
    file.extend_from_slice(&(header.len() as u16).to_le_bytes());
    file.extend_from_slice(header.as_bytes());
    file.extend_from_slice(&if big {
        1.5f64.to_be_bytes()
    } else {
        1.5f64.to_le_bytes()
    });
    let mut storage = Aligned([0; 512]);
    let offset = if unaligned { 1 } else { 0 };
    storage.0[offset..offset + file.len()].copy_from_slice(&file);
    let npy = Npy::new(&storage.0[offset..offset + file.len()]).unwrap();
    println!("dtype={:?}, aligned={}", npy.dtype(), npy.is_aligned());
    if npy.is_aligned() {
        println!(
            "aligned view value={:.6e}",
            npy.as_aligned_ref::<f64>()[(0, 0)]
        );
    }
    let matrix = npy.to_mat::<f64>();
    println!("copied value={:.6e}, expected=1.5", matrix[(0, 0)]);
    assert_eq!(matrix[(0, 0)], 1.5);
}
