//! BLAS/LAPACK ABI adapter for Julia's libblastrampoline.
//!
//! All arrays are borrowed for the duration of a call. No Julia runtime or RustCall
//! is involved. Unexported BLAS/LAPACK symbols stay on the previous LBT backend.
#![allow(clippy::too_many_arguments, clippy::missing_safety_doc)]

mod abi;
mod advanced_svd;
mod balance;
mod band_cholesky;
mod band_lu;
mod bidiagonal;
mod bidiagonal_panel;
mod bidiagonal_vectors;
mod block_unitary;
mod bunch_kaufman;
mod complete_pivot_lu;
mod complex_auxiliary;
mod dense_extensions;
mod general_equilibration;
mod generalized_eigen;
mod generalized_hessenberg;
mod generalized_selfadjoint;
mod generalized_selfadjoint_drivers;
mod givens;
mod hessenberg;
mod householder;
mod kernels;
mod lapack;
mod lapack_symmetric;
mod least_squares;
mod lq;
mod matrix_copy;
mod matrix_extensions;
mod matrix_norms;
mod matrix_permutation;
mod matrix_scaling;
#[cfg(test)]
mod matrix_transform_edges;
mod packed_cholesky;
mod packed_updates;
mod pivoted_cholesky;
mod positive_equilibration;
mod positive_tridiagonal;
mod precision_conversion;
mod qr;
mod qr_pivoted;
mod qz;
mod reduction_extensions;
mod rfp_rank;
mod rz;
mod schur;
mod selected_selfadjoint;
mod selfadjoint_reduction;
mod selfadjoint_vectors;
mod storage_conversions;
mod storage_layout;
mod structured_blas;
mod structured_evd;
mod structured_selected_evd;
mod structured_triangular;
mod structured_triangular_lapack;
mod symmetric_equilibration;
mod threading;
mod triangular_lapack;
mod tridiagonal_evd;
mod tridiagonal_lu;

use std::cell::Cell;

// BLAS cannot return Rust exceptions. Keep invalid-argument diagnostics queryable
// without unwinding, allocating an error string, or touching Julia's runtime.
thread_local! { static LAST_ERROR: Cell<i32> = const { Cell::new(0) }; }
fn invalid(argument: i32) {
    LAST_ERROR.with(|e| e.set(argument));
}

#[no_mangle]
pub extern "C" fn colmat_blas_last_error() -> i32 {
    LAST_ERROR.with(|e| e.replace(0))
}

#[no_mangle]
pub extern "C" fn colmat_blas_abi_version() -> i32 {
    2
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernels::*;
    use std::ptr::{null, null_mut};

    #[test]
    fn gemm_does_not_read_inputs_for_zero_alpha_or_empty_product() {
        unsafe {
            let mut c = [f64::NAN; 6];
            gemm(
                b'N',
                b'N',
                2,
                3,
                4,
                0.0,
                null(),
                2,
                null(),
                4,
                0.0,
                c.as_mut_ptr(),
                2,
            );
            assert_eq!(c, [0.0; 6]);
            c.fill(2.0);
            gemm(
                b'N',
                b'N',
                2,
                3,
                0,
                1.0,
                null(),
                2,
                null(),
                1,
                3.0,
                c.as_mut_ptr(),
                2,
            );
            assert_eq!(c, [6.0; 6]);
            gemm::<f64>(
                b'N',
                b'N',
                0,
                3,
                0,
                1.0,
                null(),
                1,
                null(),
                1,
                1.0,
                null_mut(),
                1,
            );
        }
    }

    #[test]
    fn reductions_are_scaled_and_do_not_access_empty_buffers() {
        unsafe {
            assert_eq!(nrm2::<f64>(0, null(), 1), 0.0);
            assert_eq!(iamax::<f32>(-1, null(), 1), 0);
            assert_eq!(dot::<f64>(0, null(), 0, null(), 0, false), 0.0);
            let huge = [3e200, 4e200];
            assert!((nrm2(2, huge.as_ptr(), 1) / 5e200 - 1.0).abs() < 1e-15);
            let tiny = [3e-200, 4e-200];
            assert!((nrm2(2, tiny.as_ptr(), 1) / 5e-200 - 1.0).abs() < 1e-15);
        }
    }

    #[test]
    fn invalid_arguments_leave_destination_untouched() {
        unsafe {
            let mut c = [42.0f64; 4];
            abi::dgemm_(
                b"?".as_ptr(),
                b"N".as_ptr(),
                &2,
                &2,
                &2,
                &1.0,
                null(),
                &2,
                null(),
                &2,
                &0.0,
                c.as_mut_ptr(),
                &2,
            );
            assert_eq!(colmat_blas_last_error(), 1);
            assert_eq!(c, [42.0; 4]);
            assert_eq!(colmat_blas_last_error(), 0);
            // Integer autodetection in LBT must not truncate a negative ILP64 N.
            assert_eq!(abi::isamax_64_(&(-4294967293i64), null(), &1), 0);
        }
    }

    #[test]
    fn triangular_zero_alpha_does_not_read_a_or_b() {
        unsafe {
            let mut b = [f64::NAN; 6];
            trmatrix(
                b'L',
                b'U',
                b'N',
                b'N',
                2,
                3,
                0.0,
                null(),
                2,
                b.as_mut_ptr(),
                2,
                true,
            );
            assert_eq!(b, [0.0; 6]);
        }
    }
}
