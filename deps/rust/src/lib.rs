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
mod bidiagonal_lapack;
mod bidiagonal_panel;
mod bidiagonal_vectors;
mod block_unitary;
mod bunch_kaufman;
mod cblas;
mod complete_pivot_lu;
mod complex_auxiliary;
mod constrained_least_squares;
mod dense_extensions;
mod evd_profile;
mod geevx_expert;
mod general_equilibration;
mod generalized_eigen;
mod generalized_hessenberg;
mod generalized_qr;
mod generalized_selfadjoint;
mod generalized_selfadjoint_drivers;
mod geqr;
mod givens;
mod hessenberg;
mod householder;
mod householder_reconstruction;
mod kernels;
mod lapack;
mod lapack_symmetric;
mod lapacke;
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
mod pentagonal;
mod pentagonal_driver;
mod pivoted_cholesky;
mod positive_equilibration;
mod positive_tridiagonal;
mod precision_conversion;
mod qr;
mod qr_pivoted;
mod qz;
mod qz_reorder;
mod qz_reorder_driver;
mod reduction_extensions;
mod reverse_qr;
mod rfp_rank;
mod rz;
mod schur;
mod schur_reorder;
mod schur_vectors;
mod selected_selfadjoint;
mod selfadjoint_range;
mod selfadjoint_reduction;
mod selfadjoint_vectors;
mod storage_conversions;
mod storage_layout;
mod structured_blas;
mod structured_evd;
mod structured_generalized;
mod structured_generalized_band;
mod structured_generalized_selected;
mod structured_selected_evd;
mod structured_triangular;
mod structured_triangular_lapack;
mod structured_tridiagonal;
mod symmetric_equilibration;
mod tgevc;
mod threading;
mod triangular_lapack;
mod triangular_product;
mod tridiagonal_evd;
mod tridiagonal_lu;
mod tsqr;
mod tsqr_driver;

/// Saved LAPACK entry points grouped by the adapter that delegates to them.
/// Julia registers the previous backend by passing the group index and the
/// 0-based slot within that group. GEEVX keeps its own interface because it
/// also distinguishes the integer width.
pub(crate) mod fallbacks {
    use core::ffi::c_void;
    use std::sync::atomic::{AtomicPtr, Ordering};

    pub(crate) const GGEVX: i32 = 0;
    pub(crate) const TRIDIAGONAL: i32 = 1;
    pub(crate) const SELFADJOINT_GV: i32 = 2;
    pub(crate) const ADVANCED_SVD: i32 = 3;
    pub(crate) const SELFADJOINT_VX: i32 = 4;
    pub(crate) const SELFADJOINT_VR: i32 = 5;
    pub(crate) const TRIANGULAR_PRODUCT: i32 = 6;
    pub(crate) const STRUCTURED_EVD: i32 = 7;
    pub(crate) const QZ_REORDER: i32 = 8;
    pub(crate) const STRUCTURED_GV: i32 = 9;
    pub(crate) const SCHUR_VECTORS: i32 = 10;
    pub(crate) const BDSDC: i32 = 11;
    pub(crate) const SCHUR_EXPERT: i32 = 12;

    static GROUPS: [&[AtomicPtr<c_void>]; 13] = [
        &crate::generalized_eigen::EXPERT,
        &crate::tridiagonal_evd::FALLBACKS,
        &crate::generalized_selfadjoint_drivers::FALLBACKS,
        &crate::advanced_svd::FALLBACKS,
        &crate::selected_selfadjoint::FALLBACKS,
        &crate::selfadjoint_range::FALLBACKS,
        &crate::triangular_product::SAVED,
        &crate::structured_selected_evd::FALLBACKS,
        &crate::qz_reorder_driver::EXPERT,
        &crate::structured_generalized::FALLBACKS,
        &crate::schur_vectors::FALLBACKS,
        &crate::bidiagonal_lapack::BDSDC_FALLBACKS,
        &crate::schur_reorder::EXPERT,
    ];

    fn slot(group: i32, id: i32) -> Option<&'static AtomicPtr<c_void>> {
        GROUPS.get(group as usize)?.get(id as usize)
    }

    pub(crate) fn set(group: i32, id: i32, pointer: *mut c_void) {
        if let Some(entry) = slot(group, id) {
            entry.store(pointer, Ordering::Release);
        }
    }

    pub(crate) fn has(group: i32, id: i32) -> bool {
        slot(group, id).is_some_and(|entry| !entry.load(Ordering::Acquire).is_null())
    }

    #[no_mangle]
    pub extern "C" fn colmat_fallback_set(group: i32, id: i32, pointer: *mut c_void) {
        set(group, id, pointer);
    }

    #[no_mangle]
    pub extern "C" fn colmat_fallback_has(group: i32, id: i32) -> i32 {
        i32::from(has(group, id))
    }
}

use std::cell::Cell;

// BLAS cannot return Rust exceptions. Keep invalid-argument diagnostics queryable
// without unwinding, allocating an error string, or touching Julia's runtime.
// Positive diagnostics identify invalid arguments; -1 means a required saved
// backend is unavailable, and -2 means an external integer output is out of range.
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
    #[test]
    fn fallback_registry_rejects_unknown_groups_and_slots() {
        assert!(!fallbacks::has(-1, 0));
        assert!(!fallbacks::has(fallbacks::GGEVX, -1));
        assert!(!fallbacks::has(fallbacks::GGEVX, 4));
        assert!(!fallbacks::has(fallbacks::SCHUR_EXPERT, 8));
        // Writing an unknown slot is a no-op instead of a panic.
        fallbacks::set(-1, 0, null_mut());
        fallbacks::set(fallbacks::GGEVX, 4, null_mut());
    }
}

mod general_evd;
