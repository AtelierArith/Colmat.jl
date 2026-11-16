use super::*;

#[test]
fn hermitian_rank_noops_preserve_even_ignored_diagonal_components() {
    let zero = c64::new(0.0, 0.0);
    let one = c64::new(1.0, 0.0);
    let mut c = [c64::new(2.0, f64::NAN)];
    unsafe {
        cblas_zherk64_(
            102,
            121,
            111,
            1,
            1,
            0.0,
            core::ptr::null(),
            1,
            1.0,
            c.as_mut_ptr(),
            1,
        );
        assert_eq!(c[0].re, 2.0);
        assert!(c[0].im.is_nan());
        cblas_zher2k64_(
            102,
            121,
            111,
            1,
            1,
            &zero,
            core::ptr::null(),
            1,
            core::ptr::null(),
            1,
            1.0,
            c.as_mut_ptr(),
            1,
        );
        assert_eq!(c[0].re, 2.0);
        assert!(c[0].im.is_nan());
        cblas_zher2k64_(
            102,
            121,
            111,
            1,
            0,
            &one,
            core::ptr::null(),
            1,
            core::ptr::null(),
            1,
            1.0,
            c.as_mut_ptr(),
            1,
        );
        assert_eq!(c[0].re, 2.0);
        assert!(c[0].im.is_nan());
    }
}

#[test]
fn true_noops_need_no_array_storage_or_negative_stride_pointer_arithmetic() {
    unsafe {
        crate::kernels::gemv::<c64>(
            b'N',
            3,
            2,
            c64::z(),
            core::ptr::null(),
            3,
            core::ptr::null(),
            -2,
            c64::o(),
            core::ptr::null_mut(),
            -3,
        );
        crate::kernels::symv_with_conjugation::<c64>(
            b'U',
            3,
            c64::z(),
            core::ptr::null(),
            3,
            core::ptr::null(),
            -2,
            c64::o(),
            core::ptr::null_mut(),
            -3,
            true,
            true,
        );
        crate::kernels::rankk::<c64>(
            b'U',
            b'N',
            3,
            4,
            c64::z(),
            core::ptr::null(),
            3,
            core::ptr::null(),
            3,
            c64::o(),
            core::ptr::null_mut(),
            3,
            true,
            true,
        );
        crate::kernels::rankk::<f32>(
            b'U',
            b'N',
            3,
            0,
            1.0,
            core::ptr::null(),
            3,
            core::ptr::null(),
            3,
            1.0,
            core::ptr::null_mut(),
            3,
            false,
            false,
        );
    }
}
