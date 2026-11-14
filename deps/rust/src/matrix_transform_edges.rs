//! Direct ABI edge tests that require no large allocation.
#[test]
fn empty_ilp64_transforms_preserve_the_full_dimension_width() {
    unsafe {
        let large = 1_i64 << 40;
        crate::matrix_copy::dlacpy_64_(
            b"A".as_ptr(),
            &0,
            &large,
            std::ptr::null(),
            &1,
            std::ptr::null_mut(),
            &1,
        );
        crate::complex_auxiliary::zlacp2_64_(
            b"A".as_ptr(),
            &0,
            &large,
            std::ptr::null(),
            &1,
            std::ptr::null_mut(),
            &1,
        );
        let mut info = [777_i64; 3];
        crate::precision_conversion::dlag2s_64_(
            &0,
            &large,
            std::ptr::null(),
            &1,
            std::ptr::null_mut(),
            &1,
            info.as_mut_ptr().add(1),
        );
        assert_eq!(info, [777, 0, 777]);
    }
}
#[test]
fn unrepresentable_band_leading_dimensions_report_info_without_unwinding() {
    unsafe {
        let mut info = [777_i64; 3];
        let output = info.as_mut_ptr().add(1);
        crate::positive_equilibration::dpbequ_64_(
            b"U".as_ptr(),
            &0,
            &i64::MAX,
            std::ptr::null(),
            &1,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            output,
        );
        assert_eq!(info, [777, -5, 777]);
        crate::general_equilibration::dgbequ_64_(
            &0,
            &0,
            &i64::MAX,
            &0,
            std::ptr::null(),
            &i64::MAX,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            output,
        );
        assert_eq!(info, [777, -6, 777]);
        crate::matrix_scaling::dlascl_64_(
            b"Z".as_ptr(),
            &(i64::MAX - 1),
            &0,
            &1.0,
            &2.0,
            &i64::MAX,
            &0,
            std::ptr::null_mut(),
            &i64::MAX,
            output,
        );
        assert_eq!(info, [777, -9, 777]);
        assert_eq!(crate::colmat_blas_last_error(), 9);
    }
}
