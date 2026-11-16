using Test, Colmat, LinearAlgebra, Libdl

# Frozen canonical scope of #271; excludes the six pre-existing CBLAS dots.
const CBLAS_271_TARGETS = split(raw"""
cblas_caxpby cblas_caxpy cblas_ccopy cblas_cdotc cblas_cdotu cblas_cgbmv
cblas_cgeadd cblas_cgemm cblas_cgemm3m cblas_cgemmt cblas_cgemmtr cblas_cgemv
cblas_cgerc cblas_cgeru cblas_chbmv cblas_chemm cblas_chemv cblas_cher
cblas_cher2 cblas_cher2k cblas_cherk cblas_chpmv cblas_chpr cblas_chpr2
cblas_cimatcopy cblas_comatcopy cblas_crotg cblas_cscal cblas_csrot cblas_csscal
cblas_cswap cblas_csymm cblas_csyr2k cblas_csyrk cblas_ctbmv cblas_ctbsv
cblas_ctpmv cblas_ctpsv cblas_ctrmm cblas_ctrmv cblas_ctrsm cblas_ctrsv
cblas_damax cblas_damin cblas_dasum cblas_daxpby cblas_daxpy cblas_dcopy
cblas_dgbmv cblas_dgeadd cblas_dgemm cblas_dgemmt cblas_dgemmtr cblas_dgemv
cblas_dger cblas_dimatcopy cblas_dnrm2 cblas_domatcopy cblas_drot cblas_drotg
cblas_dsbmv cblas_dscal cblas_dspmv cblas_dspr cblas_dspr2 cblas_dsum
cblas_dswap cblas_dsymm cblas_dsymv cblas_dsyr cblas_dsyr2 cblas_dsyr2k
cblas_dsyrk cblas_dtbmv cblas_dtbsv cblas_dtpmv cblas_dtpsv cblas_dtrmm
cblas_dtrmv cblas_dtrsm cblas_dtrsv cblas_dzasum cblas_dznrm2 cblas_icamax
cblas_icamin cblas_idamax cblas_idamin cblas_isamax cblas_isamin cblas_izamax
cblas_izamin cblas_samax cblas_samin cblas_sasum cblas_saxpby cblas_saxpy
cblas_scasum cblas_scnrm2 cblas_scopy cblas_sgbmv cblas_sgeadd cblas_sgemm
cblas_sgemmt cblas_sgemmtr cblas_sgemv cblas_sger cblas_simatcopy cblas_snrm2
cblas_somatcopy cblas_srot cblas_srotg cblas_ssbmv cblas_sscal cblas_sspmv
cblas_sspr cblas_sspr2 cblas_ssum cblas_sswap cblas_ssymm cblas_ssymv
cblas_ssyr cblas_ssyr2 cblas_ssyr2k cblas_ssyrk cblas_stbmv cblas_stbsv
cblas_stpmv cblas_stpsv cblas_strmm cblas_strmv cblas_strsm cblas_strsv
cblas_zaxpby cblas_zaxpy cblas_zcopy cblas_zdotc cblas_zdotu cblas_zdscal
cblas_zgbmv cblas_zgeadd cblas_zgemm cblas_zgemm3m cblas_zgemmt cblas_zgemmtr
cblas_zgemv cblas_zgerc cblas_zgeru cblas_zhbmv cblas_zhemm cblas_zhemv
cblas_zher cblas_zher2 cblas_zher2k cblas_zherk cblas_zhpmv cblas_zhpr
cblas_zhpr2 cblas_zimatcopy cblas_zomatcopy cblas_zscal cblas_zswap cblas_zsymm
cblas_zsyr2k cblas_zsyrk cblas_ztbmv cblas_ztbsv cblas_ztpmv cblas_ztpsv
cblas_ztrmm cblas_ztrmv cblas_ztrsm cblas_ztrsv
""")

@testset "CBLAS #271 native LBT registration" begin
    @test length(CBLAS_271_TARGETS) == 172
    @test length(unique(CBLAS_271_TARGETS)) == 172
    library = Libdl.dlopen(Colmat.backend_library())
    try
        for (interface, suffix) in ((:lp64, ""), (:ilp64, "64_"))
            for symbol in CBLAS_271_TARGETS
                expected = Libdl.dlsym(library, symbol * suffix)
                @test BLAS.lbt_get_forward(symbol, interface) == expected
            end
        end
    finally
        Libdl.dlclose(library)
    end
end
