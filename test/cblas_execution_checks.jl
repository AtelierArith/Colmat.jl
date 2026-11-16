using Test

@testset "All #271 targets completed through both ABIs and both routes" begin
    families=(
        CBLAS_271_LEVEL1,
        CBLAS_271_GEMM_TARGETS,
        CBLAS_271_MATVEC_TARGETS,
        CBLAS_271_SYMMETRIC_TARGETS,
        CBLAS_271_TRIANGULAR_TARGETS,
        CBLAS_271_RANK_UPDATE_TARGETS,
        CBLAS_271_RANKK_TARGETS,
        CBLAS_271_MATRIX_TARGETS,
    )
    combined=vcat(families...)
    @test length(combined)==172
    @test length(unique(combined))==172
    @test Set(combined)==Set(CBLAS_271_TARGETS)
    @test length(CBLAS_271_CALLS)==172*2*2
    for bits in (32, 64), route in (:dll, :lbt_trampoline)
        @test Set(
            name for (name, width, actual_route) in CBLAS_271_CALLS if
            width==bits&&actual_route==route
        )==Set(CBLAS_271_TARGETS)
    end
    for (name, bits, route, file, line, signature) in CBLAS_271_CALL_SITES
        @test (name, bits, route) in CBLAS_271_CALLS
        @test isfile(file)
        @test 1<=line<=countlines(file)
        @test !isempty(signature)
    end
end
