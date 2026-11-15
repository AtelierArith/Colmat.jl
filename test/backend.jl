@testset "Backend installation" begin
    @test all(n -> n == :Colmat || !Base.isexported(Colmat, n), names(Colmat))
    @test isfile(Colmat.backend_library())
    for interface in (:lp64, :ilp64),
        name in ("sgemm_", "dgemm_", "cgemm_", "zgemm_", "dgemv_", "ztrsm_")

        lib = BLAS.lbt_find_backing_library(name, interface)
        @test lib !== nothing
        @test realpath(lib.libname) == realpath(Colmat.backend_library())
        @test lib.interface == interface
        @test lib.f2c == :plain
        @test lib.complex_retstyle == :normal
    end
    @test BLAS.lbt_get_forward("drotm_", :ilp64) == fallback_rotm
    @test BLAS.lbt_get_forward("dggglm_", :ilp64) == fallback_lapack
    @test Colmat.load_backend() === nothing
    @test BLAS.lbt_get_forward("dggglm_", :ilp64) == fallback_lapack
end

@testset "Thread configuration" begin
    @test Colmat.get_num_threads() == BLAS.get_num_threads()
    for n in (2, 1)
        BLAS.set_num_threads(n)
        @test Colmat.get_num_threads() == n
        @test BLAS.get_num_threads() == n
        A = reshape(Float64.(1:120), 12, 10)
        B = reshape(Float64.(1:80), 10, 8)
        @test A * B ≈ [sum(A[i, k] * B[k, j] for k in 1:10) for i in 1:12, j in 1:8]
    end
    @test_throws ArgumentError Colmat.set_num_threads(0)
    @test_throws ArgumentError Colmat.set_num_threads(-1)
    @test_throws ArgumentError Colmat.set_num_threads(big(typemax(Cint)) + 1)
    @test Colmat.set_num_threads(2) === nothing
    @test Colmat.get_num_threads() == 2
    Colmat.set_num_threads(1)
end
