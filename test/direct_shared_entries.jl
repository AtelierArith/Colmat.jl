using Test, LinearAlgebra, Random, Colmat
isdefined(@__MODULE__, :direct_geqrt!) || include("qr_compact.jl")
isdefined(@__MODULE__, :direct_triangular_matrix!) || include("triangular_lapack.jl")

# DLL exports can be folded to one address by the compiler. Resolve each LBT
# stub explicitly so execution evidence still identifies the requested entry.
@testset "Named LBT entries for shared native factor implementations" begin
    Random.seed!(31724)
    library=BLAS.libblastrampoline
    for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
        for variant in (2, 3)
            m, n=7, 4
            original=randn(T, m, n)
            a=vcat(original, fill(T(NaN), 2, n))
            t, info=direct_geqrt!(a, m, n, n, J; variant, library)
            @test info==0
            @test all(isnan, a[(m + 1):end, :])
            @test all(isnan, t[(n + 1):end, :])
            q, info=direct_gemqrt!(
                'L', 'N', m, m, n, n, a, t, Matrix{T}(I, m, m), J; library
            )
            @test info==0
            @test q[:, 1:n]*triu(a[1:n, :])≈original
            @test q'*q≈Matrix{T}(I, m, m)
        end
        for family in (:lauum, :lauu2), uplo in ('L', 'U')
            a, matrix=triangular_fixture(T, 4, uplo, 'N')
            before=copy(a)
            expected=uplo=='U' ? matrix*matrix' : matrix'*matrix
            @test direct_triangular_matrix!(a, J; family, uplo, library)==0
            @test triangular_untouched(a, before, 4, uplo, 'N')
            actual=uplo=='U' ? triu(a[1:4, 1:4]) : tril(a[1:4, 1:4])
            selected=uplo=='U' ? triu(expected) : tril(expected)
            @test actual≈selected
        end
    end
end
