using Test, Colmat, LinearAlgebra, SparseArrays

function _cond_sample(T, n)
    A=spdiagm(0=>T.(4 .+ (1:n)), 1=>T.(fill(1, n-1)), -2=>T.(fill(-2, n-2)))
    A[1, n]=T(3)
    T<:Complex && (A.+=T(0.5im).*spdiagm(-1=>T.(1:(n-1)), 2=>T.(fill(1, n-2))))
    return A
end

@testset "Estimated sparse 1/Inf condition numbers" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        format in (:csc, :csr),
        n in (1, 2, 6, 40)

        A=n==1 ? sparse(T[3;;]) : _cond_sample(T, n)
        a=FaerSparseMatrix(A; format)
        R=real(T)
        for p in (1, Inf)
            @test opnorm(a, p) isa R
            @test opnorm(a, p)≈opnorm(A, p) rtol=sqrt(eps(R))
            exact=cond(Matrix(A), p)
            est=cond(a, p)
            @test est isa R
            # Higham–Tisseur estimates are lower bounds of the exact value.
            @test est<=exact*(1+sqrt(eps(R)))
            @test est>=exact/n
            # Tiny inputs are estimated exactly; larger ones may underestimate
            # (as SparseArrays does for some random starts).
            n<=2 && @test est≈exact rtol=sqrt(eps(R))
            # Deterministic: repeated estimates agree.
            @test cond(a, p)==est
        end
        @test Matrix(a)==Matrix(A)
        close(a)
    end
end
@testset "Sparse condition number edge cases follow Colmat dense conventions" begin
    for T in (Float64, ComplexF32)
        R=real(T)
        e=FaerSparseMatrix(spzeros(T, 0, 0))
        @test cond(e, 1)===zero(R)
        @test cond(e, Inf)===zero(R)
        s=FaerSparseMatrix(sparse(T[1 2; 2 4]))
        @test cond(s, 1)===R(Inf)
        @test cond(s, Inf)===R(Inf)
        @test Matrix(s)==T[1 2; 2 4]
        z=FaerSparseMatrix(spzeros(T, 3, 3))
        @test cond(z, 1)===R(Inf)
        a=FaerSparseMatrix(_cond_sample(T, 4))
        # As for SparseArrays: no sparse 2-norm condition number or operator norm.
        @test_throws ArgumentError cond(a)
        @test_throws ArgumentError cond(a, 2)
        @test_throws ArgumentError cond(a, 3)
        @test_throws ArgumentError opnorm(a, 2)
        r=FaerSparseMatrix(sparse(T[1 2 3; 4 5 6]))
        @test_throws DimensionMismatch cond(r, 1)
        @test opnorm(r, 1)≈opnorm(sparse(T[1 2 3; 4 5 6]), 1)
        @test opnorm(r, Inf)≈opnorm(sparse(T[1 2 3; 4 5 6]), Inf)
        foreach(close, (e, s, z, a, r))
        @test_throws ArgumentError cond(a, 1)
        @test_throws ArgumentError opnorm(a, 1)
    end
end
