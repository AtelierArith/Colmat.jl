using Test, Colmat, LinearAlgebra, SparseArrays

@testset "Native block BiCGSTAB" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        R=typeof(real(zero(T)))
        tolerance=1024eps(R)
        A=T[5 2 0; -1 4 1; 0 -1 3]
        T <: Complex && (A[1, 2]+=T(0.5im))
        b=T[1, 2, 3]
        for source in (
            FaerMatrix(A),
            FaerSparseMatrix(sparse(A)),
            FaerSparseMatrix(sparse(A); format = :csr),
        )
            op=FaerLinearOperator(source)
            for M in (nothing, diagonal_preconditioner(diag(A)))
                for N in (nothing, diagonal_preconditioner(diag(A)))
                    result=bicgstab(op, b; M, N)
                    @test result isa FaerBicgstabResult
                    @test issuccess(result)
                    @test result.x ≈ A\b rtol=tolerance
                    @test result.abs_residual ≈ norm(b-A*result.x) atol=tolerance
                    @test result.rel_residual ≈ norm(b-A*result.x)/norm(b) atol=tolerance
                    @test result.iterations > 0
                end
            end
            @test bicgstab(op, b; x0 = A\b).iterations==0
            @test iszero(bicgstab(op, zeros(T, 3)).x)
            native=bicgstab(op, FaerVector(b))
            @test native.x isa FaerVector{T}
            @test Vector(native.x) ≈ A\b rtol=tolerance
            B=T[1 0; 2 1; 3 -1]
            block=bicgstab(op, B; maxiter = 12, rtol = R(1e-4))
            @test Matrix(block.x) ≈ A\B rtol=R(1e-3)
            dest=zeros(T, 3)
            solved=bicgstab!(dest, op, b)
            @test solved.x===dest
            @test dest ≈ A\b rtol=tolerance
            fill!(dest, 0)
            @test_throws FaerBicgstabError bicgstab!(dest, op, b; maxiter = 0)
            @test iszero(dest)
            failed=bicgstab!(dest, op, b; maxiter = 0, check = false)
            @test !issuccess(failed)
            @test failed.reason==:no_convergence
            @test failed.iterations==0
            @test failed.x===dest
        end
        @test_throws DimensionMismatch bicgstab(A, b; N = FaerLinearOperator(I, 2; T))
        @test_throws DimensionMismatch bicgstab(A, b; M = FaerLinearOperator(I, 2; T))
        @test_throws ArgumentError bicgstab(A, b; atol = -1)
        @test_throws ArgumentError bicgstab(A, b; rtol = NaN)
        @test_throws ArgumentError bicgstab(A, b; maxiter = -1)
        @test_throws ArgumentError bicgstab(A, fill(T(NaN), 3))
    end
    mixed = bicgstab(Float32[4 1; -1 3], ComplexF64[1+im, 2])
    @test eltype(mixed.x) === ComplexF64
    @test mixed.x ≈ Float32[4 1; -1 3] \ ComplexF64[1+im, 2]
    A = [5.0 2 0; -1 4 1; 0 -1 3]
    alias = [1.0, 2, 3]
    expected = A \ copy(alias)
    @test bicgstab!(alias, A, alias).x === alias
    @test alias ≈ expected
    closed = identity_preconditioner(3)
    close(closed)
    @test_throws ArgumentError bicgstab(A, ones(3); N = closed)
    @test_throws ArgumentError bicgstab(A, ones(3); x0 = fill(Inf, 3))
    @test_throws ArgumentError cg(A, fill(NaN, 3))
    @test_throws ArgumentError cg(A, ones(3); x0 = fill(Inf, 3))
    n=50_000
    d=collect(range(2.0, 4.0; length = n))
    op=FaerLinearOperator(spdiagm(0=>d))
    result=bicgstab(op, ones(n); N = diagonal_preconditioner(d), maxiter = 2)
    @test result.converged
    @test result.x ≈ inv.(d)
    @test_throws DimensionMismatch bicgstab(ones(3, 2), ones(3))
    @test_throws DimensionMismatch bicgstab(Matrix{Float64}(I, 3, 3), ones(2))
    @test isempty(
        Test.detect_ambiguities(
            Colmat,
            Base,
            LinearAlgebra,
            SparseArrays;
            recursive = true,
        ),
    )
end
