using Test, Colmat, LinearAlgebra, SparseArrays

@testset "Native linear operators" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        R = typeof(real(zero(T)))
        A = T[4 1; 1 3]
        b = T[1, 2]
        tol = 256eps(R)
        for source in (
            FaerMatrix(A),
            FaerSparseMatrix(sparse(A)),
            FaerSparseMatrix(sparse(A); format = :csr),
            Diagonal(T[4, 3]),
        )
            expected = Matrix(source)
            op = FaerLinearOperator(source)
            @test Matrix(op) ≈ expected
            @test op*b ≈ expected*b
            @test Matrix(op*reshape(b, 2, 1)) ≈ expected*reshape(b, 2, 1)
            @test Matrix(reshape(b, 1, 2)*op) ≈ reshape(b, 1, 2)*expected
            @test Matrix(transpose(op)) ≈ transpose(expected)
            @test Matrix(adjoint(op)) ≈ adjoint(expected)
            @test Matrix(conj(op)) ≈ conj(expected)
            out = fill(T(NaN), 2)
            mul!(out, op, b)
            @test out ≈ expected*b
            mul!(out, op, fill(T(NaN), 2), zero(T), zero(T))
            @test iszero(out)
            result = cg(op, b)
            @test issuccess(result)
            @test result.x ≈ expected\b rtol=tol
            @test result.abs_residual ≈ norm(b-expected*result.x) atol=tol
            @test cg(op, b; x0 = expected\b).iterations == 0
            dest = zeros(T, 2)
            @test cg!(dest, op, b).x === dest
            @test dest ≈ expected\b rtol=tol
            fill!(dest, zero(T))
            old = copy(dest)
            @test_throws FaerCgError cg!(dest, op, b; maxiter = 0)
            @test dest == old
            close(op)
            @test !isopen(op)
            @test_throws ArgumentError op*b
        end
        op=FaerLinearOperator(I, 2; T)
        @test op*b == b
        @test cg(op, b).x ≈ b
        @test diagonal_preconditioner(T[4, 3])*b ≈ b ./ T[4, 3]
        @test cg(A, b; M = diagonal_preconditioner(T[4, 3])).x ≈ A\b rtol=tol
        failed=cg(A, b; maxiter = 0, check = false)
        @test !failed.converged
        @test failed.reason == :no_convergence
        @test failed.iterations == 0
        @test iszero(failed.x)
        @test_throws FaerCgError cg(A, b; maxiter = 0)
    end
    A=FaerMatrix([4.0 1; 1 3])
    op=FaerLinearOperator(A)
    A[1, 1]=99
    close(A)
    @test Matrix(op)==[4.0 1; 1 3]
    coefficients=op.coefficients
    coefficients[1, 1]=12
    @test op[1, 1]==4
    @test Matrix(copy(op))==Matrix(op)
    @test_throws DimensionMismatch cg(ones(2, 3), ones(2))
    @test_throws ArgumentError cg(op, ones(2); rtol = -1)
    @test_throws ArgumentError diagonal_preconditioner([0.0, 1])
    n=50_000
    diagonal=collect(range(1.0, 2.0; length = n))
    large=FaerLinearOperator(spdiagm(0=>diagonal))
    result=cg(large, ones(n); M = diagonal_preconditioner(diagonal), maxiter = 2)
    @test result.converged
    @test result.x ≈ inv.(diagonal)
end
