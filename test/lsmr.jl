using Test, Colmat, LinearAlgebra, SparseArrays

@testset "Native block LSMR" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        R = typeof(real(zero(T)))
        tolerance = 1024eps(R)
        A = T[1 0; 0 2; 1 1; 2 -1]
        T <: Complex && (A[1, 2] = T(0.5im))
        b = T[1, 2, 4, -1]
        for source in (
            FaerMatrix(A),
            FaerSparseMatrix(sparse(A)),
            FaerSparseMatrix(sparse(A); format=:csr),
        )
            for M in (nothing, diagonal_preconditioner(T[2, 3]))
                result = lsmr(source, b; M)
                @test result isa FaerLsmrResult
                @test issuccess(result)
                @test result.x ≈ A \ b rtol = tolerance
                residual = b - A * result.x
                normal = A' * residual
                reference = A' * b
                if !isnothing(M)
                    normal = Matrix(M)' * normal
                    reference = Matrix(M)' * reference
                end
                @test result.abs_residual ≈ norm(residual) atol = tolerance
                @test result.rel_residual ≈ norm(residual) / norm(b) atol = tolerance
                @test result.normal_abs_residual ≈ norm(normal) atol = tolerance
                @test result.normal_rel_residual ≈ norm(normal) / norm(reference) atol =
                    tolerance
                @test result.normal_abs_residual <= 128eps(R) * norm(reference)
            end
            native = lsmr(source, FaerVector(b))
            @test native.x isa FaerVector{T}
            @test Vector(native.x) ≈ A \ b rtol = tolerance
            @test lsmr(source, b; x0=A \ b).iterations == 0
            @test iszero(lsmr(source, zeros(T, 4)).x)
            B = hcat(b, T[1, -1, 0, 2])
            block = lsmr(source, B; rtol=R(1e-4))
            @test Matrix(block.x) ≈ A \ B rtol = R(1e-3)
            dest = zeros(T, 2)
            @test lsmr!(dest, source, b).x === dest
            @test dest ≈ A \ b rtol = tolerance
            fill!(dest, 0)
            @test_throws FaerLsmrError lsmr!(dest, source, b; maxiter=0)
            @test iszero(dest)
            failed = lsmr!(dest, source, b; maxiter=0, check=false)
            @test !failed.converged
            @test failed.reason == :no_convergence
            @test failed.iterations == 0
            @test failed.x === dest
            # Native absolute tolerance applies even with a zero relative tolerance.
            relaxed = lsmr(source, b; atol=2norm(A' * b), rtol=0, maxiter=0)
            @test relaxed.converged
            @test relaxed.iterations == 0
        end
        wide = T[1 0 1; 0 2 1]
        rhs = T[1, 2]
        @test lsmr(wide, rhs).x ≈ pinv(wide) * rhs rtol = tolerance
        @test_throws DimensionMismatch lsmr(A, b; x0=zeros(T, 4))
        @test_throws DimensionMismatch lsmr(A, b; M=identity_preconditioner(4; T))
        @test_throws DimensionMismatch lsmr!(zeros(T, 4), A, b)
        @test_throws ArgumentError lsmr(A, b; atol=-1)
        @test_throws ArgumentError lsmr(A, b; rtol=NaN)
        @test_throws ArgumentError lsmr(A, b; maxiter=-1)
        @test_throws ArgumentError lsmr(A, fill(T(NaN), 4))
    end
    for A in ([1.0 2; 2 4; 3 6], [1.0 2 3; 2 4 6], zeros(3, 2))
        b = ones(size(A, 1))
        result = lsmr(A, b)
        @test result.converged
        @test result.x ≈ pinv(A) * b
    end
    A = [1.0 0; 0 2; 1 1; 2 -1]
    B = [1.0 2 -1; 2 1 0; 4 0 3; -1 1 2]
    @test Matrix(lsmr(A, B).x) ≈ A \ B
    mixed = lsmr(Float32.(A), ComplexF64[1 + im, 2, 4, -1])
    @test eltype(mixed.x) === ComplexF64
    @test mixed.x ≈ A \ ComplexF64[1 + im, 2, 4, -1]
    precond = diagonal_preconditioner(ComplexF64[2 + im, 3 - im])
    complex_result = lsmr(A, ComplexF64[1 + im, 2, 4, -1]; M=precond)
    @test complex_result.x ≈ A \ ComplexF64[1 + im, 2, 4, -1]
    r = ComplexF64[1 + im, 2, 4, -1] - A * complex_result.x
    @test complex_result.normal_abs_residual ≈ norm(Matrix(precond)' * A' * r) atol = 1e-12
    output = FaerVector(zeros(2))
    @test lsmr!(output, A, ones(4)).x === output
    @test Vector(output) ≈ A \ ones(4)
    closed = identity_preconditioner(2)
    close(closed)
    @test_throws ArgumentError lsmr(A, ones(4); M=closed)
    @test_throws ArgumentError lsmr(A, ones(4); x0=fill(Inf, 2))
    for (m, n, k) in ((0, 3, 1), (3, 0, 1), (0, 0, 1), (3, 2, 0))
        result = lsmr(zeros(m, n), zeros(m, k))
        @test size(result.x) == (n, k)
        @test result.converged
        @test result.iterations == 0
        @test iszero(Matrix(result.x))
    end
    n = 30_000
    A = sparse(1:n, 1:n, collect(range(1.0, 2.0; length=n)), n + 1, n)
    result = lsmr(A, ones(n + 1); M=diagonal_preconditioner(diag(A)), maxiter=3)
    @test result.converged
    @test result.x ≈ inv.(diag(A))
    @test result.abs_residual ≈ 1
    @test isempty(
        Test.detect_ambiguities(Colmat, Base, LinearAlgebra, SparseArrays; recursive=true)
    )
end

@testset "Independent native LSMR absolute and relative tolerances" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        Rhs in (T, ComplexF64), storage in (:dense, :csc, :csr), columns in (1, 2),
        preconditioned in (false, true)

        A = Matrix{T}(I, 4, 4)
        # Coordinate RHS makes the QR residual exact at the equality boundary.
        B = columns == 1 ? Rhs[1, 0, 0, 0] : Rhs[1 0; 0 1; 0 0; 0 0]
        Rhs <: Complex && (B .*= Rhs(im))
        source =
            storage === :dense ? FaerMatrix(A) : FaerSparseMatrix(sparse(A); format=storage)
        phase = T <: Complex ? T(2im) : T(2)
        right = preconditioned ? Diagonal(fill(phase, 4)) : Diagonal(ones(T, 4))
        M = preconditioned ? FaerLinearOperator(right) : nothing
        P = promote_type(T, Rhs)
        realtype = typeof(real(zero(P)))
        reference = norm(P.(right)' * P.(A)' * P.(B))
        rhs = columns == 1 ? B : FaerMatrix(B)
        for (accepted, atol, rtol) in (
            (true, reference, 0),
            (false, reference / 2, 0),
            (true, 0, 1),
            (true, 2reference, 0.25),
            (true, reference / 4, 2),
            (false, reference / 4, 0.25),
        )
            result = lsmr(
                source,
                rhs;
                M,
                atol,
                rtol,
                maxiter=0,
                check=false,
                history=true,
                callback=_ -> error("no update expected"),
            )
            @test result.converged == accepted
            @test result.reason == (accepted ? :converged : :no_convergence)
            @test result.iterations == 0
            @test iszero(result.x)
            @test isempty(result.history)
            @test result.normal_abs_residual ≈ reference rtol = 32eps(realtype)
            @test result.normal_rel_residual ≈ 1 rtol = 32eps(realtype)
            @test result.estimated_abs_residual ≈ reference rtol = 32eps(realtype)
            @test result.estimated_rel_residual ≈ 1 rtol = 32eps(realtype)
            result.x isa Union{FaerMatrix,FaerVector} && close(result.x)
        end
        columns == 2 && close(rhs)
        isnothing(M) || close(M)
        close(source)
    end
end

@testset "Native absolute tolerance preserves initial iterate and zero cases" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        unit = T <: Complex ? T(im) : one(T)
        b = T[unit, 0, 0, 0]
        source = FaerSparseMatrix(sparse(Matrix{T}(I, 4, 4)); format=:csr)
        x = FaerVector(b / 2)
        result = lsmr!(
            x,
            source,
            b;
            atol=0.5,
            rtol=0,
            maxiter=0,
            history=true,
            callback=_ -> error("no update expected"),
        )
        @test result.x === x
        @test result.converged
        @test Vector(x) == b / 2
        @test result.normal_abs_residual == result.estimated_abs_residual == 0.5
        @test result.normal_rel_residual == result.estimated_rel_residual == 0.5
        @test isempty(result.history)
        @test_throws FaerLsmrError lsmr!(x, source, b; atol=0.25, rtol=0, maxiter=0)
        @test Vector(x) == b / 2
        dense_B = T[1 0; 1 0; 1 0; 0 1]
        dense_result = lsmr(source, dense_B; atol=2norm(dense_B), rtol=0, maxiter=0)
        @test dense_result.converged
        @test dense_result.iterations == 0
        @test dense_result.normal_abs_residual ≈ norm(dense_B)
        @test dense_result.estimated_abs_residual ≈ norm(dense_B) rtol =
            32eps(typeof(real(zero(T))))
        close(dense_result.x)
        close(x)
        close(source)
        zero_result = lsmr(zeros(T, 4, 2), b; atol=0, rtol=0, maxiter=0, history=true)
        @test zero_result.converged
        @test iszero(zero_result.x)
        @test zero_result.normal_abs_residual == zero_result.estimated_abs_residual == 0
        @test zero_result.normal_rel_residual == zero_result.estimated_rel_residual == 0
        @test isempty(zero_result.history)
    end
end
