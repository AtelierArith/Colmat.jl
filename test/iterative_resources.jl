using Colmat, LinearAlgebra, SparseArrays, Test
ir_array(x) = x isa AbstractVector ? Vector(x) : Matrix(x)
function ir_coefficients(T, m, n, kind)
    A = T[0.01sin(i + 2j) for i in 1:m, j in 1:n]
    T <: Complex && (A .+= T(0.01im) .* T[cos(2i + j) for i in 1:m, j in 1:n])
    kind == 0 && (A = A + A')
    for i in 1:min(m, n)
        A[i, i] += T(5 + i / max(n, 1))
    end
    return A
end
@testset "Iterative native kernels and diagnostics with scratch and teams" begin
    for width in (0, 1, 2)
        team = width == 0 ? nothing : FaerTeam(width)
        for T in (Float32, Float64, ComplexF32, ComplexF64),
            (kind, fun, bang) in ((0, cg, cg!), (1, bicgstab, bicgstab!), (2, lsmr, lsmr!)),
            (m, n) in ((0, 0), (1, 1), (8, 8), (96, 96), (8, 4), (4, 8), (0, 4), (4, 0)),
            storage in (:dense, :csc, :csr),
            preconditioned in (false, true),
            vector in (false, true)

            kind != 2 && m != n && continue
            A = ir_coefficients(T, m, n, kind)
            source = if storage == :dense
                FaerMatrix(A)
            else
                FaerSparseMatrix(sparse(A); format=storage)
            end
            op = FaerLinearOperator(source)
            close(source)
            M = preconditioned ? diagonal_preconditioner(fill(T(1), n)) : nothing
            N = if kind == 1 && preconditioned
                diagonal_preconditioner(fill(T(2), n))
            else
                nothing
            end
            pre = kind == 1 ? (; M, N) : (; M)
            B = T[sin(i + j) + 0.2cos(i + 2j) for i in 1:m, j in 1:2]
            T <: Complex && (B .+= T(0.1im) .* T[cos(i + j) for i in 1:m, j in 1:2])
            b = vector ? vec(B[:, 1]) : B
            gold = pinv(A) * b
            R = typeof(real(zero(T)))
            rtol = R === Float32 ? 1e-3 : 1e-8
            tol = R === Float32 ? 5e-3 : 2e-6
            req = scratch_requirements(fun, op, b; pre..., team)
            work = FaerScratch(fun, op, b; pre..., team)
            capacity = scratch_capacity(work)
            result = fun(op, b; pre..., scratch=work, team, rtol, maxiter=256)
            @test result.converged
            @test ir_array(result.x) ≈ gold atol = tol rtol = tol
            @test result.abs_residual ≈ norm(A * ir_array(result.x) - b) atol = tol rtol =
                tol
            @test scratch_capacity(work) == capacity
            dims = vector ? (n,) : (n, 2)
            x = zeros(T, dims)
            @test scratch_requirements(bang, x, op, b; pre..., team) == req
            updated = bang(x, op, b; pre..., scratch=work, team, rtol, maxiter=256)
            @test updated.x === x
            @test x ≈ gold atol = tol rtol = tol
            if req.bytes > 0
                short = FaerScratch(req.bytes - 1; alignment=req.alignment)
                before = copy(x)
                @test_throws ArgumentError bang(x, op, b; pre..., scratch=short, team, rtol)
                @test x == before
                @test reserve!(short, bang, x, op, b; pre..., team) === short
                close(short)
            end
            close(work)
            close(op)
            isnothing(M) || close(M)
            isnothing(N) || close(N)
            result.x isa FaerMatrix && close(result.x)
        end
        isnothing(team) || close(team)
    end
end
@testset "Iterative promotion, layouts, capacity and metadata" begin
    for fun in (cg, bicgstab, lsmr)
        A = Float32[5 0.1; 0.1 6]
        op = FaerLinearOperator(A)
        b = ComplexF64[1 + im 2; 3 4 - im]
        team = FaerTeam(2)
        work = FaerScratch(fun, op, b; team)
        result = fun(op, b; scratch=work, team, rtol=1e-10)
        @test eltype(result.x) === ComplexF64
        @test Matrix(result.x) ≈ pinv(ComplexF64.(A)) * b
        @test scratch_requirements(fun, op, view(b, :, 1:1); team).bytes >= 0
        small = FaerScratch(0; alignment=1)
        @test_throws ArgumentError fun(op, b; scratch=small, team)
        @test reserve!(small, fun, op, b; team) === small
        close(small)
        close(work)
        close(op)
        close(team)
        close(result.x)
    end
    op = FaerLinearOperator([5.0 0.1; 0.1 6])
    original = op.dims
    op.dims = (3, 3)
    @test_throws Colmat.RustCall.RustError scratch_requirements(cg, op, ones(3))
    op.dims = original
    fake = FaerLinearOperator{Float32,typeof(op.handle)}(
        op.handle, op.dims, op.storage, op.transposed, op.conjugated, op.callbacks
    )
    @test_throws Colmat.RustCall.RustError scratch_requirements(cg, fake, ones(Float32, 2))
    fake.handle = nothing
    rhs = FaerMatrix(ones(2, 1))
    rhs.dims = (3, 1)
    @test_throws Colmat.RustCall.RustError scratch_requirements(cg, op, rhs)
    @test_throws DimensionMismatch cg(op, rhs)
    rhs.dims = (2, 1)
    close(rhs)
    team = FaerTeam(2)
    work = FaerScratch(cg, op, ones(2); team)
    @test_throws ArgumentError cg(op, ones(2); scratch=:invalid)
    @test_throws ArgumentError cg(op, ones(2); team=:invalid)
    @test_throws DimensionMismatch scratch_requirements(cg, op, ones(3))
    @test_throws ArgumentError scratch_requirements(
        cg, op, ones(2); N=identity_preconditioner(2)
    )
    close(team)
    @test_throws ArgumentError cg(op, ones(2); scratch=work, team)
    team = FaerTeam(2)
    close(work)
    @test_throws ArgumentError cg(op, ones(2); scratch=work, team)
    close(team)
    close(op)
end
@testset "Iterative initial guesses, empty columns and native destinations" begin
    for (fun, bang) in ((cg, cg!), (bicgstab, bicgstab!), (lsmr, lsmr!))
        op = FaerLinearOperator([5.0 0.1; 0.1 6])
        team = FaerTeam(2)
        b = FaerMatrix(ones(2, 2))
        x = FaerMatrix(fill(0.1, 2, 2))
        work = FaerScratch(bang, x, op, b; team)
        r = bang(x, op, b; scratch=work, team, rtol=1e-8)
        @test r.x === x
        @test Matrix(x) ≈ pinv([5.0 0.1; 0.1 6]) * ones(2, 2)
        empty = zeros(2, 0)
        reserve!(work, fun, op, empty; team)
        e = fun(op, empty; scratch=work, team)
        @test e.converged
        @test size(e.x) == (2, 0)
        @test_throws DimensionMismatch scratch_requirements(fun, op, ones(2); x0=ones(3))
        @test_throws ArgumentError scratch_requirements(
            bang, zeros(Float32, 2), op, ones(2)
        )
        close(work)
        close(team)
        close(op)
        close(b)
        close(x)
        close(e.x)
    end
end
