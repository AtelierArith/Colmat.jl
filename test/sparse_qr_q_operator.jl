using Colmat, LinearAlgebra, SparseArrays, Test
sqq_tol(T, n) = 500eps(real(T)) * max(n, 1)
function sqq_fixture(T, m, n; empty=false)
    a = zeros(T, m, n)
    for j in 1:n, i in 1:m
        empty && iseven(j) && continue
        a[i, j] = T(
            if i == mod(j - 1, max(m, 1)) + 1
                4 + j
            elseif mod(i + j, 4) == 1
                0.3
            else
                0
            end,
        )
        T <: Complex && a[i, j] != 0 && (a[i, j] += T(0.2im))
    end
    return a
end

@testset "Lazy sparse QR Q operator" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        format in (:csc, :csr), algorithm in (:auto, :simplicial, :supernodal),
        (m, n, empty) in (
            (0, 0, false),
            (5, 0, false),
            (1, 1, false),
            (5, 3, false),
            (6, 6, false),
            (6, 4, true),
            (3, 6, false),
            (40, 20, false),
        )

        algorithm === :simplicial && m < n && continue
        a = sqq_fixture(T, m, n; empty)
        source = FaerSparseMatrix(sparse(a); format)
        F = qr(source; algorithm)
        close(source)
        Q = Matrix(F.Qfull)
        tol = sqq_tol(T, max(m, n))
        op = F.Qoperator
        @test op isa FaerLinearOperator{T}
        @test size(op) == (m, m)
        @test :Qoperator in propertynames(F)
        @test Vector(F.prow) == 1:m
        # Independent of the factorization and its source snapshot.
        close(F.source)
        close(F)
        @test Matrix(op) ≈ Q atol = tol rtol = tol
        k = min(m, n)
        R = Matrix{T}(I, m, k)
        @test Matrix(op * R) ≈ Q[:, 1:k] atol = tol rtol = tol
        b = T[i + 0.5j for i in 1:m, j in 1:2]
        for (variant, reference) in
            ((op, Q), (op', Q'), (transpose(op), transpose(Q)), (conj(op), conj(Q)))
            @test Matrix(variant * b) ≈ reference * b atol = tol rtol = tol
            v = b[:, 1]
            @test Vector(variant * v) ≈ reference * v atol = tol rtol = tol
            x = fill(T(42), m, 2)
            @test mul!(x, variant, b) === x
            @test x ≈ reference * b atol = tol rtol = tol
            variant === op || close(variant)
        end
        # Q'Q = I and Q' undoes Q.
        @test Matrix(op' * (op * b)) ≈ b atol = tol rtol = tol
        # Native operands, native view destinations and aliasing (staged).
        fb = FaerMatrix(b)
        y = op * fb
        @test y isa FaerMatrix{T}
        @test Matrix(y) ≈ Q * b atol = tol rtol = tol
        close(y)
        dest = view(FaerMatrix(zeros(T, m + 1, 3)), 2:(m + 1), 1:2)
        @test mul!(dest, op', fb) === dest
        @test Matrix(dest) ≈ Q' * b atol = tol rtol = tol
        @test mul!(fb, op, fb) === fb
        @test Matrix(fb) ≈ Q * b atol = tol rtol = tol
        alias = copy(b)
        @test mul!(alias, op', alias) === alias
        @test alias ≈ Q' * b atol = tol rtol = tol
        close(fb)
        close(op)
        @test_throws ArgumentError op * b
    end
end

@testset "Lazy sparse QR Q reconstructs R, permutations and views" begin
    for T in (Float64, ComplexF64), algorithm in (:auto, :simplicial, :supernodal)
        a = sqq_fixture(T, 7, 4)
        F = qr(FaerSparseMatrix(sparse(a)); algorithm, ordering=[3, 1, 4, 2])
        op = F.Qoperator
        R = vcat(Matrix(F.R), zeros(T, 3, 4))
        @test Matrix(op * R) ≈ a[:, Vector(F.pcol)]
        @test Matrix(op' * a[:, Vector(F.pcol)]) ≈ R atol = 1e-10
        @test Vector(F.prow) == 1:7
        @test Matrix(op)[Vector(F.prow), :] * R ≈ a[:, Vector(F.pcol)]
        # Native destination/source views and mixed RHS promotion.
        parent = zeros(T, 9, 4)
        dest = view(parent, 2:8, 2:3)
        src = view(a, :, 1:2)
        @test mul!(dest, op', src) === dest
        @test dest ≈ Matrix(F.Qfull)' * a[:, 1:2]
        if T === Float64
            z = ComplexF64.(a[:, 1]) .+ 0.5im
            @test Vector(op * z) ≈ Matrix(F.Qfull) * z
        end
        close(op)
        close(F)
    end
end

@testset "Lazy sparse QR Q with caller scratch and team" begin
    for T in (Float32, ComplexF64), algorithm in (:auto, :simplicial, :supernodal)
        a = sqq_fixture(T, 30, 12)
        F = qr(FaerSparseMatrix(sparse(a)); algorithm)
        Q = Matrix(F.Qfull)
        op = F.Qoperator
        close(F)
        b = T[sin(i + j) for i in 1:30, j in 1:3]
        tol = sqq_tol(T, 30)
        FaerTeam(2) do team
            for variant in (op, op')
                x = zeros(T, 30, 3)
                reference = variant === op ? Q * b : Q' * b
                req = scratch_requirements(mul!, x, variant, b; team)
                @test req.bytes > 0
                work = FaerScratch(mul!, x, variant, b; team)
                cap = scratch_capacity(work)
                for _ in 1:2
                    @test mul!(x, variant, b; scratch=work, team) === x
                    @test x ≈ reference atol = tol rtol = tol
                end
                @test scratch_capacity(work) == cap
                short = FaerScratch(req.bytes - 1; alignment=req.alignment)
                saved = copy(x)
                @test_throws ArgumentError mul!(x, variant, b; scratch=short, team)
                @test x == saved
                close(short)
                close(work)
                variant === op || close(variant)
            end
        end
        close(op)
    end
end

@testset "Lazy sparse QR Q validation" begin
    F = qr(FaerSparseMatrix(sparse([1.0 0; 0 2; 1 1])))
    op = F.Qoperator
    @test_throws DimensionMismatch op * ones(2)
    close(op)
    close(F)
    @test_throws ArgumentError F.Qoperator
    @test_throws ArgumentError F.prow
end
