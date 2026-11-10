using Test, LinearAlgebra, SparseArrays, Colmat
function qacw_fixture(T, m, n, deficient)
    a = T[0.04sin(i + 2j) for i in 1:m, j in 1:n]
    for i in 1:min(m, n)
        a[i, i] += T(3 + i)
    end
    T <: Complex && (a .+= T(0.03im) .* T[cos(i + 3j) for i in 1:m, j in 1:n])
    if deficient && min(m, n) > 0
        m > n ? (a[:, end] .= a[:, 1]) : (a[end, :] .= a[1, :])
    end
    return a
end
qacw_tol(T, n) = 500eps(real(T)) * max(n, 1)
@testset "dense QR pseudoinverse construction with caller scratch/team" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        pivoted in (false, true),
        (m, n, deficient) in ((7, 3, false), (3, 7, true), (4, 4, true), (0, 3, false))

        a = qacw_fixture(T, m, n, deficient)
        f = pivoted ? qr(FaerMatrix(a), ColumnNorm()) : qr(FaerMatrix(a))
        rt = 1e-4
        reference = pinv(a; rtol=rt)
        tol = qacw_tol(T, max(m, n))
        FaerTeam(2) do team
            for tm in (nothing, team)
                req = scratch_requirements(FaerLinearOperator, f; rtol=rt, team=tm)
                @test req.bytes >= 0 && ispow2(req.alignment)
                min(m, n) > 0 && @test req.bytes > 0
                work = FaerScratch(FaerLinearOperator, f; rtol=rt, team=tm)
                cap = scratch_capacity(work)
                for _ in 1:2
                    op = FaerLinearOperator(f; rtol=rt, scratch=work, team=tm)
                    @test size(op) == (n, m)
                    @test Matrix(op) ≈ reference atol = tol * max(norm(reference), 1) rtol =
                        tol
                    close(op)
                    @test scratch_capacity(work) == cap
                end
                if req.bytes > 0
                    short = FaerScratch(req.bytes - 1; alignment=req.alignment)
                    @test_throws ArgumentError FaerLinearOperator(
                        f; rtol=rt, scratch=short, team=tm
                    )
                    @test reserve!(short, FaerLinearOperator, f; rtol=rt, team=tm) === short
                    op = FaerLinearOperator(f; rtol=rt, scratch=short, team=tm)
                    @test Matrix(op) ≈ reference atol = tol * max(norm(reference), 1) rtol =
                        tol
                    close(op)
                    close(short)
                end
                close(work)
            end
        end
        # Without scratch a temporary workspace gives the same action.
        op = FaerLinearOperator(f; rtol=rt)
        @test Matrix(op) ≈ reference atol = tol * max(norm(reference), 1) rtol = tol
        close(op)
        close(f)
    end
    a = qacw_fixture(Float64, 4, 4, false)
    f = qr(FaerMatrix(a))
    # Square QR without tolerances keeps the nonsingular inverse: no scratch.
    @test scratch_requirements(FaerLinearOperator, f).bytes == 0
    work = FaerScratch(0)
    op = FaerLinearOperator(f; scratch=work)
    @test Matrix(op) ≈ inv(a)
    close(op)
    @test_throws ArgumentError FaerLinearOperator(f; rtol=0.1, scratch=work)
    @test_throws ArgumentError FaerLinearOperator(f; rtol=0.1, scratch=1)
    @test_throws ArgumentError FaerLinearOperator(f; rtol=0.1, team=1)
    @test_throws ArgumentError scratch_requirements(FaerLinearOperator, f; atol=-1)
    close(work)
    @test_throws ArgumentError FaerLinearOperator(f; rtol=0.1, scratch=work)
    close(f)
    @test_throws ArgumentError scratch_requirements(FaerLinearOperator, f; rtol=0.1)
end
function qacw_sparse(T, m, n)
    a = zeros(T, m, n)
    for j in 1:n, i in 1:m
        a[i, j] = T(
            if mod(j - 1, m) + 1 == i
                4 + j
            elseif (i + j) % 5 == 0
                0.1
            else
                0
            end,
        )
        T <: Complex && a[i, j] != 0 && (a[i, j] += T(0.02im * mod(i + 2j, 5)))
    end
    return a
end
@testset "wide sparse QR inverse construction with caller scratch/team" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        (m, n) in ((3, 6), (12, 30)),
        format in (:csc, :csr)

        a = qacw_sparse(T, m, n)
        source = FaerSparseMatrix(sparse(a); format)
        f = qr(source)
        close(source)
        close(f.source)
        reference = pinv(a)
        tol = qacw_tol(T, n)
        FaerTeam(2) do team
            for tm in (nothing, team)
                req = scratch_requirements(FaerLinearOperator, f; team=tm)
                @test req.bytes > 0
                work = FaerScratch(FaerLinearOperator, f; team=tm)
                op = FaerLinearOperator(f; scratch=work, team=tm)
                @test size(op) == (n, m)
                @test Matrix(op) ≈ reference atol = tol * max(norm(reference), 1) rtol = tol
                @test Matrix(op') ≈ reference' atol = tol * max(norm(reference), 1) rtol =
                    tol
                close(op)
                short = FaerScratch(req.bytes - 1; alignment=req.alignment)
                @test_throws ArgumentError FaerLinearOperator(f; scratch=short, team=tm)
                reserve!(short, FaerLinearOperator, f; team=tm)
                op = FaerLinearOperator(f; scratch=short, team=tm)
                @test Matrix(op) ≈ reference atol = tol * max(norm(reference), 1) rtol = tol
                close(op)
                close(short)
                close(work)
            end
        end
        close(f)
    end
end
@testset "sparse inverse construction without kernel scratch" begin
    a = qacw_sparse(Float64, 6, 3)
    for kind in (:qr, :lu, :cholesky)
        f = if kind === :qr
            qr(FaerSparseMatrix(sparse(a)))
        elseif kind === :lu
            lu(FaerSparseMatrix(sparse(a'a + I)))
        else
            cholesky(FaerSparseMatrix(sparse(a'a + I)))
        end
        @test scratch_requirements(FaerLinearOperator, f).bytes == 0
        work = FaerScratch(0)
        FaerTeam(2) do team
            op = FaerLinearOperator(f; scratch=work, team)
            ref = kind === :qr ? pinv(a) : inv(a'a + I)
            @test Matrix(op) ≈ ref atol = 1e-10
            close(op)
        end
        close(work)
        close(f)
    end
    # Rank-deficient tall QR is routed to the independent dense pinv(F): no
    # construction scratch is consumed.
    d = [1.0 2; 2 4; 0 0; 0 0]
    f = qr(FaerSparseMatrix(sparse(d)))
    @test scratch_requirements(FaerLinearOperator, f).bytes == 0
    op = FaerLinearOperator(f; scratch=FaerScratch(0))
    @test Matrix(op) ≈ pinv(d)
    close(op)
    close(f)
    # Rank-deficient wide R keeps the wide action; pinv(R) truncates at
    # min(m,n)*eps and its SVD uses caller scratch.
    f = qr(FaerSparseMatrix(sparse(Matrix(d'))))
    req = scratch_requirements(FaerLinearOperator, f)
    @test req.bytes > 0
    @test_throws ArgumentError FaerLinearOperator(f; scratch=FaerScratch(0))
    op = FaerLinearOperator(f; scratch=FaerScratch(FaerLinearOperator, f))
    @test Matrix(op) ≈ pinv(Matrix(d'))
    close(op)
    close(f)
end
