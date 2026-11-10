using Colmat, LinearAlgebra, SparseArrays, Test
function rrs_fixture(T, m, n)
    A = T[0.04sin(2i + j) for i in 1:m, j in 1:n]
    for i in 1:min(m, n)
        A[i, i] += T(4 + i)
    end
    T <: Complex && (A .+= T(0.02im) .* T[cos(i + j) for i in 1:m, j in 1:n])
    return A
end
function rrs_check(f, A, T, team; sparse_factor=false, qr_factor=false)
    for (g, M) in ((f, A), (transpose(f), transpose(A)), (adjoint(f), adjoint(A))),
        R in (T, ComplexF64),
        Row in (transpose, adjoint)

        v = R.(collect(1:size(g, 2)))
        R <: Complex && (v .+= R(0.3im))
        before = copy(v)
        b = Row(v)
        P = promote_type(T, R)
        gold = Matrix(b) * pinv(P.(M); rtol=1e-5)
        kw = qr_factor ? (rtol=1e-5,) : NamedTuple()
        x = zeros(P, 1, size(g, 1))
        work = FaerScratch(rdiv!, x, b, g; team, kw...)
        @test rdiv!(x, b, g; scratch=work, team, kw...) === x
        @test x ≈ gold atol = 1e-3 rtol = 1e-3
        @test v == before
        if sparse_factor
            out = /(b, g; scratch=work, team, kw...)
            @test Matrix(out) ≈ gold atol = 1e-3 rtol = 1e-3
            close(out)
        end
        native = FaerMatrix(zeros(P, size(gold)))
        @test rdiv!(native, b, g; scratch=work, team, kw...) === native
        @test Matrix(native) ≈ gold atol = 1e-3 rtol = 1e-3
        if size(g, 1) == size(g, 2)
            # A complex transpose/adjoint row is itself a writable destination.
            destv = zeros(P, size(g, 1))
            dest = Row(destv)
            copyto!(dest, b)
            @test rdiv!(dest, g; scratch=work, team, kw...) === dest
            @test Matrix(dest) ≈ gold atol = 1e-3 rtol = 1e-3
        end
        close(native)
        close(work)
    end
end
@testset "row-vector RHS retained factor right solves" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        n in (0, 1, 4),
        threaded in (false, true)

        team = threaded ? FaerTeam(2) : nothing
        A = rrs_fixture(T, n, n)
        H = A * A' + I
        for make in (lu, qr)
            a = FaerMatrix(A)
            f = make(a)
            close(a)
            rrs_check(f, A, T, team)
            close(f)
        end
        for algorithm in (:auto, :simplicial, :supernodal), kind in (:lu, :qr, :cholesky)
            M = kind === :cholesky ? Matrix{T}(H) : A
            a = FaerSparseMatrix(sparse(M))
            f = if kind === :lu
                lu(a; algorithm)
            elseif kind === :qr
                qr(a; algorithm)
            else
                cholesky(Hermitian(a); algorithm)
            end
            close(a)
            close(f.source)
            close(f.plan)
            rrs_check(f, M, T, team; sparse_factor=true, qr_factor=kind === :qr)
            close(f)
        end
        isnothing(team) || close(team)
    end
end
@testset "row-vector RHS rectangular QR right solves" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), (m, n) in ((6, 3), (3, 6))
        A = rrs_fixture(T, m, n)
        team = FaerTeam(2)
        d = qr(FaerMatrix(A))
        rrs_check(d, A, T, team)
        close(d)
        for algorithm in (:auto, :simplicial, :supernodal)
            a = FaerSparseMatrix(sparse(A))
            f = qr(a; algorithm)
            close(a)
            close(f.source)
            close(f.plan)
            rrs_check(f, A, T, team; sparse_factor=true, qr_factor=true)
            close(f)
        end
        close(team)
    end
end
@testset "right row solve conversion preserves destinations" begin
    for sparse_factor in (false, true)
        f = if sparse_factor
            lu(FaerSparseMatrix(sparse([2.0 0; 0 3])))
        else
            lu(FaerMatrix([2.0 0; 0 3]))
        end
        b = adjoint(ComplexF64[1 + im, 2 + im])
        dest = fill(9.0, 1, 2)
        work = FaerScratch(rdiv!, dest, b, f)
        @test_throws InexactError rdiv!(dest, b, f; scratch=work)
        @test dest == fill(9.0, 1, 2)
        close(work)
        close(f)
    end
end
