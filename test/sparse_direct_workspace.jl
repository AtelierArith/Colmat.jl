using Test, Colmat, LinearAlgebra, SparseArrays
function sds_matrix(T, n, kind)
    A=zeros(T, n, n)
    for i = 1:n
        A[i, i]=T(4+i)
        if i<n
            A[i, i+1]=T<:Complex ? T(0.2+0.1im) : T(0.2)
            A[i+1, i]=kind===:cholesky ? conj(A[i, i+1]) : T(0.7)
        end
    end
    if kind===:lu && n>1
        A[[1, n], :]=A[[n, 1], :]
    end
    return A
end
function sds_check(f, A, T, R, team)
    P=promote_type(T, R)
    tol=300eps(real(T))*max(size(A, 1), 1)
    for (g, M) in ((f, A), (transpose(f), transpose(A)), (adjoint(f), adjoint(A))),
        right in (false, true)

        b=fill(R<:Complex ? R(1+0.3im) : one(R), right ? (2, size(A, 1)) : (size(A, 1), 2))
        gold=right ? P.(b)/P.(M) : P.(M)\P.(b)
        x=zeros(P, size(gold))
        fun=right ? rdiv! : ldiv!
        args=right ? (x, b, g) : (x, g, b)
        req=scratch_requirements(fun, args...; team)
        s=FaerScratch(fun, args...; team)
        cap=scratch_capacity(s)
        for _ = 1:2
            @test fun(args...; scratch = s, team)===x
            @test x≈gold atol=tol rtol=tol
            @test scratch_capacity(s)==cap
        end
        plain=right ? /(b, g; team, scratch = s) : \(g, b; team, scratch = s)
        @test plain isa FaerMatrix{P}
        @test Matrix(plain)≈gold atol=tol rtol=tol
        close(plain)
        aliased=P.(b)
        @test (
            right ? rdiv!(aliased, g; scratch = s, team) :
            ldiv!(g, aliased; scratch = s, team)
        )===aliased
        @test aliased≈gold atol=tol rtol=tol
        native=FaerMatrix(zeros(P, size(gold)))
        rhs=FaerMatrix(b)
        nativeargs=right ? (native, rhs, g) : (native, g, rhs)
        @test fun(nativeargs...; scratch = s, team)===native
        @test Matrix(native)≈gold atol=tol rtol=tol
        backing=zeros(P, size(x, 1)*2, size(x, 2)*2)
        dest=@view backing[1:2:end, 1:2:end]
        viewargs=right ? (dest, b, g) : (dest, g, b)
        @test fun(viewargs...; scratch = s, team)===dest
        @test dest≈gold atol=tol rtol=tol
        if req.bytes>0
            short=FaerScratch(req.bytes-1; alignment = req.alignment)
            before=copy(x)
            @test_throws ArgumentError fun(args...; scratch = short, team)
            @test x==before
            @test reserve!(short, fun, args...; team)===short
            @test fun(args...; scratch = short, team)≈gold atol=tol rtol=tol
            close(short)
        end
        close(s)
        before=copy(x)
        @test_throws ArgumentError fun(args...; scratch = s, team)
        @test x==before
        if !right
            v=b[:, 1]
            expected=gold[:, 1]
            work=FaerScratch(ldiv!, g, v; team)
            @test \(g, v; scratch = work, team)≈expected atol=tol rtol=tol
            d=zeros(P, length(expected))
            @test ldiv!(d, g, v; scratch = work, team)===d
            @test d≈expected atol=tol rtol=tol
            nv=FaerVector(v)
            out=\(g, nv; scratch = work, team)
            @test out isa FaerVector{P}
            @test Vector(out)≈expected atol=tol rtol=tol
            close(out)
            close(nv)
            close(work)
        end
        close(rhs)
        close(native)
    end
end
@testset "sparse LU/Cholesky direct solve workspace" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        kind in (:lu, :cholesky),
        algorithm in (:auto, :simplicial, :supernodal),
        format in (:csc, :csr),
        n in (0, 1, 5)

        A=sds_matrix(T, n, kind)
        a=FaerSparseMatrix(sparse(A); format)
        order=reverse(collect(1:n))
        f=kind===:lu ? lu(a; algorithm, ordering = order) :
          cholesky(Hermitian(a); algorithm, ordering = order)
        close(a)
        close(f.source)
        close(f.plan)
        for threaded in (false, true)
            team=threaded ? FaerTeam(2) : nothing
            sds_check(f, A, T, T, team)
            isnothing(team)||close(team)
        end
        close(f)
    end
end
@testset "mixed saved sparse solves with scratch and teams" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        R in (Float32, Float64, ComplexF32, ComplexF64),
        kind in (:lu, :cholesky),
        algorithm in (:auto, :simplicial, :supernodal)

        T===R && continue
        A=sds_matrix(T, 4, kind)
        a=FaerSparseMatrix(sparse(A))
        f=kind===:lu ? lu(a; algorithm, ordering = [4, 2, 1, 3]) :
          cholesky(Hermitian(a); algorithm, ordering = [4, 2, 1, 3])
        close(a)
        close(f.source)
        close(f.plan)
        team=FaerTeam(2)
        sds_check(f, A, T, R, team)
        close(team)
        close(f)
    end
end

@testset "sparse direct workspace validation and staged conversion" begin
    for kind in (:lu, :cholesky), T in (Float32, Float64, ComplexF32, ComplexF64)
        a=FaerSparseMatrix(sparse(sds_matrix(T, 3, kind)))
        f=kind===:lu ? lu(a) : cholesky(Hermitian(a))
        b=ones(T, 3, 2)
        x=fill(T(9), 3, 2)
        before=copy(x)
        team=FaerTeam(2)
        s=FaerScratch(ldiv!, f, b; team)
        @test_throws ArgumentError ldiv!(x, f, b; scratch = :bad, team)
        @test_throws ArgumentError ldiv!(x, f, b; scratch = s, team = :bad)
        @test x==before
        @test_throws DimensionMismatch scratch_requirements(ldiv!, f, ones(T, 2, 2))
        @test_throws DimensionMismatch scratch_requirements(rdiv!, ones(T, 2, 2), f)
        @test_throws DimensionMismatch ldiv!(zeros(T, 4, 2), f, b; scratch = s, team)
        @test_throws ArgumentError ldiv!(x, f, b; atol = 0)
        @test x==before
        f.dims=(4, 4)
        @test_throws Colmat.RustCall.RustError scratch_requirements(ldiv!, f, ones(T, 4, 2))
        @test_throws Colmat.RustCall.RustError ldiv!(
            zeros(T, 4, 2),
            f,
            ones(T, 4, 2);
            scratch = s,
            team,
        )
        f.dims=(3, 3)
        rhs=FaerMatrix(b)
        rhs.dims=(4, 2)
        @test_throws Colmat.RustCall.RustError ldiv!(x, f, rhs; scratch = s, team)
        @test x==before
        rhs.dims=(3, 2)
        close(rhs)
        @test_throws ArgumentError ldiv!(x, f, rhs; scratch = s, team)
        @test fetch(@async begin
            d=zeros(T, 3, 2)
            ldiv!(d, f, b; scratch = s, team)
            d
        end)≈sds_matrix(T, 3, kind)\b atol=1e-3
        close(team)
        @test_throws ArgumentError ldiv!(x, f, b; scratch = s, team)
        @test x==before
        close(f)
        @test_throws ArgumentError scratch_requirements(ldiv!, f, b)
        close(s)
        close(a)
    end
    f=lu(FaerSparseMatrix(sparse([2.0 0; 0 3])))
    b=ones(ComplexF64, 2, 2) .* (1+im)
    dest=fill(9.0, 2, 2)
    s=FaerScratch(ldiv!, f, b)
    @test_throws InexactError ldiv!(dest, f, b; scratch = s)
    @test dest==fill(9.0, 2, 2)
    native=FaerMatrix(dest)
    @test_throws InexactError ldiv!(native, f, b; scratch = s)
    @test Matrix(native)==dest
    close(native)
    close(s)
    close(f)
    singular=lu(FaerSparseMatrix(sparse([1.0 0; 0 0])); check = false)
    @test_throws SingularException scratch_requirements(ldiv!, singular, ones(2, 1))
    close(singular)
    indefinite=cholesky(Hermitian(FaerSparseMatrix(sparse([1.0 0; 0 -1]))); check = false)
    @test_throws PosDefException scratch_requirements(ldiv!, indefinite, ones(2, 1))
    close(indefinite)
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
