using Colmat, LinearAlgebra, SparseArrays, Test
function qrs_matrix(T, m, n, deficient)
    A=T[0.02sin(2i+j) for i = 1:m, j = 1:n]
    for i = 1:min(m, n)
        A[i, i]+=T(3+i)
    end
    T<:Complex&&(A .+= T(0.03im) .* T[cos(i+2j) for i = 1:m, j = 1:n])
    if deficient&&min(m, n)>0
        m>=n ? (A[:, end]=A[:, 1]) : (A[end, :]=A[1, :])
    end
    return A
end
function qrs_check(f, A, T, R, team)
    rt=1e-5
    P=promote_type(T, R)
    tol=300eps(real(T))*max(size(A, 1), 1)
    for (g, M) in ((f, A), (transpose(f), transpose(A)), (adjoint(f), adjoint(A))),
        right in (false, true)

        b=fill(R<:Complex ? R(1+0.3im) : one(R), right ? (2, size(M, 2)) : (size(M, 1), 2))
        gold=right ? P.(b)*pinv(P.(M); rtol = rt) : pinv(P.(M); rtol = rt)*P.(b)
        x=zeros(P, size(gold))
        fun=right ? rdiv! : ldiv!
        args=right ? (x, b, g) : (x, g, b)
        req=scratch_requirements(fun, args...; team, rtol = rt)
        s=FaerScratch(fun, args...; team, rtol = rt)
        cap=scratch_capacity(s)
        for _ = 1:2
            @test fun(args...; scratch = s, team, rtol = rt)===x
            @test x≈gold atol=tol rtol=tol
            @test scratch_capacity(s)==cap
        end
        plain=right ? /(b, g; team, scratch = s, rtol = rt) :
              \(g, b; team, scratch = s, rtol = rt)
        @test plain isa FaerMatrix{P}
        @test Matrix(plain)≈gold atol=tol rtol=tol
        close(plain)
        if size(M, 1)==size(M, 2)
            aliased=P.(b)
            @test (
                right ? rdiv!(aliased, g; scratch = s, team, rtol = rt) :
                ldiv!(g, aliased; scratch = s, team, rtol = rt)
            )===aliased
            @test aliased≈gold atol=tol rtol=tol
        end
        native=FaerMatrix(zeros(P, size(gold)))
        rhs=FaerMatrix(b)
        nativeargs=right ? (native, rhs, g) : (native, g, rhs)
        @test fun(nativeargs...; scratch = s, team, rtol = rt)===native
        @test Matrix(native)≈gold atol=tol rtol=tol
        backing=zeros(P, size(x, 1)*2, size(x, 2)*2)
        dest=@view backing[1:2:end, 1:2:end]
        viewargs=right ? (dest, b, g) : (dest, g, b)
        @test fun(viewargs...; scratch = s, team, rtol = rt)===dest
        @test dest≈gold atol=tol rtol=tol
        if req.bytes>0
            short=FaerScratch(req.bytes-1; alignment = req.alignment)
            before=copy(x)
            @test_throws ArgumentError fun(args...; scratch = short, team, rtol = rt)
            @test x==before
            @test reserve!(short, fun, args...; team, rtol = rt)===short
            @test fun(args...; scratch = short, team, rtol = rt)≈gold atol=tol rtol=tol
            close(short)
        end
        close(s)
        before=copy(x)
        @test_throws ArgumentError fun(args...; scratch = s, team, rtol = rt)
        @test x==before
        if !right
            v=b[:, 1]
            expected=gold[:, 1]
            work=FaerScratch(ldiv!, g, v; team, rtol = rt)
            @test \(g, v; scratch = work, team, rtol = rt)≈expected atol=tol rtol=tol
            d=zeros(P, length(expected))
            @test ldiv!(d, g, v; scratch = work, team, rtol = rt)===d
            @test d≈expected atol=tol rtol=tol
            nv=FaerVector(v)
            out=\(g, nv; scratch = work, team, rtol = rt)
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

@testset "all sparse QR solve resources" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        algorithm in (:auto, :simplicial, :supernodal),
        format in (:csc, :csr),
        (m, n, deficient) in (
            (0, 0, false),
            (0, 3, false),
            (3, 0, false),
            (1, 1, false),
            (6, 3, false),
            (3, 6, false),
            (4, 4, false),
            (6, 3, true),
            (3, 6, true),
            (4, 4, true),
        )

        A=qrs_matrix(T, m, n, deficient)
        a=FaerSparseMatrix(sparse(A); format)
        f=qr(a; algorithm, ordering = reverse(collect(1:n)))
        close(a)
        close(f.source)
        close(f.plan)
        for threaded in (false, true)
            team=threaded ? FaerTeam(2) : nothing
            qrs_check(f, A, T, T, team)
            isnothing(team)||close(team)
        end
        close(f)
    end
end
@testset "promoted sparse QR solve resources" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        R in (Float32, Float64, ComplexF32, ComplexF64),
        algorithm in (:auto, :simplicial, :supernodal),
        (m, n, deficient) in ((6, 3, false), (3, 6, false), (6, 3, true), (3, 6, true))

        T===R&&continue
        A=qrs_matrix(T, m, n, deficient)
        a=FaerSparseMatrix(sparse(A))
        f=qr(a; algorithm, ordering = reverse(collect(1:n)))
        close(a)
        close(f.source)
        close(f.plan)
        team=FaerTeam(2)
        qrs_check(f, A, T, R, team)
        close(team)
        close(f)
    end
end
@testset "sparse QR resource validation" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        A=qrs_matrix(T, 5, 3, false)
        a=FaerSparseMatrix(sparse(A))
        f=qr(a)
        b=ones(T, 5, 2)
        dest=fill(T(7), 3, 2)
        before=copy(dest)
        team=FaerTeam(2)
        s=FaerScratch(ldiv!, dest, f, b; team)
        @test_throws ArgumentError ldiv!(dest, f, b; scratch = :bad, team)
        @test_throws ArgumentError ldiv!(dest, f, b; scratch = s, team = :bad)
        @test_throws ArgumentError ldiv!(dest, f, b; scratch = s, team, rtol = -1)
        @test dest==before
        @test_throws DimensionMismatch ldiv!(f, b; scratch = s, team)
        @test_throws DimensionMismatch rdiv!(ones(T, 2, 3), f; scratch = s, team)
        f.dims=(4, 3)
        @test_throws Colmat.RustCall.RustError scratch_requirements(ldiv!, f, ones(T, 4, 2))
        @test_throws Colmat.RustCall.RustError ldiv!(
            dest,
            f,
            ones(T, 4, 2);
            scratch = s,
            team,
        )
        @test dest==before
        f.dims=(5, 3)
        rhs=FaerMatrix(b)
        rhs.dims=(4, 2)
        @test_throws Colmat.RustCall.RustError ldiv!(dest, f, rhs; scratch = s, team)
        @test dest==before
        rhs.dims=(5, 2)
        close(rhs)
        @test_throws ArgumentError ldiv!(dest, f, rhs; scratch = s, team)
        task=fetch(@async begin
            d=zeros(T, 3, 2)
            ldiv!(d, f, b; scratch = s, team)
            d
        end)
        @test task≈pinv(A)*b atol=1e-3
        close(team)
        @test_throws ArgumentError ldiv!(dest, f, b; scratch = s, team)
        @test dest==before
        close(f)
        @test_throws ArgumentError scratch_requirements(ldiv!, f, b)
        close(s)
        close(a)
    end
    f=qr(FaerSparseMatrix(sparse([2.0 0; 0 3; 0 0])))
    b=ones(ComplexF64, 3, 2) .* (1+im)
    dest=fill(9.0, 2, 2)
    s=FaerScratch(ldiv!, f, b)
    @test_throws InexactError ldiv!(dest, f, b; scratch = s)
    @test dest==fill(9.0, 2, 2)
    native=FaerMatrix(dest)
    @test_throws InexactError ldiv!(native, f, b; scratch = s)
    @test Matrix(native)==dest
    close(s)
    close(native)
    close(f)
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
@testset "original sparse QR precision after promotion" begin
    for T in (Float32, ComplexF32), algorithm in (:auto, :simplicial, :supernodal)
        a=FaerSparseMatrix(sparse(T[1 0; 0 1e-8; 0 0]))
        f=qr(a; algorithm)
        b=ones(ComplexF64, 3, 1)
        s=FaerScratch(ldiv!, f, b)
        @test Matrix(\(f, b; scratch = s))≈ComplexF64[1; 0;;] atol=1e-5
        reserve!(s, ldiv!, f, b; rtol = 0)
        @test Matrix(\(f, b; scratch = s, rtol = 0))[2, 1]≈1/Float64(Float32(1e-8)) rtol=1e-5
        close(s)
        close(f)
        close(a)
    end
end

@testset "borrowed full-rank sparse QR resources avoid dense spectral workspace" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        algorithm in (:auto, :simplicial, :supernodal)

        n=80
        m=100
        A=zeros(T, m, n)
        for j = 1:n
            A[j, j]=T(4+j/n)
            A[j+1, j]=T<:Complex ? T(0.01+0.02im) : T(0.01)
        end
        a=FaerSparseMatrix(sparse(A))
        f=qr(a; algorithm, ordering = reverse(collect(1:n)))
        close(a)
        close(f.source)
        close(f.plan)
        b=ones(T, m, 2)
        team=FaerTeam(2)
        req=scratch_requirements(ldiv!, f, b; team)
        @test req.bytes<sizeof(T)*n*n
        work=FaerScratch(ldiv!, f, b; team)
        x=zeros(T, n, 2)
        @test ldiv!(x, f, b; scratch = work, team)===x
        @test x≈A\b atol=1e-3
        close(work)
        close(team)
        close(f)
    end
end
