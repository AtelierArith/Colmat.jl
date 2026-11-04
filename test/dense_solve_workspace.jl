using Test, Colmat, LinearAlgebra
function dsw_fixture(T, m, n; deficient = false, prefix = false)
    a=T[0.03sin(2i+j) for i = 1:m, j = 1:n]
    for i = 1:min(m, n)
        a[i, i]+=T(3+i)
    end
    T<:Complex && (a .+= T(0.02im) .* T[cos(i+2j) for i = 1:m, j = 1:n])
    if deficient && min(m, n)>0
        m>=n ? (a[:, end].=a[:, 1]) : (a[end, :].=a[1, :])
    end
    prefix && (a[:, 1:min(m, n)].=0)
    a
end
function dsw_check(f, a, T, S, team; check_default = false)
    tol=1000eps(real(T))*max(size(a)..., 1)
    for (g, ref) in ((f, a), (transpose(f), transpose(a)), (adjoint(f), adjoint(a))),
        right in (false, true)

        b=fill(
            S<:Complex ? S(1+0.2im) : one(S),
            right ? (2, size(ref, 2)) : (size(ref, 1), 2),
        )
        gold=right ? b*pinv(Matrix(ref)) : pinv(Matrix(ref))*b
        P=promote_type(T, S)
        dest=zeros(P, size(gold))
        fun=right ? rdiv! : ldiv!
        args=right ? (dest, b, g) : (dest, g, b)
        req=scratch_requirements(fun, args...; team)
        work=FaerScratch(fun, args...; team)
        cap=scratch_capacity(work)
        if S!==T || check_default
            baseline=zeros(P, size(gold))
            baselineargs=right ? (baseline, b, g) : (baseline, g, b)
            @test fun(baselineargs...; team)===baseline
            @test baseline≈gold atol=tol*max(norm(gold), 1) rtol=tol
        end
        for _ = 1:2
            @test fun(args...; scratch = work, team)===dest
            @test dest≈gold atol=tol*max(norm(gold), 1) rtol=tol
            @test scratch_capacity(work)==cap
        end
        if req.bytes>0
            short=FaerScratch(req.bytes-1; alignment = req.alignment)
            before=copy(dest)
            @test_throws ArgumentError fun(args...; scratch = short, team)
            @test dest==before
            reserve!(short, fun, args...; team)
            @test fun(args...; scratch = short, team)===dest
            @test dest≈gold atol=tol*max(norm(gold), 1) rtol=tol
            close(short)
        end
        if size(ref, 1)==size(ref, 2)
            aliased=P.(b)
            callargs=right ? (aliased, g) : (g, aliased)
            @test fun(callargs...; scratch = work, team)===aliased
            @test aliased≈gold atol=tol*max(norm(gold), 1) rtol=tol
        end
        native=FaerMatrix(zeros(P, size(gold)))
        rhs=FaerMatrix(b)
        nativeargs=right ? (native, rhs, g) : (native, g, rhs)
        @test fun(nativeargs...; scratch = work, team)===native
        @test Matrix(native)≈gold atol=tol*max(norm(gold), 1) rtol=tol
        close(rhs)
        close(native)
        close(work)
    end
end
@testset "retained dense solve scratch" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        n in (0, 1, 5),
        threaded in (false, true)

        team=threaded ? FaerTeam(2) : nothing
        a=dsw_fixture(T, n, n)
        h=Matrix(a*a'+I)
        for (make, ref) in (
            (lu, a),
            (full_piv_lu, a),
            (qr, a),
            (x->qr(x, ColumnNorm()), a),
            (cholesky, h),
            (ldlt, h),
            (bunchkaufman, h),
        )
            source=FaerMatrix(ref)
            f=make(source)
            close(source)
            dsw_check(f, ref, T, T, team)
            close(f)
        end
        isnothing(team)||close(team)
    end
end
@testset "rectangular and deficient saved QR workspace" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        pivot in (false, true),
        (m, n, def, prefix) in (
            (7, 3, false, false),
            (3, 7, false, false),
            (7, 3, true, false),
            (3, 7, true, false),
            (4, 4, true, false),
            (0, 4, false, false),
            (4, 0, false, false),
            (3, 7, false, true),
        ),
        threaded in (false, true)

        a=dsw_fixture(T, m, n; deficient = def, prefix)
        f=pivot ? qr(FaerMatrix(a), ColumnNorm()) : qr(FaerMatrix(a))
        team=threaded ? FaerTeam(2) : nothing
        dsw_check(f, a, T, T, team)
        m!=n && @test_throws DimensionMismatch FaerScratch(ldiv!, f, zeros(T, m, 2))
        close(f)
        isnothing(team)||close(team)
    end
end
@testset "mixed dense solve scratch policy" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        S in (Float32, Float64, ComplexF32, ComplexF64),
        make in (lu, full_piv_lu, qr, x->qr(x, ColumnNorm()), cholesky, ldlt, bunchkaufman)

        a=dsw_fixture(T, 4, 4)
        a=Matrix(a*a'+I)
        f=make(FaerMatrix(a))
        team=FaerTeam(2)
        dsw_check(f, a, T, S, team)
        close(f)
        close(team)
    end
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        pivot in (false, true),
        (m, n) in ((7, 3), (3, 7))

        a=dsw_fixture(T, m, n; deficient = true)
        f=pivot ? qr(FaerMatrix(a), ColumnNorm()) : qr(FaerMatrix(a))
        team=FaerTeam(2)
        dsw_check(f, a, T, ComplexF64, team)
        close(f)
        close(team)
    end
end
@testset "blocked dense solve workspace" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        pivot in (false, true),
        (m, n) in ((129, 65), (33, 70))

        a=dsw_fixture(T, m, n)
        f=pivot ? qr(FaerMatrix(a), ColumnNorm()) : qr(FaerMatrix(a))
        team=FaerTeam(2)
        dsw_check(f, a, T, T, team)
        close(f)
        close(team)
    end
end
@testset "dense solve scratch lifetimes, views, vectors and errors" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        make in (lu, full_piv_lu, qr, x->qr(x, ColumnNorm()), cholesky, ldlt, bunchkaufman)

        a=dsw_fixture(T, 4, 4)
        a=Matrix(a*a'+I)
        f=make(FaerMatrix(a))
        team=FaerTeam(2)
        b=fill(T<:Complex ? T(1+0.2im) : one(T), 4, 2)
        gold=a\b
        backing=fill(T(42), 8, 4)
        v=view(backing, 1:2:8, 1:2:4)
        work=FaerScratch(ldiv!, v, f, b; team)
        @test ldiv!(v, f, b; scratch = work, team)===v
        @test v≈gold rtol=2000eps(real(T)) atol=2000eps(real(T))
        shared=copy(b)
        column=view(shared, :, 1:1)
        @test ldiv!(column, f, column; scratch = work, team)===column
        @test shared[:, 1]≈gold[:, 1] rtol=2000eps(real(T)) atol=2000eps(real(T))
        x=copy(b[:, 1])
        y=zeros(T, 4)
        reserve!(work, ldiv!, y, f, x; team)
        @test ldiv!(y, f, x; scratch = work, team)===y
        @test y≈gold[:, 1] rtol=2000eps(real(T)) atol=2000eps(real(T))
        nx=FaerVector(x)
        ny=FaerVector(zeros(T, 4))
        @test ldiv!(ny, f, nx; scratch = work, team)===ny
        @test Vector(ny)≈gold[:, 1] rtol=2000eps(real(T)) atol=2000eps(real(T))
        reserve!(work, ldiv!, v, f, b; team)
        copied=deepcopy(f)
        close(f)
        @test_throws ArgumentError scratch_requirements(ldiv!, f, b)
        tasks=[@async begin
            out=copy(b)
            ldiv!(copied, out; scratch = work, team)
            out
        end for _ = 1:3]
        @test all(
            out->isapprox(out, gold; rtol = 2000eps(real(T)), atol = 2000eps(real(T))),
            fetch.(tasks),
        )
        before=copy(v)
        @test_throws ArgumentError ldiv!(v, copied, b; scratch = :bad, team)
        @test v==before
        bad=copy(b)
        bad[1]=T(Inf)
        @test_throws Union{ArgumentError,Colmat.RustCall.RustError} ldiv!(
            v,
            copied,
            bad;
            scratch = work,
            team,
        )
        @test v==before
        req=scratch_requirements(ldiv!, v, copied, b; team)
        if req.alignment>1
            weak=FaerScratch(req.bytes; alignment = 1)
            @test_throws ArgumentError ldiv!(v, copied, b; scratch = weak, team)
            @test v==before
            close(weak)
        end
        close(team)
        @test_throws ArgumentError scratch_requirements(ldiv!, v, copied, b; team)
        @test_throws ArgumentError ldiv!(v, copied, b; scratch = work, team)
        @test v==before
        close(work)
        @test_throws ArgumentError ldiv!(v, copied, b; scratch = work)
        @test v==before
        close(copied)
        close(nx)
        close(ny)
    end
end
@testset "dense solve scratch authoritative factor metadata" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        f=qr(FaerMatrix(Matrix{T}(I, 4, 4)))
        f.dims=(4, 3)
        @test_throws Colmat.RustCall.RustError scratch_requirements(
            ldiv!,
            zeros(T, 3, 2),
            f,
            ones(T, 4, 2),
        )
        @test_throws Colmat.RustCall.RustError ldiv!(
            zeros(T, 3, 2),
            f,
            ones(T, 4, 2);
            scratch = FaerScratch(),
        )
        close(f)
    end
    f=lu(FaerMatrix(zeros(3, 3)); check = false)
    @test_throws SingularException FaerScratch(ldiv!, f, ones(3, 2))
    @test_throws SingularException ldiv!(
        zeros(3, 2),
        f,
        ones(3, 2);
        scratch = FaerScratch(),
    )
    close(f)
end
@testset "source-precision QR rank survives promotion" begin
    for T in (Float32, ComplexF32), pivot in (false, true)
        a=T[1 0; 0 1e-8]
        f=pivot ? qr(FaerMatrix(a), ColumnNorm()) : qr(FaerMatrix(a))
        b=ones(ComplexF64, 2, 1)
        expected=ComplexF64[1; 0;;]
        @test Matrix(f\FaerMatrix(b))≈expected
        x=zeros(ComplexF64, 2, 1)
        work=FaerScratch(ldiv!, x, f, b)
        @test ldiv!(x, f, b; scratch = work)===x
        @test x≈expected
        @test ldiv!(x, f, b)===x
        @test x≈expected
        close(work)
        close(f)
    end
end
@testset "dense solve scratch conversion staging" begin
    for T in (ComplexF32, ComplexF64)
        f=lu(FaerMatrix(T[3 im; 0 4]))
        b=ones(T, 2, 1)
        dest=fill(42.0, 2, 1)
        work=FaerScratch(ldiv!, dest, f, b)
        @test_throws InexactError ldiv!(dest, f, b; scratch = work)
        @test dest==fill(42.0, 2, 1)
        close(work)
        close(f)
    end
end
@testset "rectangular LU/full LU solve scratch" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        S in (T, ComplexF64),
        make in (lu, full_piv_lu),
        (m, n) in ((7, 3), (3, 7), (0, 3), (3, 0))

        a=dsw_fixture(T, m, n)
        f=make(FaerMatrix(a))
        team=FaerTeam(2)
        dsw_check(f, a, T, S, team)
        close(f)
        close(team)
    end
end
@testset "dense solve checks array owners before conversion" begin
    f=lu(FaerMatrix(Matrix{Float64}(I, 4, 4)))
    dest=ones(4, 2)
    b=ones(4, 2)
    work=FaerScratch(ldiv!, dest, f, b)
    for (Actual, Claimed) in ((Float32, ComplexF64), (ComplexF64, Float32))
        source=FaerMatrix(ones(Actual, 4, 2))
        invalid=FaerMatrix{Claimed,typeof(source.handle)}(source.handle, size(source))
        @test_throws Colmat.RustCall.RustError scratch_requirements(ldiv!, dest, f, invalid)
        @test_throws Colmat.RustCall.RustError ldiv!(dest, f, invalid; scratch = work)
        @test_throws Colmat.RustCall.RustError scratch_requirements(ldiv!, invalid, f, b)
        @test_throws Colmat.RustCall.RustError ldiv!(invalid, f, b; scratch = work)
        @test dest==ones(4, 2)
        close(invalid)
        close(source)
    end
    vector=FaerVector(ones(4))
    vector.len=5
    @test_throws Colmat.RustCall.RustError scratch_requirements(ldiv!, zeros(4), f, vector)
    @test_throws Colmat.RustCall.RustError ldiv!(zeros(4), f, vector; scratch = work)
    close(vector)
    source=FaerMatrix(ones(4, 2))
    source.dims=(4, 3)
    @test_throws Colmat.RustCall.RustError scratch_requirements(
        ldiv!,
        zeros(4, 3),
        f,
        source,
    )
    @test_throws Colmat.RustCall.RustError ldiv!(zeros(4, 3), f, source; scratch = work)
    close(source)
    close(f)
    close(work)
end
@testset "dense solve collection layouts" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        f=qr(FaerMatrix(Matrix{T}(I, 4, 4)))
        d=FaerDiagonal(T[1, 2, 3, 4])
        dest=zeros(T, 4, 4)
        work=FaerScratch(ldiv!, dest, f, d)
        @test ldiv!(dest, f, d; scratch = work)===dest
        @test dest≈Matrix(Diagonal(T[1, 2, 3, 4]))
        close(f)
        close(d)
        close(work)
        f=lu(FaerMatrix(reshape(T[2], 1, 1)))
        r=FaerRow(T[2, 4, 6, 8])
        dest=FaerRow(zeros(T, 4))
        work=FaerScratch(ldiv!, dest, f, r)
        @test ldiv!(dest, f, r; scratch = work)===dest
        @test Matrix(dest)≈reshape(T[1, 2, 3, 4], 1, 4)
        close(f)
        close(r)
        close(dest)
        close(work)
    end
end
