using Colmat, Test, LinearAlgebra, Random, SparseArrays
function qzc_check(f, a, b)
    n=size(a, 1)
    R=typeof(real(zero(eltype(a))))
    first=f isa FaerGeneralizedHessenberg ? f.H : f.S
    q, z, s, t=map(Matrix, (f.Q, f.Z, first, f.T))
    scale=max(n, 1)*eps(R)*max(norm(a)+norm(b), one(R))
    @test (norm(q*s*z'-a)+norm(q*t*z'-b))/scale < 20
    @test norm(q'*q-I)/(max(n, 1)*eps(R)) < 20
    @test norm(z'*z-I)/(max(n, 1)*eps(R)) < 20
    @test t==triu(t)
    @test s==triu(s, eltype(a)<:Real || f isa FaerGeneralizedHessenberg ? -1 : 0)
    if !(f isa FaerGeneralizedHessenberg)
        @test all(isfinite, f.α)&&all(isfinite, f.β)
    end
end
@testset "QZ/Hessenberg construction declared workspace and teams" begin
    rng=MersenneTwister(64931)
    for T in (Float32, Float64, ComplexF32, ComplexF64), width in (1, 2)
        team=FaerTeam(width)
        for n in (0, 1, 3, 11, 35, 65)
            a=Matrix{T}(I, n, n)+T(0.01)*randn(rng, T, n, n)
            b=T(2)*Matrix{T}(I, n, n)+T(0.01)*randn(rng, T, n, n)
            aa, bb=FaerMatrix(a), FaerMatrix(b)
            for fn in (hessenberg, schur)
                req=scratch_requirements(fn, aa, bb; team)
                work=FaerScratch(req.bytes; alignment = req.alignment)
                cap=scratch_capacity(work)
                f=fn(aa, bb; scratch = work, team)
                qzc_check(f, a, b)
                @test scratch_capacity(work)==cap
                @test Matrix(aa)==a && Matrix(bb)==b
                bang=fn===schur ? schur! : hessenberg!
                @test scratch_requirements(bang, aa, bb; team)==req
                x, y=copy(aa), copy(bb)
                vx=view(x, :, :)
                vy=view(y, :, :)
                g=bang(x, y; scratch = work, team)
                @test (fn===schur ? g.S : g.H)===x && g.T===y
                @test Matrix(vx)==Matrix(x) && Matrix(vy)==Matrix(y)
                qzc_check(g, a, b)
                close(g)
                close(f)
                close(work)
            end
            h=hessenberg(aa, bb; team)
            old=map(Matrix, (h.H, h.T, h.Q, h.Z))
            work=FaerScratch(schur, h; team)
            f=schur(h; scratch = work, team)
            qzc_check(f, a, b)
            @test map(Matrix, (h.H, h.T, h.Q, h.Z))==old
            @test scratch_requirements(schur!, h; team)==scratch_requirements(
                schur,
                h;
                team,
            )
            hh=deepcopy(h)
            owners=(hh.H, hh.T, hh.Q, hh.Z)
            g=schur!(hh; scratch = work, team)
            @test (g.S, g.T, g.Q, g.Z)===owners
            qzc_check(g, a, b)
            close(g)
            close(f)
            close(h)
            close(work)
            close(aa)
            close(bb)
        end
        close(team)
    end
end
@testset "QZ workspace reserve, reject, and transaction boundaries" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), fn in (hessenberg, schur)
        a=FaerMatrix(T[4 1; 2 3])
        b=FaerMatrix(T[2 0; 0 2])
        req=scratch_requirements(fn, a, b)
        @test req.bytes>0
        short=FaerScratch(req.bytes-1; alignment = req.alignment)
        orig=map(Matrix, (a, b))
        @test_throws ArgumentError fn(a, b; scratch = short)
        bang=fn===schur ? schur! : hessenberg!
        @test_throws ArgumentError bang(a, b; scratch = short)
        @test map(Matrix, (a, b))==orig
        @test reserve!(short, fn, a, b)===short
        f=fn(a, b; scratch = short)
        qzc_check(f, orig...)
        close(f)
        weak=FaerScratch(req.bytes; alignment = 1)
        @test_throws ArgumentError fn(a, b; scratch = weak)
        @test reserve!(weak, bang, a, b)===weak
        close(fn(a, b; scratch = weak))
        close(short)
        close(weak)
        @test_throws ArgumentError fn(a, b; scratch = short)
        @test_throws ArgumentError fn(a, b; scratch = 1)
        @test_throws ArgumentError fn(a, b; team = 1)
        team=FaerTeam(2)
        close(team)
        @test_throws ArgumentError fn(a, b; team)
        @test_throws ArgumentError bang(a, a)
        close(a)
        close(b)
    end
    for fn in (schur, schur!)
        a=FaerMatrix([4.0 1; 2 3])
        b=FaerMatrix([2.0 0; 0 2])
        h=hessenberg(a, b)
        req=scratch_requirements(fn, h)
        work=FaerScratch(req.bytes-1; alignment = req.alignment)
        old=map(Matrix, (h.H, h.T, h.Q, h.Z))
        @test_throws ArgumentError fn(h; scratch = work)
        @test map(Matrix, (h.H, h.T, h.Q, h.Z))==old
        reserve!(work, fn, h)
        close(fn(h; scratch = work))
        close(work)
        close(h)
        close(a)
        close(b)
    end
end
@testset "Mixed sources, strided views, and shared resource Tasks" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        a=T[4 1; 2 3]
        b=ComplexF64[2+0.1im 0.2; 0 3-0.2im]
        parent=FaerMatrix([a zeros(T, 2, 2); zeros(T, 2, 4)])
        av=view(parent, 1:2, 1:2)
        rv=view(parent, 2:-1:1, 2:-1:1)
        bb=FaerMatrix(b)
        for (x, y) in ((av, bb), (rv, bb), (av, b), (a, bb), (bb, av)),
            fn in (schur, hessenberg)

            work=FaerScratch(fn, x, y)
            expected=(Matrix(x), Matrix(y))
            for task in [Threads.@spawn fn(x, y; scratch = work) for _ = 1:3]
                f=fetch(task)
                qzc_check(f, ComplexF64.(expected[1]), ComplexF64.(expected[2]))
                close(f)
            end
            close(work)
        end
        close(parent)
        close(bb)
    end
    @test isempty(
        detect_ambiguities(Colmat, Base, LinearAlgebra, SparseArrays; recursive = true),
    )
end
struct QzcNoRead <: AbstractMatrix{ComplexF64} end
Base.size(::QzcNoRead) = (2, 2)
Base.getindex(::QzcNoRead, i::Int, j::Int) = error("unexpected coefficient conversion")
@testset "Native metadata and resource preflight precedes conversion" begin
    a=FaerMatrix(Float32[4 1; 2 3])
    b=QzcNoRead()
    req=scratch_requirements(schur, a, b)
    @test req.bytes>0
    short=FaerScratch(0)
    @test_throws ArgumentError schur(a, b; scratch = short)
    close(short)
    @test_throws ArgumentError hessenberg(a, b; scratch = short)
    a.dims=(3, 3)
    for fn in (schur, hessenberg)
        @test_throws Colmat.RustCall.RustError scratch_requirements(fn, a, b)
        @test_throws Colmat.RustCall.RustError fn(a, b)
    end
    a.dims=(2, 2)
    close(a)
    @test_throws ArgumentError scratch_requirements(schur, a, b)
    a=FaerMatrix([4.0 1; 2 3])
    b=FaerMatrix([2.0 0; 0 2])
    h=hessenberg(a, b)
    for m in (h.H, h.T, h.Q, h.Z)
        m.dims=(3, 3)
    end
    @test_throws Colmat.RustCall.RustError scratch_requirements(schur, h)
    @test_throws Colmat.RustCall.RustError schur(h)
    for m in (h.H, h.T, h.Q, h.Z)
        m.dims=(2, 2)
    end
    close(h)
    close(a)
    close(b)
end

@testset "Construction owners outlive inputs and resources" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), fn in (schur, hessenberg)
        aa, bb=FaerMatrix(T[4 1; 2 3]), FaerMatrix(T[2 0; 0 2])
        work=FaerScratch(fn, aa, bb)
        team=FaerTeam(2)
        reserve!(work, fn, aa, bb; team)
        f=fn(aa, bb; scratch = work, team)
        close(work)
        close(team)
        close(aa)
        close(bb)
        GC.gc()
        qzc_check(f, T[4 1; 2 3], T[2 0; 0 2])
        close(f)
    end
    a=FaerMatrix(Float32[4 1; 2 3])
    b=FaerMatrix(Float64[2 0; 0 2])
    a.dims=(3, 3)
    for x in (a, Hermitian(a), Symmetric(a), transpose(a), adjoint(a)),
        fn in (schur, hessenberg)

        @test_throws Colmat.RustCall.RustError scratch_requirements(fn, x, b)
        @test_throws Colmat.RustCall.RustError fn(x, b)
    end
    a.dims=(2, 2)
    close(a)
    close(b)
end

@testset "Native algorithm failures preserve staged destinations" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        a=FaerMatrix(T[4 1; 2 3])
        b=FaerMatrix(T[2 0; 0 2])
        work=FaerScratch(schur, a, b)
        a[1, 1]=T(NaN)
        old=map(Matrix, (a, b))
        for fn in (schur!, hessenberg!)
            @test_throws Colmat.RustCall.RustError fn(a, b; scratch = work)
            @test isequal(map(Matrix, (a, b)), old)
        end
        a[1, 1]=T(4)
        h=hessenberg(a, b)
        h.T[2, 1]=one(T)
        reserve!(work, schur, h)
        oldh=map(Matrix, (h.H, h.T, h.Q, h.Z))
        @test_throws Colmat.RustCall.RustError schur!(h; scratch = work)
        @test map(Matrix, (h.H, h.T, h.Q, h.Z))==oldh
        close(h)
        close(work)
        close(a)
        close(b)
    end
end
@testset "Homogeneous regular and singular pencils with scratch" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        team=FaerTeam(2)
        for (ad, bd) in ((T[1, 2], T[0, 1]), (T[1, 0, 2], T[0, 0, 1]))
            a=FaerMatrix(Matrix(Diagonal(ad)))
            b=FaerMatrix(Matrix(Diagonal(bd)))
            work=FaerScratch(schur, a, b; team)
            h=hessenberg(a, b; scratch = work, team)
            for f in (schur(a, b; scratch = work, team), schur(h; scratch = work, team))
                ss, tt=Matrix(f.S), Matrix(f.T)
                @test ss==triu(ss) && tt==triu(tt)
                # A globally singular pencil has no invariant finite spectrum.
                # Its homogeneous metadata must instead match computed Schur blocks.
                for i in eachindex(ad)
                    if iszero(ss[i, i])&&iszero(tt[i, i])
                        @test iszero(f.α[i])&&iszero(f.β[i])
                    else
                        @test !iszero(f.α[i])||!iszero(f.β[i])
                        scale=max(
                            abs(f.α[i])*abs(tt[i, i])+abs(f.β[i])*abs(ss[i, i]),
                            one(real(zero(T))),
                        )
                        @test abs(f.α[i]*tt[i, i]-f.β[i]*ss[i, i]) <=
                              20*eps(typeof(real(zero(T))))*scale
                    end
                end
                if length(ad)==2
                    @test count(i->iszero(f.β[i])&&!iszero(f.α[i]), 1:2)==1
                    @test count(i->!iszero(f.β[i])&&isapprox(f.α[i]/f.β[i], 2), 1:2)==1
                end
                qzc_check(f, Matrix(a), Matrix(b))
                close(f)
            end
            close(h)
            close(work)
            close(a)
            close(b)
        end
        close(team)
    end
end

@testset "Structured external wrappers validate native payloads" begin
    v=FaerVector(Float32[3, 4, 5])
    u=FaerVector(Float32[0.1, 0.2])
    l=FaerVector(Float32[0.3, 0.4])
    b=FaerMatrix(Matrix{Float32}(I, 3, 3))
    wrappers=(Diagonal(v), Bidiagonal(v, u, :U), SymTridiagonal(v, u), Tridiagonal(l, v, u))
    for x in wrappers
        @test scratch_requirements(schur, x, b).bytes>0
    end
    v.len=4
    for x in wrappers
        @test_throws Colmat.RustCall.RustError scratch_requirements(schur, x, b)
    end
    v.len=3
    close(v)
    close(u)
    close(l)
    close(b)
end

@testset "Sparse external matrices validate original metadata" begin
    a=FaerSparseMatrix(sparse(Float32[4 1; 2 3]))
    p=FaerSparsity(a)
    b=FaerMatrix([2.0 0; 0 2])
    for x in (a, p)
        work=FaerScratch(schur, x, b)
        f=schur(x, b; scratch = work)
        qzc_check(f, Float64.(Matrix(x)), Matrix(b))
        close(f)
        close(work)
        saved=x.format
        x.format=:csr
        @test_throws Colmat.RustCall.RustError scratch_requirements(schur, x, b)
        @test_throws Colmat.RustCall.RustError schur(x, b)
        x.format=saved
    end
    close(a)
    close(p)
    close(b)
end
