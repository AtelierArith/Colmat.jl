using Colmat, Test, LinearAlgebra, Random
function scw_check(f, a)
    n=size(a, 1)
    R=typeof(real(zero(eltype(a))))
    h=f isa FaerHessenberg ? Matrix(f.H) : Matrix(f.T)
    q=f isa FaerHessenberg ? Matrix(f.Q) : Matrix(f.Z)
    @test norm(q*h*q'-a)/(max(n, 1)*eps(R)*max(norm(a), one(R))) < 20
    @test norm(q'*q-I)/(max(n, 1)*eps(R)) < 20
    @test h==triu(h, f isa FaerHessenberg || eltype(a)<:Real ? -1 : 0)
    if !(f isa FaerHessenberg)
        @test all(isfinite, f.values)
        @test length(f.values)==n
        n<=17 && @test sort(f.values; by = x -> (real(x), imag(x))) ≈
              sort(eigvals(a); by = x -> (real(x), imag(x))) rtol=100*max(n, 1)*eps(R) atol=100*max(
            n,
            1,
        )*eps(R)
    end
end
@testset "Single Hessenberg/Schur construction scratch and teams" begin
    rng=MersenneTwister(65042)
    for T in (Float32, Float64, ComplexF32, ComplexF64), width in (1, 2)
        team=FaerTeam(width)
        for n in (0, 1, 3, 17, 65, 257)
            a=Matrix(Diagonal(T.(1:n)))+T(0.01)*randn(rng, T, n, n)
            aa=FaerMatrix(a)
            for fn in (hessenberg, schur)
                req=scratch_requirements(fn, aa; team)
                work=FaerScratch(fn, aa; team)
                cap=scratch_capacity(work)
                @test cap.bytes>=req.bytes && cap.alignment>=req.alignment
                f=fn(aa; scratch = work, team)
                scw_check(f, a)
                @test Matrix(aa)==a
                @test scratch_capacity(work)==cap
                bang=fn===schur ? schur! : hessenberg!
                @test scratch_requirements(bang, aa; team)==req
                x=copy(aa)
                vx=view(x, :, :)
                g=bang(x; scratch = work, team)
                @test (fn===schur ? g.T : g.H)===x
                @test Matrix(vx)==Matrix(x)
                scw_check(g, a)
                close(g)
                close(f)
                close(work)
            end
            h=hessenberg(aa; team)
            original=map(Matrix, (h.H, h.Q))
            work=FaerScratch(schur, h; team)
            cap=scratch_capacity(work)
            f=schur(h; scratch = work, team)
            scw_check(f, a)
            @test map(Matrix, (h.H, h.Q))==original
            vh, vq=view(h.H, :, :), view(h.Q, :, :)
            g=schur!(h; scratch = work, team)
            @test g.T===h.H && g.Z===h.Q
            @test Matrix(vh)==Matrix(g.T) && Matrix(vq)==Matrix(g.Z)
            scw_check(g, a)
            @test scratch_requirements(schur!, h; team)==scratch_requirements(
                schur,
                h;
                team,
            )
            @test scratch_capacity(work)==cap
            close(f)
            close(g)
            close(h)
            close(work)
            close(aa)
        end
        close(team)
    end
end
@testset "Single Schur capacities, reserve and failed writeback" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), fn in (hessenberg, schur)
        a=FaerMatrix(T[4 1; 2 3])
        old=Matrix(a)
        team=FaerTeam(2)
        req=scratch_requirements(fn, a; team)
        short=FaerScratch(req.bytes-1; alignment = req.alignment)
        bang=fn===schur ? schur! : hessenberg!
        for op in (fn, bang)
            @test_throws ArgumentError op(a; scratch = short, team)
            @test Matrix(a)==old
            @test_throws ArgumentError op(a; scratch = 1)
            @test_throws ArgumentError op(a; team = 1)
        end
        reserve!(short, fn, a; team)
        close(fn(a; scratch = short, team))
        if req.alignment>1
            unaligned=FaerScratch(req.bytes; alignment = 1)
            @test_throws ArgumentError fn(a; scratch = unaligned, team)
            close(unaligned)
        end
        a[1, 1]=T(NaN)
        for op in (fn, bang)
            @test_throws Colmat.RustCall.RustError op(a; scratch = short, team)
            @test isequal(Matrix(a), T[NaN 1; 2 3])
        end
        close(short)
        @test_throws ArgumentError fn(a; scratch = short)
        close(team)
        @test_throws ArgumentError scratch_requirements(fn, a; team)
        @test_throws ArgumentError fn(a; team)
        close(a)
    end
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        aa=FaerMatrix(T[4 1 2; 2 3 0; 1 0 5])
        h=hessenberg(aa)
        work=FaerScratch(schur, h)
        original=map(Matrix, (h.H, h.Q))
        short=FaerScratch(0)
        for op in (schur, schur!)
            @test_throws ArgumentError op(h; scratch = short)
            @test map(Matrix, (h.H, h.Q))==original
        end
        reserve!(short, schur!, h)
        close(schur(h; scratch = short))
        h.H[3, 1]=one(T)
        broken=map(Matrix, (h.H, h.Q))
        for op in (schur, schur!)
            @test_throws Colmat.RustCall.RustError op(h; scratch = work)
            @test map(Matrix, (h.H, h.Q))==broken
        end
        close(work)
        close(short)
        close(h)
        close(aa)
    end
end
@testset "Single Schur views, lifetime and sharing" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        a=T[4 1 2 0; 2 3 0 1; 1 0 5 0; 0 2 0 6]
        aa=FaerMatrix(a)
        team=FaerTeam(2)
        for av in (view(aa, :, :), view(aa, 4:-1:1, 4:-1:1), view(aa, 1:2:4, 1:2:4)),
            fn in (hessenberg, schur)

            expected=Matrix(av)
            work=FaerScratch(fn, av; team)
            for task in [Threads.@spawn fn(av; scratch = work, team) for _ = 1:3]
                f=fetch(task)
                scw_check(f, expected)
                close(f)
            end
            close(work)
        end
        for fn in (schur, hessenberg)
            work=FaerScratch(fn, aa; team)
            f=fn(aa; scratch = work, team)
            close(work)
            close(aa)
            GC.gc()
            scw_check(f, a)
            close(f)
            aa=FaerMatrix(a)
        end
        close(aa)
        close(team)
    end
    @test isempty(detect_ambiguities(Colmat, Base, LinearAlgebra; recursive = true))
end
@testset "Single Schur original and saved native metadata" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        a=FaerMatrix(T[4 1; 2 3])
        av=view(a, :, :)
        a.dims=(3, 3)
        for x in (a, av), op in (schur, hessenberg)
            @test_throws r"current native dimensions" scratch_requirements(op, x)
            @test_throws r"current native dimensions" op(x)
        end
        a.dims=(2, 2)
        h=hessenberg(a)
        for m in (h.H, h.Q)
            m.dims=(3, 3)
            for op in (schur, schur!)
                @test_throws r"current native dimensions" scratch_requirements(op, h)
                @test_throws r"current native dimensions" op(h)
            end
            m.dims=(2, 2)
        end
        close(h.Q)
        @test_throws ArgumentError scratch_requirements(schur, h)
        @test_throws ArgumentError schur(h)
        close(h)
        close(a)
        @test_throws ArgumentError schur(av)
        @test_throws ArgumentError scratch_requirements(hessenberg, av)
    end
    for T in (Float32, ComplexF32, ComplexF64)
        a=FaerMatrix(T[4 1; 2 3])
        h=hessenberg(a)
        U=T===Float32 ? ComplexF64 : Float32
        replacement=Matrix{U}(I, 2, 2)
        for m in (a, h.H, h.Q)
            original=Matrix(m)
            GC.@preserve replacement Colmat._native_call(
                Colmat.RustCrate.load,
                m,
                Ptr{UInt8}(pointer(replacement)),
                UInt(2),
                UInt(2),
                Colmat._tag(U),
            )
            if m===a
                for op in (schur, hessenberg, schur!, hessenberg!)
                    @test_throws r"requested scalar type" op(a)
                    @test_throws r"requested scalar type" scratch_requirements(op, a)
                end
            else
                for op in (schur, schur!)
                    @test_throws r"requested scalar type" op(h)
                    @test_throws r"requested scalar type" scratch_requirements(op, h)
                end
            end
            GC.@preserve original Colmat._native_call(
                Colmat.RustCrate.load,
                m,
                Ptr{UInt8}(pointer(original)),
                UInt(2),
                UInt(2),
                Colmat._tag(T),
            )
        end
        close(a)
        close(h)
    end
end
@testset "Square matrix views retaining row collection owners" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        value=T<:Complex ? T(2+im) : T(2)
        row=FaerRow(T[3, value, 5])
        v=view(row, 1:1, 2:2)
        expected=reshape(T[value], 1, 1)
        team=FaerTeam(2)
        for fn in (schur, hessenberg), resource in (nothing, team)
            work=FaerScratch(fn, v; team = resource)
            f=fn(v; scratch = work, team = resource)
            scw_check(f, expected)
            @test Matrix(row)==reshape(T[3, value, 5], 1, 3)
            close(f)
            close(work)
        end
        row.len=4
        for fn in (schur, hessenberg)
            @test_throws Colmat.RustCall.RustError scratch_requirements(fn, v)
            @test_throws Colmat.RustCall.RustError fn(v)
        end
        row.len=3
        close(row)
        @test_throws ArgumentError schur(v)
        @test_throws ArgumentError scratch_requirements(hessenberg, v)
        close(team)
    end
end
