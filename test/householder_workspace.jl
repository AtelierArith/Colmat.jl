using Colmat, Test, LinearAlgebra, Random
function hhw_fixture(T, m, k, b)
    v=zeros(T, m, k)
    for j = 1:k
        v[j, j]=1
        for i = (j+1):m
            v[i, j]=T(0.07sin(i+2j))
            T<:Complex && (v[i, j]+=T(0.03im*cos(2i+j)))
        end
    end
    tau=[sum(abs2, v[:, j])/2 for j = 1:k]
    q=FaerHouseholderSequence(v, tau; blocksize = b)
    ref=Matrix{T}(I, m, m)
    for j = 1:k
        ref=ref*(Matrix{T}(I, m, m)-v[:, j]*v[:, j]'/tau[j])
    end
    q, ref
end
@testset "Householder caller workspaces and teams" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        (m, k, b) in ((0, 0, 1), (4, 0, 3), (1, 1, 1), (7, 7, 3), (9, 5, 2), (129, 65, 32)),
        kind in (:elementary, :sequence)

        h, ref=if kind===:elementary
            x=T[1+i/10 for i = 1:m]
            T<:Complex && (x .*= T(1+0.2im))
            h=householder(x)
            v=Vector(h.v)
            h, Matrix{T}(I, m, m)-v*v'/h.tau
        else
            hhw_fixture(T, m, k, b)
        end
        for (g, gold) in (
                (h, ref),
                (transpose(h), transpose(ref)),
                (adjoint(h), adjoint(ref)),
                (conj(h), conj(ref)),
            ),
            right in (false, true)

            a=T[sin(i+2j) for i = 1:(right ? 3 : m), j = 1:(right ? m : 3)]
            expected=right ? a*gold : gold*a
            team=FaerTeam(2)
            fun=right ? rmul! : lmul!
            args=right ? (a, g) : (g, a)
            req=scratch_requirements(fun, args...; team)
            work=FaerScratch(fun, args...; team)
            cap=scratch_capacity(work)
            for native in (false, true)
                dest=native ? FaerMatrix(a) : copy(a)
                callargs=right ? (dest, g) : (g, dest)
                @test fun(callargs...; scratch = work, team)===dest
                @test Matrix(dest)≈expected atol=1500eps(real(T))*max(m, 1) rtol=1500eps(
                    real(T),
                )*max(m, 1)
                @test scratch_capacity(work)==cap
                native && close(dest)
            end
            dest=copy(a)
            @test mul!(
                dest,
                right ? a : g,
                right ? g : a,
                T(2),
                T(3);
                scratch = work,
                team,
            )===dest
            @test dest≈2expected+3a atol=2000eps(real(T))*max(m, 1) rtol=2000eps(real(T))*max(
                m,
                1,
            )
            if req.bytes>0
                short=FaerScratch(req.bytes-1; alignment = req.alignment)
                dest=copy(a)
                args=right ? (dest, g) : (g, dest)
                @test_throws ArgumentError fun(args...; scratch = short, team)
                @test dest==a
                reserve!(short, fun, args...; team)
                @test fun(args...; scratch = short, team)===dest
                @test dest≈expected atol=1500eps(real(T))*max(m, 1) rtol=1500eps(real(T))*max(
                    m,
                    1,
                )
                close(short)
            end
            dest=copy(a)
            invfun=right ? rdiv! : ldiv!
            args=right ? (dest, g) : (g, dest)
            inversework=FaerScratch(invfun, args...; team)
            @test invfun(args...; scratch = inversework, team)===dest
            @test dest≈(right ? a*gold' : gold'*a) atol=1500eps(real(T))*max(m, 1) rtol=1500eps(
                real(T),
            )*max(m, 1)
            close(inversework)
            close(work)
            close(team)
            right && g!==h && close(g)
        end
        close(h)
    end
end
@testset "Householder resources: promotion, alias, views, vectors, failures" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        for (h, ref) in
            (hhw_fixture(T, 7, 4, 2), (householder(T[1, 2, 3, 4, 5, 6, 7]), nothing))
            gold=isnothing(ref) ? Matrix(h) : ref
            b=ComplexF64[sin(i+2j)+0.1im for i = 1:7, j = 1:2]
            dest=copy(b)
            team=FaerTeam(2)
            s=FaerScratch(mul!, dest, h, b, 2.0, 3.0; team)
            @test mul!(dest, h, dest, 2.0, 3.0; scratch = s, team)===dest
            @test dest≈2gold*b+3b rtol=400eps(real(T)) atol=400eps(real(T))
            backing=zeros(ComplexF64, 14, 4)
            v=view(backing, 1:2:14, 1:2:4)
            copyto!(v, b)
            @test lmul!(h, v; scratch = s, team)===v
            @test v≈gold*b rtol=400eps(real(T)) atol=400eps(real(T))
            nb=FaerMatrix(b)
            nv=view(nb, 1:7, 1:2)
            @test lmul!(h, nv; scratch = s, team)===nv
            @test Matrix(nv)≈gold*b rtol=400eps(real(T)) atol=400eps(real(T))
            x=copy(b[:, 1])
            sx=FaerScratch(lmul!, h, x; team)
            @test lmul!(h, x; scratch = sx, team)===x
            @test x≈gold*b[:, 1] rtol=400eps(real(T)) atol=400eps(real(T))
            nx=FaerVector(b[:, 1])
            lmul!(h, nx; scratch = sx, team)
            @test Vector(nx)≈gold*b[:, 1] rtol=400eps(real(T)) atol=400eps(real(T))
            copied=deepcopy(h)
            close(h)
            @test_throws ArgumentError lmul!(h, dest; scratch = s, team)
            tasks=[@async begin
                y=copy(b)
                lmul!(copied, y; scratch = s, team)
                y
            end for _ = 1:3]
            @test all(
                y->isapprox(y, gold*b; rtol = 400eps(real(T)), atol = 400eps(real(T))),
                fetch.(tasks),
            )
            close(team)
            before=copy(dest)
            @test_throws ArgumentError mul!(dest, copied, b, 0, 0; scratch = s, team)
            @test dest==before
            @test_throws ArgumentError lmul!(copied, dest; scratch = :invalid)
            close(s)
            @test_throws ArgumentError lmul!(copied, dest; scratch = s)
            @test dest==before
            close(copied)
            close(nb)
            close(nx)
            close(sx)
        end
    end
end
@testset "Householder native metadata validation" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        h, ref=hhw_fixture(T, 5, 3, 2)
        h.n=6
        a=ones(ComplexF64, 6, 2)
        @test_throws Colmat.RustCall.RustError scratch_requirements(lmul!, h, a)
        @test_throws Colmat.RustCall.RustError lmul!(h, a)
        @test a==ones(6, 2)
        h.n=5
        h.k=2
        @test_throws Colmat.RustCall.RustError scratch_requirements(lmul!, h, ones(T, 5, 2))
        h.k=3
        h.blocksize=4
        @test_throws Colmat.RustCall.RustError lmul!(h, ones(T, 5, 2))
        close(h)
    end
end
@testset "Householder workspace rejection and inverse destinations" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), sequence in (false, true)
        h=sequence ? first(hhw_fixture(T, 5, 3, 2)) : householder(T[1, 2, 3, 4, 5])
        a=T[sin(i+2j) for i = 1:5, j = 1:2]
        gold=Matrix(h)
        dest=copy(a)
        work=FaerScratch(ldiv!, dest, h, a)
        @test ldiv!(dest, h, a; scratch = work)===dest
        @test dest≈gold'*a rtol=400eps(real(T)) atol=400eps(real(T))
        x=copy(a[:, 1])
        y=copy(x)
        reserve!(work, mul!, y, h, x)
        @test mul!(y, h, x; scratch = work)===y
        @test y≈gold*x rtol=400eps(real(T)) atol=400eps(real(T))
        @test_throws ArgumentError mul!(dest, h, a, 0, 0; scratch = :invalid)
        before=copy(dest)
        @test_throws ArgumentError lmul!(h, dest; team = :invalid)
        @test dest==before
        req=scratch_requirements(lmul!, h, a)
        if req.alignment>1
            weak=FaerScratch(req.bytes; alignment = 1)
            @test_throws ArgumentError lmul!(h, dest; scratch = weak)
            @test dest==before
            close(weak)
        end
        native=FaerMatrix(a)
        close(native)
        @test_throws ArgumentError scratch_requirements(lmul!, h, native)
        h.n=6
        @test_throws Colmat.RustCall.RustError lmul!(
            h,
            ones(ComplexF64, 6, 2);
            scratch = work,
        )
        close(h)
        close(work)
    end
end
