using Test, Colmat, LinearAlgebra, Libdl, Random

const TGEVC_NATIVE = Ref{Any}(nothing)
tgevc_library() = TGEVC_NATIVE[] === nothing ? (TGEVC_NATIVE[] = Libdl.dlopen(Colmat.backend_library())) : TGEVC_NATIVE[]

for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
    prefix = T === Float32 ? "s" : T === Float64 ? "d" : T === ComplexF32 ? "c" : "z"
    suffix = J === Int64 ? "_64_" : "_"
    @eval function tgevc_native_call(
        ::Type{$T}, ::Type{$J}, side, howmny, n, s::Matrix{$T}, t::Matrix{$T};
        select=fill($J(1), n), vl=zeros($T, max(1, n), max(1, n)),
        vr=zeros($T, max(1, n), max(1, n)), mm=n, lds=max(1, n),
        ldp=max(1, n), ldvl=max(1, n), ldvr=max(1, n),
    )
        pointer = Libdl.dlsym(tgevc_library(), $(prefix * "tgevc" * suffix))
        m = Ref{$J}(-777)
        info = Ref{$J}(777)
        work = fill($T(NaN), max(1, 6n))
        ccall(
            pointer,
            Cvoid,
            (Ref{UInt8}, Ref{UInt8}, Ptr{$J}, Ref{$J}, Ptr{$T}, Ref{$J},
             Ptr{$T}, Ref{$J}, Ptr{$T}, Ref{$J}, Ptr{$T}, Ref{$J}, Ref{$J},
             Ref{$J}, Ptr{$T}, Ref{$J}),
            side, howmny, select, n, s, lds, t, ldp, vl, ldvl, vr, ldvr, mm,
            m, work, info,
        )
        (; info=info[], m=m[], vl=copy(vl), vr=copy(vr), work=work)
    end
end

function random_pencil(::Type{T}, n; seed=1) where {T}
    rng = MersenneTwister(seed)
    a = randn(rng, T, n, n)
    b = randn(rng, T, n, n)
    s, t, alpha, beta, q, z = LAPACK.gges!('V', 'V', copy(a), copy(b))
    return (; a, b, s, t, q, z, values=alpha ./ beta)
end

function residual_right(a, b, λ, x)
    isfinite(λ) || return norm(b * x) / (1 + norm(b) * norm(x))
    return norm(a * x - λ * (b * x)) / ((norm(a) + abs(λ) * norm(b)) * norm(x))
end

function residual_left(a, b, λ, y)
    isfinite(λ) || return norm(adjoint(b) * y) / (1 + norm(b) * norm(y))
    return norm(adjoint(a) * y - conj(λ) * (adjoint(b) * y)) /
        ((norm(a) + abs(λ) * norm(b)) * norm(y))
end

@testset "TGEVC complex back transform and selection" begin
    for (T, tol) in ((ComplexF32, 2f-3), (ComplexF64, 2e-11)),
        J in (Int32, Int64), n in (1, 2, 6), side in ('L', 'R', 'B'),
        howmny in ('A', 'S', 'B')

        pencil = random_pencil(T, n; seed=n)
        values = pencil.values
        select = fill(J(1), n)
        if howmny == 'S' && n > 1
            for j in 1:n
                select[j] = isodd(j) ? J(1) : J(0)
            end
        end
        vl = zeros(T, n, n)
        vr = zeros(T, n, n)
        if howmny == 'B'
            vl .= pencil.q
            vr .= pencil.z
        end
        result = tgevc_native_call(
            T, J, side, howmny, n, pencil.s, pencil.t;
            select=select, vl=vl, vr=vr, mm=n,
        )
        @test result.info == 0
        @test result.m == (howmny == 'S' ? sum(select) : n)
        selected = howmny == 'S' ? findall(!iszero, select) : collect(1:n)
        a, b = howmny == 'B' ? (pencil.a, pencil.b) : (pencil.s, pencil.t)
        if side in ('R', 'B')
            for (k, j) in enumerate(selected)
                @test residual_right(a, b, values[j], result.vr[:, k]) <= tol
            end
        end
        if side in ('L', 'B')
            for (k, j) in enumerate(selected)
                @test residual_left(a, b, values[j], result.vl[:, k]) <= tol
            end
        end
        for j in result.m+1:n
            @test all(iszero, result.vl[:, j])
            @test all(iszero, result.vr[:, j])
        end
    end
end

@testset "TGEVC real back transform and conjugate pairs" begin
    for (T, tol) in ((Float32, 3f-3), (Float64, 3e-11)),
        J in (Int32, Int64), n in (1, 3, 6), side in ('L', 'R', 'B'),
        howmny in ('A', 'B')

        pencil = random_pencil(T, n; seed=10 + n)
        vl = zeros(T, n, n)
        vr = zeros(T, n, n)
        howmny == 'B' && (vl .= pencil.q; vr .= pencil.z)
        result = tgevc_native_call(
            T, J, side, howmny, n, pencil.s, pencil.t;
            vl=vl, vr=vr, mm=n,
        )
        @test result.info == 0
        @test result.m == n
        a, b = howmny == 'B' ? (pencil.a, pencil.b) : (pencil.s, pencil.t)
        j = 1
        k = 1
        while j <= n
            pair = j < n && pencil.s[j + 1, j] != 0
            λ1 = pencil.values[j]
            λ2 = pencil.values[j + pair]
            if pair
                if side in ('R', 'B')
                    v = complex.(result.vr[:, k], result.vr[:, k + 1])
                    @test min(
                        residual_right(a, b, λ1, v),
                        residual_right(a, b, λ2, v),
                    ) <= tol
                end
                if side in ('L', 'B')
                    y = complex.(result.vl[:, k], result.vl[:, k + 1])
                    @test min(
                        residual_left(a, b, λ1, y),
                        residual_left(a, b, λ2, y),
                    ) <= tol
                end
                k += 2
                j += 2
            else
                if side in ('R', 'B')
                    @test residual_right(a, b, λ1, result.vr[:, k]) <= tol
                end
                if side in ('L', 'B')
                    @test residual_left(a, b, λ1, result.vl[:, k]) <= tol
                end
                k += 1
                j += 1
            end
        end
    end

    # An explicit real 2x2 block with complex eigenvalues and identity Q/Z.
    for (T, tol) in ((Float64, 1e-11),), J in (Int32, Int64)
        s = T[0 1; -1 0]
        t = Matrix{T}(I, 2, 2)
        r = tgevc_native_call(T, J, 'B', 'A', 2, s, t; mm=2)
        @test r.info == 0
        @test r.m == 2
        v = complex.(r.vr[:, 1], r.vr[:, 2])
        @test min(residual_right(s, t, im, v), residual_right(s, t, -im, v)) <= tol
        @test maximum(abs.(r.vr[:, 1]) .+ abs.(r.vr[:, 2])) ≈ 1 atol = 100tol
    end
end

@testset "TGEVC infinite eigenvalues and diagnostics" begin
    T, J = Float64, Int64
    s = T[1 0 0; 0 2 1; 0 0 3]
    t = T[1 0 0; 0 1 0; 0 0 0]
    r = tgevc_native_call(T, J, 'B', 'A', 3, s, t; mm=3)
    @test r.info == 0
    @test r.m == 3
    @test norm(t * r.vr[:, 3]) <= 1e-12 * norm(r.vr[:, 3])
    @test residual_right(s, t, Inf, r.vr[:, 3]) <= 1e-12
    @test residual_left(s, t, Inf, r.vl[:, 3]) <= 1e-12

    pencil = random_pencil(Float64, 4; seed=5)
    select = J[1, 0, 1, 0]
    for (side, how, n, expected) in (
        ('x', 'A', 4, -1),
        ('B', 'x', 4, -2),
        ('B', 'A', -1, -4),
    )
        r = tgevc_native_call(
            T, J, side, how, n, pencil.s, pencil.t; select=select, mm=4
        )
        @test r.info == expected
        @test r.m == -777
    end
    r = tgevc_native_call(
        T, J, 'B', 'S', 4, pencil.s, pencil.t; select=select, mm=1
    )
    @test r.info == -13
    for (lds, ldp, ldvl, ldvr, expected) in (
        (3, 4, 4, 4, -6),
        (4, 3, 4, 4, -8),
        (4, 4, 3, 4, -10),
        (4, 4, 4, 3, -12),
    )
        r = tgevc_native_call(
            T, J, 'B', 'A', 4, pencil.s, pencil.t;
            mm=4, lds=lds, ldp=ldp, ldvl=ldvl, ldvr=ldvr,
        )
        @test r.info == expected
    end
    bad = copy(pencil.s)
    bad[2, 1] = 1
    bad[3, 2] = 1
    r = tgevc_native_call(T, J, 'B', 'A', 4, bad, pencil.t; mm=4)
    @test r.info == -5
    block = T[0 1 0 0; -1 0 0 0; 0 0 2 1; 0 0 0 3]
    blockt = Matrix{T}(I, 4, 4)
    blockt[2, 2] = 0
    r = tgevc_native_call(T, J, 'B', 'A', 4, block, blockt; mm=4)
    @test r.info == -7
end
