using Test, LinearAlgebra, Libdl, Random, Colmat

# Concrete C signatures exercise both integer widths, not just LBT inventory.
for (T, prefix) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    I in (Int32, Int64)

    symbol = Symbol(prefix * "larf_" * (I === Int64 ? "64_" : ""))
    @eval function apply_larf!(side, v::Vector{$T}, inc, tau::$T, c::Matrix{$T}, ::Type{$I})
        f = Libdl.dlsym(Libdl.dlopen(Colmat.backend_library()), $(QuoteNode(symbol)))
        m, n = size(c)
        work = zeros($T, max(m, n))
        ccall(
            f,
            Cvoid,
            (
                Ref{UInt8},
                Ref{$I},
                Ref{$I},
                Ptr{$T},
                Ref{$I},
                Ref{$T},
                Ptr{$T},
                Ref{$I},
                Ptr{$T},
            ),
            side,
            m,
            n,
            v,
            inc,
            tau,
            c,
            max(1, m),
            work,
        )
        c
    end
end

@testset "Householder LARF direct ABI" begin
    Random.seed!(264)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        I in (Int32, Int64), side in ('L', 'R'),
        inc in (1, 2, -1, -2)

        m, n = 5, 3
        len = side == 'L' ? m : n
        v = randn(T, len)
        storage = fill(T(NaN), 1+(len-1)*abs(inc))
        indices = inc > 0 ? (1:inc:length(storage)) : (length(storage):inc:1)
        storage[indices] = v
        tau = T(0.3)
        h = Matrix{T}(LinearAlgebra.I, len, len) - tau*v*v'
        c = randn(T, m, n)
        expected = side == 'L' ? h*c : c*h
        @test apply_larf!(side, storage, inc, tau, copy(c), I) ≈ expected
        @test apply_larf!(side, storage, inc, zero(T), copy(c), I) == c
        @test size(apply_larf!(side, T[], inc, tau, zeros(T, 0, 0), I)) == (0, 0)
    end
end

for (T, prefix) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    I in (Int32, Int64)

    symbol = Symbol(prefix * "larfy_" * (I === Int64 ? "64_" : ""))
    @eval function apply_larfy!(u, v::Vector{$T}, inc, tau::$T, c::Matrix{$T}, ::Type{$I})
        f = Libdl.dlsym(Libdl.dlopen(Colmat.backend_library()), $(QuoteNode(symbol)))
        n = size(c, 1)
        work = zeros($T, n)
        ccall(
            f,
            Cvoid,
            (Ref{UInt8}, Ref{$I}, Ptr{$T}, Ref{$I}, Ref{$T}, Ptr{$T}, Ref{$I}, Ptr{$T}),
            u,
            n,
            v,
            inc,
            tau,
            c,
            max(1, n),
            work,
        )
        c
    end
end
@testset "Householder LARFY selected triangle" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        I in (Int32, Int64), u in ('U', 'L'),
        inc in (1, -2)

        n=5
        v=randn(T, n)
        storage=fill(T(NaN), 1+(n-1)*abs(inc))
        indices=inc>0 ? (1:inc:length(storage)) : (length(storage):inc:1)
        storage[indices]=v
        tau=T <: Real ? T(0.3) : T(0.3+0.2im)
        h=Matrix{T}(LinearAlgebra.I, n, n)-tau*v*v'
        a=randn(T, n, n)
        c=Matrix(Hermitian(a))
        expected=h*c*h'
        for j in 1:n, i in 1:n
            if (u=='U' && i>j) || (u=='L' && i<j)
                c[i, j]=T(NaN)
            end
        end
        result=apply_larfy!(u, storage, inc, tau, c, I)
        @test Matrix(Hermitian(result, Symbol(u))) ≈ expected
        for j in 1:n, i in 1:n
            if (u=='U' && i>j) || (u=='L' && i<j)
                @test isnan(result[i, j])
            end
        end
    end
end

for (T, prefix) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    I in (Int32, Int64)

    suffix=I===Int64 ? "64_" : ""
    ft=Symbol(prefix*"larft_"*suffix)
    fb=Symbol(prefix*"larfb_"*suffix)
    @eval function block_factor(
        d,
        s,
        n,
        v::Matrix{$T},
        tau::Vector{$T},
        ::Type{$I};
        library=Colmat.backend_library(),
    )
        k=length(tau)
        t=fill($T(NaN), k, k)
        f=Libdl.dlsym(Libdl.dlopen(library), $(QuoteNode(ft)))
        ccall(
            f,
            Cvoid,
            (
                Ref{UInt8},
                Ref{UInt8},
                Ref{$I},
                Ref{$I},
                Ptr{$T},
                Ref{$I},
                Ptr{$T},
                Ptr{$T},
                Ref{$I},
            ),
            d,
            s,
            n,
            k,
            v,
            size(v, 1),
            tau,
            t,
            k,
        )
        t
    end
    @eval function block_apply!(
        side, tr, d, s, v::Matrix{$T}, t::Matrix{$T}, c::Matrix{$T}, ::Type{$I}
    )
        m, n=size(c)
        k=size(t, 1)
        ldw=side=='L' ? n : m
        work=zeros($T, ldw, k)
        f=Libdl.dlsym(Libdl.dlopen(Colmat.backend_library()), $(QuoteNode(fb)))
        ccall(
            f,
            Cvoid,
            (
                Ref{UInt8},
                Ref{UInt8},
                Ref{UInt8},
                Ref{UInt8},
                Ref{$I},
                Ref{$I},
                Ref{$I},
                Ptr{$T},
                Ref{$I},
                Ptr{$T},
                Ref{$I},
                Ptr{$T},
                Ref{$I},
                Ptr{$T},
                Ref{$I},
            ),
            side,
            tr,
            d,
            s,
            m,
            n,
            k,
            v,
            size(v, 1),
            t,
            k,
            c,
            m,
            work,
            ldw,
        )
        c
    end
end
@testset "Householder compact block representation" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        I in (Int32, Int64), d in ('F', 'B'),
        s in ('C', 'R')

        order, k=6, 3
        b=randn(T, order, k)
        for j in 1:k
            pivot=d=='F' ? j : order-k+j
            b[pivot, j]=1
            b[d=='F' ? (1:(pivot - 1)) : ((pivot + 1):order), j].=0
        end
        tau=randn(T, k)/4
        tau[2]=0 # identity reflector within a nontrivial block
        h=Matrix{T}(LinearAlgebra.I, order, order)
        for j in (d=='F' ? (1:k) : (k:-1:1))
            h=h*(LinearAlgebra.I-tau[j]*b[:, j]*b[:, j]')
        end
        v=s=='C' ? copy(b) : copy(b')
        for j in 1:k, i in 1:order
            pivot=d=='F' ? j : order-k+j
            if (d=='F' && i<=pivot)||(d=='B' && i>=pivot)
                if s=='C'
                    v[i, j]=T(NaN)
                else
                    v[j, i]=T(NaN)
                end
            end
        end
        t=block_factor(d, s, order, v, tau, I)
        # The untouched OpenBLAS library supplies independent LAPACK factors.
        external_library=BLAS.lbt_find_backing_library("dgeqrf_", :ilp64).libname
        external_t=block_factor(d, s, order, v, tau, Int64; library=external_library)
        @test Matrix(d=='F' ? UpperTriangular(t) : LowerTriangular(t)) ≈
            Matrix(d=='F' ? UpperTriangular(external_t) : LowerTriangular(external_t))
        triangular=Matrix(d=='F' ? UpperTriangular(t) : LowerTriangular(t))
        @test Matrix{T}(LinearAlgebra.I, order, order)-b*triangular*b' ≈ h
        for side in ('L', 'R'), tr in ('N', T<:Real ? 'T' : 'C')
            c=randn(T, side=='L' ? order : 4, side=='L' ? 4 : order)
            op=tr=='N' ? h : h'
            expected=side=='L' ? op*c : c*op
            @test block_apply!(side, tr, d, s, v, t, copy(c), I) ≈ expected
            @test block_apply!(side, tr, d, s, v, external_t, copy(c), I) ≈ expected
        end
    end
end

for (T, prefix) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    I in (Int32, Int64),
    positive in (false, true)

    sym=Symbol(prefix*(positive ? "larfgp_" : "larfg_")*(I===Int64 ? "64_" : ""))
    @eval function generate_reflector(
        a::$T, x::Vector{$T}, inc, ::Type{$I}, ::Val{$positive}
    )
        head=Ref(a)
        tau=Ref(zero($T))
        n=1+cld(length(x), inc)
        f=Libdl.dlsym(Libdl.dlopen(Colmat.backend_library()), $(QuoteNode(sym)))
        ccall(f, Cvoid, (Ref{$I}, Ref{$T}, Ptr{$T}, Ref{$I}, Ref{$T}), n, head, x, inc, tau)
        head[], tau[], x
    end
end
@testset "Householder generation at scalar extremes" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        I in (Int32, Int64), positive in (false, true),
        inc in (1, 2)

        R=real(T)
        for scale in (one(R), floatmin(R), floatmax(R)/16),
            phase in (T(1), T(-1), T<:Real ? T(1) : T(im))

            a=phase*scale
            original=T[scale / 2, -scale / 4, scale / 8]
            storage=fill(T(NaN), 1+2inc)
            storage[1:inc:end]=original
            beta, tau, out=generate_reflector(a, storage, inc, I, Val(positive))
            v=vcat(one(T), out[1:inc:end])
            h=Matrix{T}(LinearAlgebra.I, 4, 4)-tau*v*v'
            normalized=vcat(a/scale, original/scale)
            @test h'*normalized ≈ vcat(beta/scale, zeros(T, 3)) atol=30eps(R)
            @test h'*h ≈ Matrix{T}(LinearAlgebra.I, 4, 4) atol=30eps(R)
            @test imag(beta)==0
            @test !positive || real(beta)>=0
            @test all(isnan, out[2:inc:(inc == 1 ? 0 : length(out))])
        end
        for a in (T(2), T(-2), T<:Real ? zero(T) : T(2im))
            beta, tau, _=generate_reflector(a, T[], inc, I, Val(positive))
            @test conj(1-tau)*a ≈ beta
            @test !positive || real(beta)>=0
        end
    end
end

for (T, prefix) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    I in (Int32, Int64)

    sym=Symbol(prefix*"larfx_"*(I===Int64 ? "64_" : ""))
    @eval function apply_larfx!(side, v::Vector{$T}, tau::$T, c::Matrix{$T}, ::Type{$I})
        m, n=size(c)
        work=zeros($T, max(m, n))
        f=Libdl.dlsym(Libdl.dlopen(Colmat.backend_library()), $(QuoteNode(sym)))
        GC.@preserve work ccall(
            f,
            Cvoid,
            (Ref{UInt8}, Ref{$I}, Ref{$I}, Ptr{$T}, Ref{$T}, Ptr{$T}, Ref{$I}, Ptr{$T}),
            side,
            m,
            n,
            v,
            tau,
            c,
            max(1, m),
            length(v)<=10 ? Ptr{$T}(C_NULL) : pointer(work),
        )
        c
    end
end
@testset "Householder LARFX and LBT registration" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        I in (Int32, Int64), side in ('L', 'R'),
        order in (1, 10, 11)

        v=randn(T, order)
        tau=T<:Real ? T(0.2) : T(0.2+0.1im)
        h=Matrix{T}(LinearAlgebra.I, order, order)-tau*v*v'
        c=randn(T, side=='L' ? order : 3, side=='L' ? 3 : order)
        @test apply_larfx!(side, v, tau, copy(c), I) ≈ (side=='L' ? h*c : c*h)
    end
    for prefix in ("s", "d", "c", "z"),
        family in ("larf", "larfx", "larfy", "larft", "larfb", "larfg", "larfgp"),
        abi in (:lp64, :ilp64)

        lib=BLAS.lbt_find_backing_library(prefix*family*"_", abi)
        @test lib !== nothing
        @test realpath(lib.libname)==realpath(Colmat.backend_library())
    end
end
