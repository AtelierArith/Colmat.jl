using Test, Colmat, LinearAlgebra, Libdl, Random
isdefined(@__MODULE__, :lapacke288_selected_entry) || include("lapacke_calls.jl")
for (T, prefix) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    J in (Int32, Int64)

    @eval function lapacke288_gpair(
        ::Type{$T},
        ::Type{$J},
        rq,
        high,
        layout,
        f,
        s,
        p,
        a,
        la,
        ta,
        b,
        lb,
        tb,
        w,
        lw;
        route=:dll,
    )
        name="LAPACKE_"*$prefix*(rq ? "ggrqf" : "ggqrf")*(high ? "" : "_work")
        symbol=name*($J===Int64 ? "64_" : "")
        native=Libdl.dlopen(Colmat.backend_library())
        @test BLAS.lbt_get_forward(name, $J===Int64 ? :ilp64 : :lp64)==Libdl.dlsym(
            native, symbol
        )
        pointer=Libdl.dlsym(
            route==:dll ? native : Libdl.dlopen(BLAS.libblastrampoline), symbol
        )
        lapacke288_selected_entry(name, $J, route) do
            if high
                ccall(
                    pointer,
                    $J,
                    (Cint, $J, $J, $J, Ptr{$T}, $J, Ptr{$T}, Ptr{$T}, $J, Ptr{$T}),
                    layout,
                    f,
                    s,
                    p,
                    a,
                    la,
                    ta,
                    b,
                    lb,
                    tb,
                )
            else
                ccall(
                    pointer,
                    $J,
                    (
                        Cint,
                        $J,
                        $J,
                        $J,
                        Ptr{$T},
                        $J,
                        Ptr{$T},
                        Ptr{$T},
                        $J,
                        Ptr{$T},
                        Ptr{$T},
                        $J,
                    ),
                    layout,
                    f,
                    s,
                    p,
                    a,
                    la,
                    ta,
                    b,
                    lb,
                    tb,
                    w,
                    lw,
                )
            end
        end
    end
end
for (T, prefix) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    rq in (false, true)

    symbol=Symbol(prefix*(T<:Real ? "org" : "ung")*(rq ? "rq" : "qr")*"_64_")
    @eval function lapacke288_gpair_q(::Val{$rq}, a::Matrix{$T}, tau::Vector{$T}, library)
        rows, cols=size(a)
        order=$rq ? cols : rows
        k=min(rows, cols)
        q=fill($T(NaN), max(1, order), order)
        if $rq
            q[(order - k + 1):order, :]=a[(rows - k + 1):rows, :]
        else
            q[1:rows, 1:k]=a[:, 1:k]
        end
        w=zeros($T, max(1, order))
        info=Ref{Int64}(-777)
        ccall(
            Libdl.dlsym(Libdl.dlopen(library), $(QuoteNode(symbol))),
            Cvoid,
            (
                Ref{Int64},
                Ref{Int64},
                Ref{Int64},
                Ptr{$T},
                Ref{Int64},
                Ptr{$T},
                Ptr{$T},
                Ref{Int64},
                Ref{Int64},
            ),
            order,
            order,
            k,
            q,
            max(1, order),
            tau,
            w,
            length(w),
            info,
        )
        @test info[]==0
        q[1:order, :]
    end
end
function lapacke288_gpair_store(a, layout, ld)
    rows, cols=size(a)
    out=fill(eltype(a)(777), max(1, (layout==101 ? rows : cols)*ld)+2)
    for j in 1:cols, i in 1:rows
        out[1 + (layout == 101 ? (i - 1) * ld + j : (j - 1) * ld + i)]=a[i, j]
    end
    return out
end
function lapacke288_gpair_load(a, layout, rows, cols, ld)
    return [
        a[1 + (layout == 101 ? (i - 1) * ld + j : (j - 1) * ld + i)] for
        i in 1:rows, j in 1:cols
    ]
end
function lapacke288_gpair_triangle(a, rq)
    rows, cols=size(a)
    return [
        ((rq ? i<=j+rows-cols : i<=j) ? a[i, j] : zero(eltype(a))) for
        i in 1:rows, j in 1:cols
    ]
end
@testset "Generalized QR/RQ C factors consumed by independent Q generators" begin
    rng=MersenneTwister(288452)
    reference=only(
        filter(
            l->l.interface==:ilp64&&occursin("openblas", lowercase(l.libname)),
            BLAS.get_config().loaded_libs,
        ),
    ).libname
    diagnostic=Libdl.dlsym(Libdl.dlopen(Colmat.backend_library()), :colmat_blas_last_error)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        rq in (false, true),
        (f, s, p) in (
            (0, 0, 0),
            (0, 3, 4),
            (3, 0, 4),
            (3, 4, 0),
            (1, 1, 1),
            (2, 4, 5),
            (5, 2, 4),
            (4, 5, 2),
            (6, 3, 7),
            (3, 6, 2),
        ),
        mode in (:dense, :zero, :rankone),
        exponent in (-1, 0, 1),
        layout in (101, 102),
        route in (:dll, :lbt_trampoline),
        high in (false, true)

        R=real(T)
        scale=R(2)^(exponent*(R===Float32 ? 40 : 400))
        ar, ac, br, bc=rq ? (f, p, s, p) : (f, s, f, p)
        matrix(rows, cols) = scale*(
            if mode==:zero
                zeros(T, rows, cols)
            elseif mode==:rankone
                randn(rng, T, rows)*randn(rng, T, 1, cols)
            else
                randn(rng, T, rows, cols)
            end
        )
        ao=matrix(ar, ac)
        bo=matrix(br, bc)
        la=(layout==101 ? ac : ar)+3
        lb=(layout==101 ? bc : br)+4
        aa=lapacke288_gpair_store(ao, layout, la)
        bb=lapacke288_gpair_store(bo, layout, lb)
        sa=copy(aa)
        sb=copy(bb)
        ta=fill(T(777), min(ar, ac)+2)
        tb=fill(T(777), min(br, bc)+2)
        w=fill(T(779), max(1, f, s, p)+2)
        av=view(aa, 2:(length(aa) - 1))
        bv=view(bb, 2:(length(bb) - 1))
        tav=view(ta, 2:(length(ta) - 1))
        tbv=view(tb, 2:(length(tb) - 1))
        wv=view(w, 2:(length(w) - 1))
        call(high, lw) = lapacke288_gpair(
            T, J, rq, high, layout, f, s, p, av, la, tav, bv, lb, tbv, wv, lw; route
        )
        if !high
            @test call(false, -1)==0
            @test real(w[2])>=max(1, f, s, p)
            query=w[2]
            saved=deepcopy((aa, bb, ta, tb))
            w[2]=T(779)
            @test call(false, 0)==-12
            @test w[2]==query && isequal((aa, bb, ta, tb), saved)
            @test ccall(diagnostic, Cint, ())==11
        end
        @test call(high, length(w)-2)==0
        a=lapacke288_gpair_load(aa, layout, ar, ac, la)
        b=lapacke288_gpair_load(bb, layout, br, bc, lb)
        qa=lapacke288_gpair_q(Val(rq), a, ta[2:(end - 1)], reference)
        qb=lapacke288_gpair_q(Val(!rq), b, tb[2:(end - 1)], reference)
        ra=lapacke288_gpair_triangle(a, rq)
        rb=lapacke288_gpair_triangle(b, !rq)
        tol=300eps(R)*max(1, f, s, p)
        @test qa'*qa≈Matrix{T}(I, size(qa, 1), size(qa, 1)) atol=tol rtol=tol
        @test qb'*qb≈Matrix{T}(I, size(qb, 1), size(qb, 1)) atol=tol rtol=tol
        @test (rq ? ra*qa : qa*ra)≈ao atol=tol*max(norm(ao), scale) rtol=tol
        @test (rq ? qb*rb*qa : qa*rb*qb)≈bo atol=tol*max(norm(bo), scale) rtol=tol
        for (x, old, rows, cols, ld) in ((aa, sa, ar, ac, la), (bb, sb, br, bc, lb))
            touched=Set(
                1+(layout==101 ? (i-1)*ld+j : (j-1)*ld+i) for j in 1:cols for i in 1:rows
            )
            @test all(isequal(x[i], old[i]) for i in eachindex(x) if !(i in touched))
        end
        @test ta[[1, end]]==tb[[1, end]]==T[777, 777]
        @test w[[1, end]]==T[779, 779]
    end
end
