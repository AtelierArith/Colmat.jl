using Test, Colmat, LinearAlgebra, Libdl, Random
isdefined(@__MODULE__, :lapacke288_selected_entry) || include("lapacke_calls.jl")
isdefined(@__MODULE__, :direct_tp_factor!) || include("pentagonal.jl")
for (T, prefix) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    J in (Int32, Int64)

    for blocked in (false, true), high in (false, true)
        name="LAPACKE_"*prefix*(blocked ? "tpqrt" : "tpqrt2")*(high ? "" : "_work")*(
            J===Int64 ? "64_" : ""
        )
        sig=Any[Cint, J, J, J]
        args=Any[:layout, :m, :n, :l]
        blocked && (push!(sig, J); push!(args, :nb))
        append!(sig, [Ptr{T}, J, Ptr{T}, J, Ptr{T}, J])
        append!(args, [:a, :lda, :b, :ldb, :t, :ldt])
        blocked && !high && (push!(sig, Ptr{T}); push!(args, :work))
        canonical=replace(name, r"64_$"=>"")
        @eval function lapacke288_tp_factor_call(
            ::Type{$T},
            ::Type{$J},
            ::Val{$blocked},
            ::Val{$high},
            layout,
            m,
            n,
            l,
            nb,
            a,
            lda,
            b,
            ldb,
            t,
            ldt,
            work;
            route=:dll,
        )
            native=Libdl.dlopen(Colmat.backend_library())
            @test BLAS.lbt_get_forward($canonical, $(J===Int64 ? :(:ilp64) : :(:lp64)))==Libdl.dlsym(
                native, $(QuoteNode(Symbol(name)))
            )
            handle=route==:dll ? native : Libdl.dlopen(BLAS.libblastrampoline)
            pointer=Libdl.dlsym(handle, $(QuoteNode(Symbol(name))))
            lapacke288_selected_entry($canonical, $J, route) do
                ccall(pointer, $J, $(Expr(:tuple, sig...)), $(args...))
            end
        end
    end
end
function lapacke288_tp_store(x, layout, ld)
    rows, cols=size(x)
    v=fill(eltype(x)(777), max(1, (layout==101 ? rows : cols)*ld)+2)
    for j in 1:cols, i in 1:rows
        v[1 + (layout == 101 ? (i - 1) * ld + j : (j - 1) * ld + i)]=x[i, j]
    end
    return v
end
function lapacke288_tp_load(v, layout, rows, cols, ld)
    return [
        v[1 + (layout == 101 ? (i - 1) * ld + j : (j - 1) * ld + i)] for
        i in 1:rows, j in 1:cols
    ]
end
@testset "Triangular-pentagonal C factors and independent block consumers" begin
    rng=MersenneTwister(288388)
    reference=pentagonal_reference()
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        (m, n) in ((0, 0), (0, 3), (3, 0), (1, 1), (3, 5), (5, 3)),
        l in unique((0, min(m, n))),
        blocked in (false, true),
        nb in (blocked ? unique((1, max(1, min(n, 2)), max(1, n))) : (max(1, n),)),
        layout in (101, 102),
        high in (false, true),
        route in (:dll, :lbt_trampoline)

        rows=blocked ? nb : max(1, n)
        a0=triu(randn(rng, T, n, n))
        b0=randn(rng, T, m, n)
        for j in 1:n, i in 1:m
            if i>m-l && i-(m-l)>j
                b0[i, j]=0
            end
        end
        ar=copy(a0)
        br=copy(b0)
        result=direct_tp_factor!(
            ar,
            br,
            Int64,
            Val(false),
            Val(blocked);
            m,
            n,
            l,
            nb,
            lda=max(1, n),
            ldb=max(1, m),
            library=reference,
        )
        @test result.info==0
        lda=(layout==101 ? n : n)+3
        ldb=(layout==101 ? n : m)+4
        ldt=(layout==101 ? n : rows)+5
        aa=lapacke288_tp_store(a0, layout, lda)
        bb=lapacke288_tp_store(b0, layout, ldb)
        tt=lapacke288_tp_store(fill(T(777), rows, n), layout, ldt)
        work=fill(T(777), max(1, nb*n)+2)
        originala=copy(aa)
        originalb=copy(bb)
        originalt=copy(tt)
        av=view(aa, 2:(length(aa) - 1))
        bv=view(bb, 2:(length(bb) - 1))
        tv=view(tt, 2:(length(tt) - 1))
        wv=view(work, 2:(length(work) - 1))
        info=lapacke288_tp_factor_call(
            T,
            J,
            Val(blocked),
            Val(high),
            layout,
            m,
            n,
            l,
            nb,
            av,
            lda,
            bv,
            ldb,
            tv,
            ldt,
            wv;
            route,
        )
        @test info==0
        a=lapacke288_tp_load(aa, layout, n, n, lda)
        b=lapacke288_tp_load(bb, layout, m, n, ldb)
        t=lapacke288_tp_load(tt, layout, rows, n, ldt)
        tol=200*eps(real(T))
        @test a≈ar rtol=tol atol=tol
        @test b≈br rtol=tol atol=tol
        for j in 1:n, i in 1:(1 + (j - 1) % rows)
            @test t[i, j]≈result.t[i, j] rtol=tol atol=tol
        end
        for (v, old, r, c, ld, kind) in (
            (aa, originala, n, n, lda, :a),
            (bb, originalb, m, n, ldb, :b),
            (tt, originalt, rows, n, ldt, :t),
        )
            touched=Set(
                1+(layout==101 ? (i-1)*ld+j : (j-1)*ld+i) for j in 1:c for i in 1:r if (
                    if kind==:a
                        i<=j
                    elseif kind==:b
                        i<=m-l || i-(m-l)<=j
                    else
                        i<=1+(j-1)%rows
                    end
                )
            )
            @test all(isequal(v[i], old[i]) for i in eachindex(v) if !(i in touched))
        end
        @test work[[1, end]]==T[777, 777]
        if m>0 && n>0
            ca=Matrix{T}(I, n, n)
            cb=zeros(T, m, n)
            q=direct_tp_apply!(
                b,
                t,
                ca,
                cb,
                Int64,
                Val(false);
                m,
                n,
                k=n,
                l,
                nb=(blocked ? nb : n),
                library=reference,
            )
            @test q.info==0
            @test vcat(ca, cb)*triu(a)≈vcat(a0, b0) rtol=10tol atol=10tol
            @test vcat(ca, cb)'*vcat(ca, cb)≈Matrix{T}(I, n, n) rtol=tol atol=tol
        end
    end
end
