using Test, Colmat, LinearAlgebra, Libdl, Random
isdefined(@__MODULE__, :lapacke288_tp_store) || include("lapacke_pentagonal_qr.jl")
for (T, prefix) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    J in (Int32, Int64), family in (:tpmqrt, :tprfb),
    high in (false, true)

    block=family==:tprfb
    name="LAPACKE_"*prefix*String(family)*(high ? "" : "_work")*(J===Int64 ? "64_" : "")
    sig=Any[Cint, Int8, Int8]
    args=Any[:layout, :side, :trans]
    block && (append!(sig, [Int8, Int8]); append!(args, [:direct, :store]))
    append!(sig, [J, J, J, J])
    append!(args, [:m, :n, :k, :l])
    !block && (push!(sig, J); push!(args, :nb))
    append!(sig, [Ptr{T}, J, Ptr{T}, J, Ptr{T}, J, Ptr{T}, J])
    append!(args, [:v, :ldv, :t, :ldt, :a, :lda, :b, :ldb])
    !high && (push!(sig, Ptr{T}); push!(args, :work))
    block && !high && (push!(sig, J); push!(args, :ldwork))
    canonical=replace(name, r"64_$"=>"")
    @eval function lapacke288_tp_application_call(
        ::Type{$T},
        ::Type{$J},
        ::Val{$(QuoteNode(family))},
        ::Val{$high},
        layout,
        side,
        trans,
        direct,
        store,
        m,
        n,
        k,
        l,
        nb,
        v,
        ldv,
        t,
        ldt,
        a,
        lda,
        b,
        ldb,
        work,
        ldwork;
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
function lapacke288_tp_application_storage(T, layout, v, t, a, b)
    vr, vc=size(v)
    tr, tc=size(t)
    ar, ac=size(a)
    br, bc=size(b)
    ldv=(layout==101 ? vc : vr)+3
    ldt=(layout==101 ? tc : tr)+4
    lda=(layout==101 ? ac : ar)+5
    ldb=(layout==101 ? bc : br)+6
    vv=lapacke288_tp_store(v, layout, ldv)
    tt=lapacke288_tp_store(t, layout, ldt)
    aa=lapacke288_tp_store(a, layout, lda)
    bb=lapacke288_tp_store(b, layout, ldb)
    return (; vv, tt, aa, bb, ldv, ldt, lda, ldb)
end
function lapacke288_tp_application_guards(data, original, layout, ar, ac, br, bc)
    @test isequal(data.vv, original.vv)
    @test isequal(data.tt, original.tt)
    for (v, old, r, c, ld) in
        ((data.aa, original.aa, ar, ac, data.lda), (data.bb, original.bb, br, bc, data.ldb))
        touched=Set(1+(layout==101 ? (i-1)*ld+j : (j-1)*ld+i) for j in 1:c for i in 1:r)
        @test all(isequal(v[i], old[i]) for i in eachindex(v) if !(i in touched))
    end
end
@testset "TPRFB C layouts against elementary reflector products" begin
    rng=MersenneTwister(288389)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        k in (1, 3),
        tail in (1, 4),
        l in unique((0, min(k, tail))),
        direct in ('F', 'B'),
        store in ('C', 'R'),
        side in ('L', 'R'),
        trans in ('N', T<:Complex ? 'C' : 'T'),
        layout in (101, 102),
        high in (false, true),
        route in (:dll, :lbt_trampoline)

        rawv, rawt, h=tp_block_fixture(T, k, tail, l, direct, store, rng)
        vr, vc=store=='C' ? (tail, k) : (k, tail)
        v=copy(rawv[1:vr, 1:vc])
        t=copy(rawt[1:k, 1:k])
        if high
            replace!(v, T(NaN)=>zero(T))
            replace!(t, T(NaN)=>zero(T))
        end
        rows, cols=side=='L' ? (k+tail, 5) : (5, k+tail)
        c=randn(rng, T, rows, cols)
        op=trans=='N' ? h : h'
        expected=side=='L' ? op*c : c*op
        head=direct=='F' ? (1:k) : ((tail + 1):(k + tail))
        rest=direct=='F' ? ((k + 1):(k + tail)) : (1:tail)
        a=side=='L' ? c[head, :] : c[:, head]
        b=side=='L' ? c[rest, :] : c[:, rest]
        m, n=size(b)
        ar, ac=size(a)
        br, bc=size(b)
        data=lapacke288_tp_application_storage(T, layout, v, t, a, b)
        original=deepcopy(data)
        ldwork=(side=='L' ? k : m)+2
        work=fill(T(777), ldwork*(side=='L' ? n : k)+2)
        info=lapacke288_tp_application_call(
            T,
            J,
            Val(:tprfb),
            Val(high),
            layout,
            side,
            trans,
            direct,
            store,
            m,
            n,
            k,
            l,
            0,
            view(data.vv, 2:(length(data.vv) - 1)),
            data.ldv,
            view(data.tt, 2:(length(data.tt) - 1)),
            data.ldt,
            view(data.aa, 2:(length(data.aa) - 1)),
            data.lda,
            view(data.bb, 2:(length(data.bb) - 1)),
            data.ldb,
            view(work, 2:(length(work) - 1)),
            ldwork;
            route,
        )
        @test info==0
        actual=similar(c)
        if side=='L'
            actual[head, :]=lapacke288_tp_load(data.aa, layout, ar, ac, data.lda)
            actual[rest, :]=lapacke288_tp_load(data.bb, layout, br, bc, data.ldb)
        else
            actual[:, head]=lapacke288_tp_load(data.aa, layout, ar, ac, data.lda)
            actual[:, rest]=lapacke288_tp_load(data.bb, layout, br, bc, data.ldb)
        end
        @test actual≈expected rtol=200eps(real(T)) atol=200eps(real(T))
        lapacke288_tp_application_guards(data, original, layout, ar, ac, br, bc)
        @test work[[1, end]]==T[777, 777]
        @test pentagonal_diagnostic()==0
    end
end
@testset "TPMQRT C application consumes independent pentagonal factors" begin
    rng=MersenneTwister(288390)
    reference=pentagonal_reference()
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        k in (1, 3),
        tail in (1, 4),
        l in unique((0, min(k, tail))),
        nb in unique((1, min(2, k), k)),
        side in ('L', 'R'),
        trans in ('N', T<:Complex ? 'C' : 'T'),
        layout in (101, 102),
        high in (false, true),
        route in (:dll, :lbt_trampoline)

        a, b, _, _=tp_fixture(T, k, tail, l, false, rng)
        factor=direct_tp_factor!(
            a, b, Int64, Val(false), Val(true); m=tail, n=k, l, nb, library=reference
        )
        @test factor.info==0
        q=tp_full_q(b, factor.t, Int64, false, k, tail, l, nb; library=reference)
        v=copy(b[1:tail, :])
        t=copy(factor.t[1:nb, :])
        if high
            replace!(v, T(NaN)=>zero(T))
            replace!(t, T(NaN)=>zero(T))
        end
        rows, cols=side=='L' ? (k+tail, 5) : (5, k+tail)
        c=randn(rng, T, rows, cols)
        op=trans=='N' ? q : q'
        expected=side=='L' ? op*c : c*op
        a=side=='L' ? c[1:k, :] : c[:, 1:k]
        b=side=='L' ? c[(k + 1):end, :] : c[:, (k + 1):end]
        m, n=size(b)
        ar, ac=size(a)
        br, bc=size(b)
        data=lapacke288_tp_application_storage(T, layout, v, t, a, b)
        original=deepcopy(data)
        work=fill(T(777), max(m, n)*nb+2)
        info=lapacke288_tp_application_call(
            T,
            J,
            Val(:tpmqrt),
            Val(high),
            layout,
            side,
            trans,
            'F',
            'C',
            m,
            n,
            k,
            l,
            nb,
            view(data.vv, 2:(length(data.vv) - 1)),
            data.ldv,
            view(data.tt, 2:(length(data.tt) - 1)),
            data.ldt,
            view(data.aa, 2:(length(data.aa) - 1)),
            data.lda,
            view(data.bb, 2:(length(data.bb) - 1)),
            data.ldb,
            view(work, 2:(length(work) - 1)),
            0;
            route,
        )
        @test info==0
        actual=if side=='L'
            vcat(
                lapacke288_tp_load(data.aa, layout, ar, ac, data.lda),
                lapacke288_tp_load(data.bb, layout, br, bc, data.ldb),
            )
        else
            hcat(
                lapacke288_tp_load(data.aa, layout, ar, ac, data.lda),
                lapacke288_tp_load(data.bb, layout, br, bc, data.ldb),
            )
        end
        @test actual≈expected rtol=200eps(real(T)) atol=200eps(real(T))
        lapacke288_tp_application_guards(data, original, layout, ar, ac, br, bc)
        @test work[[1, end]]==T[777, 777]
        @test pentagonal_diagnostic()==0
    end
end
