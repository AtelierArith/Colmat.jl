using Test, Colmat, LinearAlgebra, Random
isdefined(@__MODULE__, :lapacke288_tp_factor_call) || include("lapacke_pentagonal_qr.jl")
@testset "Pentagonal C factors across scales, rank and partial overlap" begin
    rng=MersenneTwister(288391)
    reference=pentagonal_reference()
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        (m, n) in ((3, 5), (5, 3)),
        l in unique((0, 1, min(m, n))),
        nb in unique((1, 2, n)),
        layout in (101, 102),
        route in (:dll, :lbt_trampoline),
        high in (false, true),
        rank in (:full, :zero, :one),
        exponent in (-1, 0, 1)

        power=T<:Union{Float32,ComplexF32} ? 40 : 400
        scale=real(T)(2)^(power*exponent)
        a0=triu(randn(rng, T, n, n))
        b0=randn(rng, T, m, n)
        for j in 1:n, i in 1:m
            if i>m-l && i-(m-l)>j
                b0[i, j]=0
            end
        end
        if rank!=:full
            fill!(b0, 0)
            for j in 1:n, i in 1:n
                if rank==:zero || i!=1
                    a0[i, j]=0
                end
            end
        end
        a0 .*= scale
        b0 .*= scale
        ar=copy(a0)
        br=copy(b0)
        result=direct_tp_factor!(
            ar, br, Int64, Val(false), Val(true); m, n, l, nb, library=reference
        )
        @test result.info==0
        lda=n+3
        ldb=(layout==101 ? n : m)+4
        ldt=(layout==101 ? n : nb)+5
        aa=lapacke288_tp_store(a0, layout, lda)
        bb=lapacke288_tp_store(b0, layout, ldb)
        tt=lapacke288_tp_store(zeros(T, nb, n), layout, ldt)
        work=fill(T(779), nb*n+2)
        @test lapacke288_tp_factor_call(
            T,
            J,
            Val(true),
            Val(high),
            layout,
            m,
            n,
            l,
            nb,
            view(aa, 2:(length(aa) - 1)),
            lda,
            view(bb, 2:(length(bb) - 1)),
            ldb,
            view(tt, 2:(length(tt) - 1)),
            ldt,
            view(work, 2:(length(work) - 1));
            route,
        )==0
        a=lapacke288_tp_load(aa, layout, n, n, lda)
        b=lapacke288_tp_load(bb, layout, m, n, ldb)
        t=lapacke288_tp_load(tt, layout, nb, n, ldt)
        tol=200eps(real(T))
        @test a/scale≈ar/scale rtol=tol atol=tol
        @test b≈br rtol=tol atol=tol
        for j in 1:n, i in 1:(1 + (j - 1) % nb)
            @test t[i, j]≈result.t[i, j] rtol=tol atol=tol
        end
        q=tp_full_q(b, t, Int64, false, n, m, l, nb; library=reference)
        @test q[:, 1:n]*(a/scale)≈vcat(a0/scale, b0/scale) rtol=10tol atol=10tol
        @test q'*q≈Matrix{T}(I, m+n, m+n) rtol=tol atol=tol
        @test work[[1, end]]==T[779, 779]
    end
end
