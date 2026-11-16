using Test, LinearAlgebra, Colmat

for (T, prefix) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    J in (Int32, Int64)

    coefficient=T<:Complex ? :(Ref{$T}) : T
    alpha=T<:Complex ? :(Ref(alpha)) : :alpha
    beta=T<:Complex ? :(Ref(beta)) : :beta
    @eval function cblas271_gemm(
        op,
        order,
        ta,
        tb,
        m,
        n,
        k,
        alpha::$T,
        a,
        la,
        b,
        lb,
        beta::$T,
        c::Vector{$T},
        lc,
        ::Type{$J},
        route;
        u=121,
    )
        name="cblas_"*$prefix*op
        if op=="gemmt"||op=="gemmtr"
            @assert m==n
            @cblas271_call name $J route Cvoid (
                Cint,
                Cint,
                Cint,
                Cint,
                $J,
                $J,
                $coefficient,
                Ptr{$T},
                $J,
                Ptr{$T},
                $J,
                $coefficient,
                Ptr{$T},
                $J,
            ) order u ta tb $J(n) $J(k) $alpha a $J(la) b $J(lb) $beta c $J(lc)
        else
            @cblas271_call name $J route Cvoid (
                Cint,
                Cint,
                Cint,
                $J,
                $J,
                $J,
                $coefficient,
                Ptr{$T},
                $J,
                Ptr{$T},
                $J,
                $coefficient,
                Ptr{$T},
                $J,
            ) order ta tb $J(m) $J(n) $J(k) $alpha a $J(la) b $J(lb) $beta c $J(lc)
        end
        return nothing
    end
end

@testset "CBLAS GEMM families independent scalar products" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        route in (:dll, :lbt_trampoline)

        tolerance=real(T)===Float32 ? 4e-5 : 1e-11
        operations=if T<:Complex
            ("gemm", "gemm3m", "gemmt", "gemmtr")
        else
            ("gemm", "gemmt", "gemmtr")
        end
        nonzero_alpha=cblas271_value(T, 1, 2, 1)
        nonzero_beta=cblas271_value(T, 2, 1, -5)
        for op in operations,
            order in (101, 102),
            ta in 111:114,
            tb in 111:114,
            k in (0, 4),
            zero_alpha in (false, true),
            zero_beta in (false, true)

            triangular=op=="gemmt"||op=="gemmtr"
            m, n=triangular ? (3, 3) : (2, 3)
            ar, ac=ta in (111, 114) ? (m, k) : (k, m)
            br, bc=tb in (111, 114) ? (k, n) : (n, k)
            logical_a=[cblas271_value(T, i, j, 1) for i in 1:ar, j in 1:ac]
            logical_b=[cblas271_value(T, i, j, 3) for i in 1:br, j in 1:bc]
            la=max(1, order==101 ? ac : ar)+2
            lb=max(1, order==101 ? bc : br)+3
            lc=(order==101 ? n : m)+2
            a=cblas271_pack(logical_a, order, la)
            b=cblas271_pack(logical_b, order, lb)
            alpha=zero_alpha ? zero(T) : nonzero_alpha
            beta=zero_beta ? zero(T) : nonzero_beta
            ap=zero_alpha||k==0 ? Ptr{T}(C_NULL) : a
            bp=zero_alpha||k==0 ? Ptr{T}(C_NULL) : b
            for u in (triangular ? (121, 122) : (121,))
                initial=fill(T(97), lc*(order==101 ? m : n)+3)
                expected=copy(initial)
                written=falses(length(initial))
                oa=cblas271_op(logical_a, ta)
                ob=cblas271_op(logical_b, tb)
                for j in 1:n, i in 1:m
                    if !triangular||(u==121&&i<=j)||(u==122&&i>=j)
                        position=cblas271_matrix_index(order, lc, i, j)
                        written[position]=true
                        old=cblas271_value(T, i, j, 4)
                        initial[position]=zero_beta ? T(NaN) : old
                        product=zero(T)
                        if !zero_alpha
                            for l in 1:k
                                product+=oa[i, l]*ob[l, j]
                            end
                        end
                        expected[position]=alpha*product+(zero_beta ? zero(T) : beta*old)
                    end
                end
                actual=copy(initial)
                cblas271_gemm(
                    op,
                    order,
                    ta,
                    tb,
                    m,
                    n,
                    k,
                    alpha,
                    ap,
                    la,
                    bp,
                    lb,
                    beta,
                    actual,
                    lc,
                    J,
                    route;
                    u,
                )
                @test all(eachindex(actual)) do index
                    if written[index]
                        isapprox(
                        actual[index], expected[index]; atol=tolerance, rtol=tolerance
                    )
                    else
                        actual[index]==expected[index]
                    end
                end
            end
        end
    end
end

const CBLAS_271_GEMM_TARGETS=split(raw"""
cblas_sgemm cblas_dgemm cblas_cgemm cblas_zgemm cblas_cgemm3m cblas_zgemm3m
cblas_sgemmt cblas_dgemmt cblas_cgemmt cblas_zgemmt
cblas_sgemmtr cblas_dgemmtr cblas_cgemmtr cblas_zgemmtr
""")
@testset "Every GEMM target completed through every ABI and route" begin
    @test length(CBLAS_271_GEMM_TARGETS)==14
    @test Set(CBLAS_271_GEMM_TARGETS)⊆Set(CBLAS_271_TARGETS)
    for bits in (32, 64), route in (:dll, :lbt_trampoline)
        observed=Set(
            name for (name, width, actual_route) in CBLAS_271_CALLS if
            width==bits&&actual_route==route&&name in CBLAS_271_GEMM_TARGETS
        )
        @test observed==Set(CBLAS_271_GEMM_TARGETS)
    end
end
