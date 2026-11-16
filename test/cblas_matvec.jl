using Test, LinearAlgebra, Colmat

for (T, prefix) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    J in (Int32, Int64)

    coefficient=T<:Complex ? :(Ref{$T}) : T
    alpha=T<:Complex ? :(Ref(alpha)) : :alpha
    beta=T<:Complex ? :(Ref(beta)) : :beta
    @eval function cblas271_matvec(
        op,
        order,
        t,
        m,
        n,
        kl,
        ku,
        alpha::$T,
        a,
        ld,
        x,
        ix,
        beta::$T,
        y::Vector{$T},
        iy,
        ::Type{$J},
        route,
    )
        name="cblas_"*$prefix*op
        if op=="gbmv"
            @cblas271_call name $J route Cvoid (
                Cint,
                Cint,
                $J,
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
            ) order t $J(m) $J(n) $J(kl) $J(ku) $alpha a $J(ld) x $J(ix) $beta y $J(iy)
        else
            @cblas271_call name $J route Cvoid (
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
            ) order t $J(m) $J(n) $alpha a $J(ld) x $J(ix) $beta y $J(iy)
        end
        return nothing
    end
end

@testset "CBLAS dense and general-band matrix-vector products" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        route in (:dll, :lbt_trampoline)

        tolerance=real(T)===Float32 ? 4e-5 : 1e-11
        for op in ("gemv", "gbmv"),
            order in (101, 102), t in 111:114,
            (m, n) in ((2, 5), (5, 2))

            widths=op=="gbmv" ? ((0, 0), (1, 2), (m+1, n+1)) : ((0, 0),)
            for (kl, ku) in widths
                logical=[
                    op=="gemv"||(i<=j+kl&&j<=i+ku) ? cblas271_value(T, i, j, 1) : zero(T)
                    for i in 1:m, j in 1:n
                ]
                if op=="gemv"
                    ld=(order==101 ? n : m)+2
                    a=cblas271_pack(logical, order, ld)
                else
                    ld=kl+ku+3
                    a=fill(T(NaN), ld*(order==101 ? m : n)+3)
                    for i in 1:m, j in 1:n
                        if i<=j+kl&&j<=i+ku
                            position=order==101 ? (i-1)*ld+kl+j-i+1 : (j-1)*ld+ku+i-j+1
                            a[position]=logical[i, j]
                        end
                    end
                end
                operated=cblas271_op(logical, t)
                ny, nx=size(operated)
                logical_x=T[cblas271_value(T, i, 1, 2) for i in 1:nx]
                old_y=T[cblas271_value(T, i, 2, 1) for i in 1:ny]
                for ix in (-2, 1, 2),
                    iy in (-3, 1, 3), zero_alpha in (false, true),
                    zero_beta in (false, true)

                    alpha=zero_alpha ? zero(T) : cblas271_value(T, 1, 2, 1)
                    beta=zero_beta ? zero(T) : cblas271_value(T, 2, 1, -5)
                    x=cblas271_storage(logical_x, ix)
                    initial=cblas271_storage(zero_beta ? fill(T(NaN), ny) : old_y, iy)
                    indices=cblas271_indices(ny, iy)
                    expected=copy(initial)
                    for i in 1:ny
                        product=zero(T)
                        if !zero_alpha
                            for j in 1:nx
                                product+=operated[i, j]*logical_x[j]
                            end
                        end
                        expected[indices[i]]=alpha*product+(
                            zero_beta ? zero(T) : beta*old_y[i]
                        )
                    end
                    ap=zero_alpha ? Ptr{T}(C_NULL) : a
                    xp=zero_alpha ? Ptr{T}(C_NULL) : x
                    actual=copy(initial)
                    cblas271_matvec(
                        op,
                        order,
                        t,
                        m,
                        n,
                        kl,
                        ku,
                        alpha,
                        ap,
                        ld,
                        xp,
                        ix,
                        beta,
                        actual,
                        iy,
                        J,
                        route,
                    )
                    written=falses(length(actual))
                    written[indices].=true
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
end

const CBLAS_271_MATVEC_TARGETS=split(raw"""
cblas_sgemv cblas_dgemv cblas_cgemv cblas_zgemv
cblas_sgbmv cblas_dgbmv cblas_cgbmv cblas_zgbmv
""")
@testset "Every GEMV/GBMV target completed through every ABI and route" begin
    @test length(CBLAS_271_MATVEC_TARGETS)==8
    @test Set(CBLAS_271_MATVEC_TARGETS)⊆Set(CBLAS_271_TARGETS)
    for bits in (32, 64), route in (:dll, :lbt_trampoline)
        observed=Set(
            name for (name, width, actual_route) in CBLAS_271_CALLS if
            width==bits&&actual_route==route&&name in CBLAS_271_MATVEC_TARGETS
        )
        @test observed==Set(CBLAS_271_MATVEC_TARGETS)
    end
end
