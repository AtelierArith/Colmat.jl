using Test, LinearAlgebra, Colmat

for (T, prefix) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    J in (Int32, Int64)

    coefficient=T<:Complex ? :(Ref{$T}) : T
    alpha=T<:Complex ? :(Ref(alpha)) : :alpha
    beta=T<:Complex ? :(Ref(beta)) : :beta
    dense=T<:Complex ? "hemv" : "symv"
    band=T<:Complex ? "hbmv" : "sbmv"
    packed=T<:Complex ? "hpmv" : "spmv"
    @eval function cblas271_symmetric_mv(
        storage,
        order,
        u,
        n,
        k,
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
        name="cblas_"*$prefix*(
            if storage===:dense
                $dense
            elseif storage===:band
                $band
            else
                $packed
            end
        )
        if storage===:dense
            @cblas271_call name $J route Cvoid (
                Cint,
                Cint,
                $J,
                $coefficient,
                Ptr{$T},
                $J,
                Ptr{$T},
                $J,
                $coefficient,
                Ptr{$T},
                $J,
            ) order u $J(n) $alpha a $J(ld) x $J(ix) $beta y $J(iy)
        elseif storage===:band
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
            ) order u $J(n) $J(k) $alpha a $J(ld) x $J(ix) $beta y $J(iy)
        else
            @cblas271_call name $J route Cvoid (
                Cint,
                Cint,
                $J,
                $coefficient,
                Ptr{$T},
                Ptr{$T},
                $J,
                $coefficient,
                Ptr{$T},
                $J,
            ) order u $J(n) $alpha a x $J(ix) $beta y $J(iy)
        end
        return nothing
    end
    @eval function cblas271_symmetric_mm(
        hermitian,
        order,
        side,
        u,
        m,
        n,
        alpha::$T,
        a,
        la,
        b,
        lb,
        beta::$T,
        c::Vector{$T},
        lc,
        ::Type{$J},
        route,
    )
        name="cblas_"*$prefix*(hermitian ? "hemm" : "symm")
        @cblas271_call name $J route Cvoid (
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
        ) order side u $J(m) $J(n) $alpha a $J(la) b $J(lb) $beta c $J(lc)
        return nothing
    end
end

@testset "CBLAS symmetric/Hermitian dense band and packed matvec" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        route in (:dll, :lbt_trampoline)

        hermitian=T<:Complex
        tolerance=real(T)===Float32 ? 4e-5 : 1e-11
        for n in (1, 4),
            storage in (:dense, :band, :packed), order in (101, 102),
            u in (121, 122)

            widths=storage===:band ? (0, 1, n+1) : (n+1,)
            for k in widths
                matrix=cblas271_symmetric_matrix(T, n, u, k, hermitian)
                a, ld=cblas271_triangle_storage(matrix, order, u, storage, k; hermitian)
                logical_x=T[cblas271_value(T, i, 1, 2) for i in 1:n]
                old_y=T[cblas271_value(T, i, 2, 1) for i in 1:n]
                for ix in (-2, 1, 2),
                    iy in (-3, 1, 3), zero_alpha in (false, true),
                    zero_beta in (false, true)

                    alpha=zero_alpha ? zero(T) : cblas271_value(T, 1, 2, 1)
                    beta=zero_beta ? zero(T) : cblas271_value(T, 2, 1, -5)
                    x=cblas271_storage(logical_x, ix)
                    initial=cblas271_storage(zero_beta ? fill(T(NaN), n) : old_y, iy)
                    indices=cblas271_indices(n, iy)
                    expected=copy(initial)
                    for i in 1:n
                        product=zero(T)
                        if !zero_alpha
                            for j in 1:n
                                product+=matrix[i, j]*logical_x[j]
                            end
                        end
                        expected[indices[i]]=alpha*product+(
                            zero_beta ? zero(T) : beta*old_y[i]
                        )
                    end
                    actual=copy(initial)
                    cblas271_symmetric_mv(
                        storage,
                        order,
                        u,
                        n,
                        k,
                        alpha,
                        zero_alpha ? Ptr{T}(C_NULL) : a,
                        ld,
                        zero_alpha ? Ptr{T}(C_NULL) : x,
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
                                actual[index],
                                expected[index];
                                atol=tolerance,
                                rtol=tolerance,
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

@testset "CBLAS SYMM/HEMM independent scalar matrix products" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        route in (:dll, :lbt_trampoline)

        tolerance=real(T)===Float32 ? 4e-5 : 1e-11
        for hermitian in (T<:Complex ? (false, true) : (false,)),
            (m, n) in ((2, 3), (3, 2)), order in (101, 102), side in (141, 142),
            u in (121, 122)

            dim=side==141 ? m : n
            matrix=cblas271_symmetric_matrix(T, dim, u, dim, hermitian)
            a, la=cblas271_triangle_storage(matrix, order, u, :dense, dim; hermitian)
            logical_b=T[cblas271_value(T, i, j, 3) for i in 1:m, j in 1:n]
            lb=(order==101 ? n : m)+2
            lc=lb+1
            b=cblas271_pack(logical_b, order, lb)
            for zero_alpha in (false, true), zero_beta in (false, true)
                alpha=zero_alpha ? zero(T) : cblas271_value(T, 1, 2, 1)
                beta=zero_beta ? zero(T) : cblas271_value(T, 2, 1, -5)
                initial=fill(T(97), lc*(order==101 ? m : n)+3)
                expected=copy(initial)
                written=falses(length(initial))
                for i in 1:m, j in 1:n
                    position=cblas271_matrix_index(order, lc, i, j)
                    written[position]=true
                    old=cblas271_value(T, i, j, 4)
                    initial[position]=zero_beta ? T(NaN) : old
                    product=zero(T)
                    if !zero_alpha
                        for l in 1:dim
                            product+=if side==141
                                matrix[i, l]*logical_b[l, j]
                            else
                                logical_b[i, l]*matrix[l, j]
                            end
                        end
                    end
                    expected[position]=alpha*product+(zero_beta ? zero(T) : beta*old)
                end
                actual=copy(initial)
                cblas271_symmetric_mm(
                    hermitian,
                    order,
                    side,
                    u,
                    m,
                    n,
                    alpha,
                    zero_alpha ? Ptr{T}(C_NULL) : a,
                    la,
                    zero_alpha ? Ptr{T}(C_NULL) : b,
                    lb,
                    beta,
                    actual,
                    lc,
                    J,
                    route,
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

const CBLAS_271_SYMMETRIC_TARGETS=split(raw"""
cblas_ssymv cblas_dsymv cblas_chemv cblas_zhemv
cblas_ssbmv cblas_dsbmv cblas_chbmv cblas_zhbmv
cblas_sspmv cblas_dspmv cblas_chpmv cblas_zhpmv
cblas_ssymm cblas_dsymm cblas_csymm cblas_zsymm cblas_chemm cblas_zhemm
""")
@testset "Every symmetric product target completed through every ABI and route" begin
    @test length(CBLAS_271_SYMMETRIC_TARGETS)==18
    @test Set(CBLAS_271_SYMMETRIC_TARGETS)⊆Set(CBLAS_271_TARGETS)
    for bits in (32, 64), route in (:dll, :lbt_trampoline)
        observed=Set(
            name for (name, width, actual_route) in CBLAS_271_CALLS if
            width==bits&&actual_route==route&&name in CBLAS_271_SYMMETRIC_TARGETS
        )
        @test observed==Set(CBLAS_271_SYMMETRIC_TARGETS)
    end
end
