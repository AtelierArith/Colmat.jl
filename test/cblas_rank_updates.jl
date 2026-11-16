using Test, LinearAlgebra, Colmat

for (T, prefix) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    J in (Int32, Int64)

    coefficient=T<:Complex ? :(Ref{$T}) : T
    alpha=T<:Complex ? :(Ref(alpha)) : :alpha
    R=real(T)
    @eval function cblas271_ger(
        conjugate_y,
        order,
        m,
        n,
        alpha::$T,
        x,
        ix,
        y,
        iy,
        a::Vector{$T},
        la,
        ::Type{$J},
        route,
    )
        name="cblas_"*$prefix*$(T<:Complex ? :(conjugate_y ? "gerc" : "geru") : "ger")
        @cblas271_call name $J route Cvoid (
            Cint, $J, $J, $coefficient, Ptr{$T}, $J, Ptr{$T}, $J, Ptr{$T}, $J
        ) order $J(m) $J(n) $alpha x $J(ix) y $J(iy) a $J(la)
        return nothing
    end
    @eval function cblas271_rank_update(
        two,
        storage,
        order,
        u,
        n,
        alpha::$T,
        x,
        ix,
        y,
        iy,
        a::Vector{$T},
        la,
        ::Type{$J},
        route,
    )
        family=if storage===:packed
            $(T<:Complex ? "hpr" : "spr")
        else
            $(T<:Complex ? "her" : "syr")
        end
        name="cblas_"*$prefix*family*(two ? "2" : "")
        if two
            if storage===:packed
                @cblas271_call name $J route Cvoid (
                    Cint, Cint, $J, $coefficient, Ptr{$T}, $J, Ptr{$T}, $J, Ptr{$T}
                ) order u $J(n) $alpha x $J(ix) y $J(iy) a
            else
                @cblas271_call name $J route Cvoid (
                    Cint, Cint, $J, $coefficient, Ptr{$T}, $J, Ptr{$T}, $J, Ptr{$T}, $J
                ) order u $J(n) $alpha x $J(ix) y $J(iy) a $J(la)
            end
        elseif storage===:packed
            @cblas271_call name $J route Cvoid (Cint, Cint, $J, $R, Ptr{$T}, $J, Ptr{$T}) order u $J(
                n
            ) $R(real(alpha)) x $J(ix) a
        else
            @cblas271_call name $J route Cvoid (
                Cint, Cint, $J, $R, Ptr{$T}, $J, Ptr{$T}, $J
            ) order u $J(n) $R(real(alpha)) x $J(ix) a $J(la)
        end
        return nothing
    end
end

@testset "CBLAS GER/GERU/GERC independent outer products" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        route in (:dll, :lbt_trampoline)

        tolerance=real(T)===Float32 ? 4e-5 : 1e-11
        for conjugate_y in (T<:Complex ? (false, true) : (false,)),
            (m, n) in ((2, 3), (3, 2)),
            order in (101, 102),
            ix in (-2, 1, 2),
            iy in (-3, 1, 3),
            zero_alpha in (false, true)

            alpha=zero_alpha ? zero(T) : cblas271_value(T, 1, 2, 1)
            logical_x=T[cblas271_value(T, i, 1, 2) for i in 1:m]
            logical_y=T[cblas271_value(T, j, 2, 1) for j in 1:n]
            x=cblas271_storage(logical_x, ix)
            y=cblas271_storage(logical_y, iy)
            matrix=T[cblas271_value(T, i, j, 4) for i in 1:m, j in 1:n]
            la=(order==101 ? n : m)+2
            actual=cblas271_pack(matrix, order, la)
            initial=copy(actual)
            expected=copy(actual)
            written=falses(length(actual))
            if !zero_alpha
                for i in 1:m, j in 1:n
                    position=cblas271_matrix_index(order, la, i, j)
                    written[position]=true
                    right=conjugate_y ? conj(logical_y[j]) : logical_y[j]
                    expected[position]+=alpha*logical_x[i]*right
                end
            end
            cblas271_ger(
                conjugate_y,
                order,
                m,
                n,
                alpha,
                zero_alpha ? Ptr{T}(C_NULL) : x,
                ix,
                zero_alpha ? Ptr{T}(C_NULL) : y,
                iy,
                actual,
                la,
                J,
                route,
            )
            @test all(eachindex(actual)) do index
                if written[index]
                    isapprox(actual[index], expected[index]; atol=tolerance, rtol=tolerance)
                else
                    isequal(actual[index], initial[index])
                end
            end
        end
    end
end

@testset "CBLAS dense and packed symmetric/Hermitian rank updates" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        route in (:dll, :lbt_trampoline)

        hermitian=T<:Complex
        tolerance=real(T)===Float32 ? 4e-5 : 1e-11
        for n in (1, 4),
            two in (false, true),
            storage in (:dense, :packed),
            order in (101, 102),
            u in (121, 122),
            ix in (-2, 1, 2),
            iy in (-3, 1, 3),
            all_zero in (false, true),
            zero_alpha in (false, true)

            alpha=if zero_alpha
                zero(T)
            elseif hermitian&&!two
                T(0.75)
            else
                cblas271_value(T, 1, 2, 1)
            end
            logical_x=T[all_zero||i==2 ? zero(T) : cblas271_value(T, i, 1, 2) for i in 1:n]
            logical_y=T[all_zero||i==3 ? zero(T) : cblas271_value(T, i, 2, 1) for i in 1:n]
            x=cblas271_storage(logical_x, ix)
            y=cblas271_storage(logical_y, iy)
            matrix=cblas271_symmetric_matrix(T, n, u, n, hermitian)
            initial, la=cblas271_triangle_storage(matrix, order, u, storage, n; hermitian)
            expected=copy(initial)
            written=falses(length(initial))
            pairs=if order==101
                [(i, j) for i in 1:n for j in 1:n if (u==121&&i<=j)||(u==122&&i>=j)]
            else
                [(i, j) for j in 1:n for i in 1:n if (u==121&&i<=j)||(u==122&&i>=j)]
            end
            if !zero_alpha
                for (packed_index, (i, j)) in enumerate(pairs)
                    position=if storage===:packed
                        packed_index
                    else
                        cblas271_matrix_index(order, la, i, j)
                    end
                    written[position]=true
                    right=two ? logical_y[j] : logical_x[j]
                    increment=alpha*logical_x[i]*(hermitian ? conj(right) : right)
                    if two
                        increment+=(hermitian ? conj(alpha) : alpha)*logical_y[i]*(
                            hermitian ? conj(logical_x[j]) : logical_x[j]
                        )
                    end
                    value=matrix[i, j]+increment
                    expected[position]=hermitian&&i==j ? T(real(value)) : value
                end
            end
            actual=copy(initial)
            cblas271_rank_update(
                two,
                storage,
                order,
                u,
                n,
                alpha,
                zero_alpha ? Ptr{T}(C_NULL) : x,
                ix,
                zero_alpha ? Ptr{T}(C_NULL) : y,
                iy,
                actual,
                la,
                J,
                route,
            )
            @test all(eachindex(actual)) do index
                if written[index]
                    isapprox(actual[index], expected[index]; atol=tolerance, rtol=tolerance)
                else
                    isequal(actual[index], initial[index])
                end
            end
        end
    end
end

const CBLAS_271_RANK_UPDATE_TARGETS=split(raw"""
cblas_cgerc cblas_cgeru cblas_cher cblas_cher2
cblas_chpr cblas_chpr2 cblas_dger cblas_dspr
cblas_dspr2 cblas_dsyr cblas_dsyr2 cblas_sger
cblas_sspr cblas_sspr2 cblas_ssyr cblas_ssyr2
cblas_zgerc cblas_zgeru cblas_zher cblas_zher2
cblas_zhpr cblas_zhpr2
""")
@testset "Every vector rank-update target completed through every ABI and route" begin
    @test length(CBLAS_271_RANK_UPDATE_TARGETS)==22
    @test Set(CBLAS_271_RANK_UPDATE_TARGETS)⊆Set(CBLAS_271_TARGETS)
    for bits in (32, 64), route in (:dll, :lbt_trampoline)
        observed=Set(
            name for (name, width, actual_route) in CBLAS_271_CALLS if
            width==bits&&actual_route==route&&name in CBLAS_271_RANK_UPDATE_TARGETS
        )
        @test observed==Set(CBLAS_271_RANK_UPDATE_TARGETS)
    end
end
