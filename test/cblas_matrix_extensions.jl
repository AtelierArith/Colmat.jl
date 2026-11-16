using Test, LinearAlgebra, Colmat

for (T, prefix) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    J in (Int32, Int64)

    coefficient=T<:Complex ? :(Ref{$T}) : T
    alpha=T<:Complex ? :(Ref(alpha)) : :alpha
    beta=T<:Complex ? :(Ref(beta)) : :beta
    @eval function cblas271_matrixcopy(
        inplace, order, t, m, n, alpha::$T, a, la, b::Vector{$T}, lb, ::Type{$J}, route
    )
        name="cblas_"*$prefix*(inplace ? "imatcopy" : "omatcopy")
        if inplace
            @cblas271_call name $J route Cvoid (
                Cint, Cint, $J, $J, $coefficient, Ptr{$T}, $J, $J
            ) order t $J(m) $J(n) $alpha a $J(la) $J(lb)
        else
            @cblas271_call name $J route Cvoid (
                Cint, Cint, $J, $J, $coefficient, Ptr{$T}, $J, Ptr{$T}, $J
            ) order t $J(m) $J(n) $alpha a $J(la) b $J(lb)
        end
        return nothing
    end
    @eval function cblas271_geadd(
        order, m, n, alpha::$T, a, la, beta::$T, b, lb, ::Type{$J}, route
    )
        name="cblas_"*$prefix*"geadd"
        @cblas271_call name $J route Cvoid (
            Cint, $J, $J, $coefficient, Ptr{$T}, $J, $coefficient, Ptr{$T}, $J
        ) order $J(m) $J(n) $alpha a $J(la) $beta b $J(lb)
        return nothing
    end
end

@testset "CBLAS out-of-place and overlapping in-place matrix copies" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        route in (:dll, :lbt_trampoline)

        tolerance=real(T)===Float32 ? 4e-5 : 1e-11
        for (m, n) in ((2, 5), (5, 2), (1, 4), (0, 3), (2, 0)),
            order in (101, 102), t in 111:114,
            alpha in (zero(T), one(T), cblas271_value(T, 1, 2, 1))

            logical=T[cblas271_value(T, i, j, 1) for i in 1:m, j in 1:n]
            operated=cblas271_op(logical, t)
            rows, cols=size(operated)
            la=max(1, order==101 ? n : m)+2
            lb=max(1, order==101 ? cols : rows)+3
            source=cblas271_pack(logical, order, la)
            output_length=lb*(order==101 ? rows : cols)+3
            append!(source, fill(T(97), max(0, output_length-length(source))))
            for inplace in (false, true)
                a=copy(source)
                b=fill(T(97), output_length)
                initial=inplace ? copy(a) : copy(b)
                expected=copy(initial)
                written=falses(length(initial))
                for i in 1:rows, j in 1:cols
                    position=cblas271_matrix_index(order, lb, i, j)
                    written[position]=true
                    expected[position]=alpha*operated[i, j]
                end
                ap=!inplace&&T<:Real&&iszero(alpha) ? Ptr{T}(C_NULL) : a
                cblas271_matrixcopy(inplace, order, t, m, n, alpha, ap, la, b, lb, J, route)
                actual=inplace ? a : b
                @test all(eachindex(actual)) do index
                    if written[index]
                        isapprox(
                        actual[index], expected[index]; atol=tolerance, rtol=tolerance
                    )
                    else
                        isequal(actual[index], initial[index])
                    end
                end
                if !inplace
                    @test isequal(a, source)
                end
            end
        end
    end
end

@testset "CBLAS GEADD independent matrix combinations" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        route in (:dll, :lbt_trampoline)

        tolerance=real(T)===Float32 ? 4e-5 : 1e-11
        coefficients=(
            (zero(T), zero(T)),
            (zero(T), one(T)),
            (cblas271_value(T, 1, 2, 1), zero(T)),
            (cblas271_value(T, 1, 2, 1), cblas271_value(T, 2, 1, -5)),
        )
        for (m, n) in ((2, 5), (5, 2), (1, 4), (0, 3), (2, 0)),
            order in (101, 102),
            (alpha, beta) in coefficients

            logical_a=T[cblas271_value(T, i, j, 1) for i in 1:m, j in 1:n]
            logical_b=T[cblas271_value(T, i, j, 3) for i in 1:m, j in 1:n]
            la=max(1, order==101 ? n : m)+2
            lb=la+1
            a=cblas271_pack(logical_a, order, la)
            initial=cblas271_pack(iszero(beta) ? fill(T(NaN), m, n) : logical_b, order, lb)
            expected=copy(initial)
            written=falses(length(initial))
            for i in 1:m, j in 1:n
                position=cblas271_matrix_index(order, lb, i, j)
                written[position]=true
                expected[position]=alpha*logical_a[i, j]+beta*logical_b[i, j]
            end
            actual=copy(initial)
            cblas271_geadd(
                order,
                m,
                n,
                alpha,
                iszero(alpha) ? Ptr{T}(C_NULL) : a,
                la,
                beta,
                actual,
                lb,
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
            if iszero(alpha)&&isone(beta)
                ccall(Libdl.dlsym(CBLAS_271_NATIVE, "colmat_blas_last_error"), Cint, ())
                cblas271_geadd(
                    order,
                    m,
                    n,
                    alpha,
                    Ptr{T}(C_NULL),
                    la,
                    beta,
                    Ptr{T}(C_NULL),
                    lb,
                    J,
                    route,
                )
                @test ccall(
                    Libdl.dlsym(CBLAS_271_NATIVE, "colmat_blas_last_error"), Cint, ()
                )==0
            end
        end
    end
end

const CBLAS_271_MATRIX_TARGETS=split(raw"""
cblas_somatcopy cblas_domatcopy cblas_comatcopy cblas_zomatcopy
cblas_simatcopy cblas_dimatcopy cblas_cimatcopy cblas_zimatcopy
cblas_sgeadd cblas_dgeadd cblas_cgeadd cblas_zgeadd
""")
@testset "Every matrix extension completed through every ABI and route" begin
    @test length(CBLAS_271_MATRIX_TARGETS)==12
    @test Set(CBLAS_271_MATRIX_TARGETS)⊆Set(CBLAS_271_TARGETS)
    for bits in (32, 64), route in (:dll, :lbt_trampoline)
        observed=Set(
            name for (name, width, actual_route) in CBLAS_271_CALLS if
            width==bits&&actual_route==route&&name in CBLAS_271_MATRIX_TARGETS
        )
        @test observed==Set(CBLAS_271_MATRIX_TARGETS)
    end
end
