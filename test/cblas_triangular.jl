using Test, LinearAlgebra, Colmat

function cblas271_triangular_matrix(::Type{T}, n, u, k, unit) where {T}
    return T[
        if abs(i-j)>k||!((u==121&&i<=j)||(u==122&&i>=j))
            zero(T)
        elseif i==j
            unit ? one(T) : T(2)+cblas271_value(T, i, i, 5)
        else
            cblas271_value(T, i, j, 1)/T(3)
        end for i in 1:n, j in 1:n
    ]
end

for (T, prefix) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    J in (Int32, Int64)

    coefficient=T<:Complex ? :(Ref{$T}) : T
    alpha=T<:Complex ? :(Ref(alpha)) : :alpha
    @eval function cblas271_triangular_mv(
        storage, solve, order, u, t, d, n, k, a, ld, x::Vector{$T}, inc, ::Type{$J}, route
    )
        family=(
            if storage===:dense
                "tr"
            elseif storage===:band
                "tb"
            else
                "tp"
            end
        )*(solve ? "sv" : "mv")
        name="cblas_"*$prefix*family
        if storage===:dense
            @cblas271_call name $J route Cvoid (
                Cint, Cint, Cint, Cint, $J, Ptr{$T}, $J, Ptr{$T}, $J
            ) order u t d $J(n) a $J(ld) x $J(inc)
        elseif storage===:band
            @cblas271_call name $J route Cvoid (
                Cint, Cint, Cint, Cint, $J, $J, Ptr{$T}, $J, Ptr{$T}, $J
            ) order u t d $J(n) $J(k) a $J(ld) x $J(inc)
        else
            @cblas271_call name $J route Cvoid (
                Cint, Cint, Cint, Cint, $J, Ptr{$T}, Ptr{$T}, $J
            ) order u t d $J(n) a x $J(inc)
        end
        return nothing
    end
    @eval function cblas271_triangular_mm(
        solve,
        order,
        side,
        u,
        t,
        d,
        m,
        n,
        alpha::$T,
        a,
        la,
        b::Vector{$T},
        lb,
        ::Type{$J},
        route,
    )
        name="cblas_"*$prefix*(solve ? "trsm" : "trmm")
        @cblas271_call name $J route Cvoid (
            Cint, Cint, Cint, Cint, Cint, $J, $J, $coefficient, Ptr{$T}, $J, Ptr{$T}, $J
        ) order side u t d $J(m) $J(n) $alpha a $J(la) b $J(lb)
        return nothing
    end
end

@testset "CBLAS dense band and packed triangular products and solves" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        route in (:dll, :lbt_trampoline)

        tolerance=real(T)===Float32 ? 4e-5 : 1e-11
        for n in (1, 4),
            storage in (:dense, :band, :packed),
            order in (101, 102),
            u in (121, 122),
            t in 111:114,
            unit in (false, true)

            widths=storage===:band ? (0, 1, n+1) : (n+1,)
            for k in widths
                matrix=cblas271_triangular_matrix(T, n, u, k, unit)
                a, ld=cblas271_triangle_storage(matrix, order, u, storage, k; unit)
                operated=cblas271_op(matrix, t)
                logical_x=T[i==2 ? zero(T) : cblas271_value(T, i, 1, 2) for i in 1:n]
                product=fill(zero(T), n)
                for i in 1:n, j in 1:n
                    product[i]+=operated[i, j]*logical_x[j]
                end
                ap=unit&&(n==1||(storage===:band&&k==0)) ? Ptr{T}(C_NULL) : a
                for inc in (-2, 1, 3), solve in (false, true)
                    input=solve ? product : logical_x
                    output=solve ? logical_x : product
                    actual=cblas271_storage(input, inc)
                    expected=cblas271_storage(output, inc)
                    indices=cblas271_indices(n, inc)
                    cblas271_triangular_mv(
                        storage,
                        solve,
                        order,
                        u,
                        t,
                        unit ? 132 : 131,
                        n,
                        k,
                        ap,
                        ld,
                        actual,
                        inc,
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

@testset "CBLAS TRMM/TRSM independent left and right products" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        route in (:dll, :lbt_trampoline)

        tolerance=real(T)===Float32 ? 4e-5 : 1e-11
        for (m, n) in ((2, 3), (3, 2)),
            order in (101, 102), side in (141, 142), u in (121, 122), t in 111:114,
            unit in (false, true)

            dim=side==141 ? m : n
            matrix=cblas271_triangular_matrix(T, dim, u, dim, unit)
            a, la=cblas271_triangle_storage(matrix, order, u, :dense, dim; unit)
            operated=cblas271_op(matrix, t)
            known=T[cblas271_value(T, i, j, 2) for i in 1:m, j in 1:n]
            product=fill(zero(T), m, n)
            for i in 1:m, j in 1:n, l in 1:dim
                product[i, j]+=if side==141
                    operated[i, l]*known[l, j]
                else
                    known[i, l]*operated[l, j]
                end
            end
            lb=(order==101 ? n : m)+3
            for solve in (false, true), zero_alpha in (false, true)
                alpha=zero_alpha ? zero(T) : cblas271_value(T, 1, 2, 1)
                input=if zero_alpha
                    fill(T(NaN), m, n)
                elseif solve
                    product ./ alpha
                else
                    known
                end
                output=if zero_alpha
                    fill(zero(T), m, n)
                elseif solve
                    known
                else
                    alpha .* product
                end
                actual=cblas271_pack(input, order, lb)
                expected=cblas271_pack(output, order, lb)
                cblas271_triangular_mm(
                    solve,
                    order,
                    side,
                    u,
                    t,
                    unit ? 132 : 131,
                    m,
                    n,
                    alpha,
                    zero_alpha ? Ptr{T}(C_NULL) : a,
                    la,
                    actual,
                    lb,
                    J,
                    route,
                )
                written=falses(length(actual))
                for i in 1:m, j in 1:n
                    written[cblas271_matrix_index(order, lb, i, j)]=true
                end
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

const CBLAS_271_TRIANGULAR_TARGETS=split(raw"""
cblas_ctbmv cblas_ctbsv cblas_ctpmv cblas_ctpsv
cblas_ctrmm cblas_ctrmv cblas_ctrsm cblas_ctrsv
cblas_dtbmv cblas_dtbsv cblas_dtpmv cblas_dtpsv
cblas_dtrmm cblas_dtrmv cblas_dtrsm cblas_dtrsv
cblas_stbmv cblas_stbsv cblas_stpmv cblas_stpsv
cblas_strmm cblas_strmv cblas_strsm cblas_strsv
cblas_ztbmv cblas_ztbsv cblas_ztpmv cblas_ztpsv
cblas_ztrmm cblas_ztrmv cblas_ztrsm cblas_ztrsv
""")
@testset "Every triangular target completed through every ABI and route" begin
    @test length(CBLAS_271_TRIANGULAR_TARGETS)==32
    @test Set(CBLAS_271_TRIANGULAR_TARGETS)⊆Set(CBLAS_271_TARGETS)
    for bits in (32, 64), route in (:dll, :lbt_trampoline)
        observed=Set(
            name for (name, width, actual_route) in CBLAS_271_CALLS if
            width==bits&&actual_route==route&&name in CBLAS_271_TRIANGULAR_TARGETS
        )
        @test observed==Set(CBLAS_271_TRIANGULAR_TARGETS)
    end
end

@testset "CBLAS and standard Fortran BLAS triangular interoperability" begin
    for (T, prefix) in
        ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
        J in (Int32, Int64),
        route in (:dll, :lbt_trampoline)

        tolerance=real(T)===Float32 ? 4e-5 : 1e-11
        fortran_interface=LinearAlgebra.BlasInt===Int64 ? :ilp64 : :lp64
        fortran_suffix=LinearAlgebra.BlasInt===Int64 ? "64_" : ""
        for family in ("trmm", "trsm")
            name=prefix*family*"_"
            @test BLAS.lbt_get_forward(name, fortran_interface)==Libdl.dlsym(
                CBLAS_271_NATIVE, name*fortran_suffix
            )
        end
        for (m, n) in ((2, 3), (3, 2)),
            order in (101, 102), side in (141, 142), u in (121, 122), t in (111, 113),
            unit in (false, true)

            dim=side==141 ? m : n
            matrix=cblas271_triangular_matrix(T, dim, u, dim, unit)
            a, la=cblas271_triangle_storage(matrix, order, u, :dense, dim; unit)
            known=T[cblas271_value(T, i, j, 2) for i in 1:m, j in 1:n]
            lb=(order==101 ? n : m)+3
            side_f=side==141 ? 'L' : 'R'
            u_f=u==121 ? 'U' : 'L'
            t_f=t==111 ? 'N' : 'C'
            d_f=unit ? 'U' : 'N'
            # The C entry produces storage consumed by standard Fortran BLAS.
            c_product=cblas271_pack(known, order, lb)
            cblas271_triangular_mm(
                false,
                order,
                side,
                u,
                t,
                unit ? 132 : 131,
                m,
                n,
                one(T),
                a,
                la,
                c_product,
                lb,
                J,
                route,
            )
            transferred=T[
                c_product[cblas271_matrix_index(order, lb, i, j)] for i in 1:m, j in 1:n
            ]
            BLAS.trsm!(side_f, u_f, t_f, d_f, one(T), matrix, transferred)
            @test all(
                isapprox(transferred[i, j], known[i, j]; atol=tolerance, rtol=tolerance) for
                i in 1:m, j in 1:n
            )
            # Reverse direction: consume a standard Fortran product in C.
            fortran_product=copy(known)
            BLAS.trmm!(side_f, u_f, t_f, d_f, one(T), matrix, fortran_product)
            c_rhs=cblas271_pack(fortran_product, order, lb)
            cblas271_triangular_mm(
                true,
                order,
                side,
                u,
                t,
                unit ? 132 : 131,
                m,
                n,
                one(T),
                a,
                la,
                c_rhs,
                lb,
                J,
                route,
            )
            expected=cblas271_pack(known, order, lb)
            written=falses(length(c_rhs))
            for i in 1:m, j in 1:n
                written[cblas271_matrix_index(order, lb, i, j)]=true
            end
            @test all(eachindex(c_rhs)) do index
                if written[index]
                    isapprox(c_rhs[index], expected[index]; atol=tolerance, rtol=tolerance)
                else
                    c_rhs[index]==expected[index]
                end
            end
        end
    end
end
