using Test, LinearAlgebra, Colmat

for (T, prefix) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    J in (Int32, Int64)

    coefficient=T<:Complex ? :(Ref{$T}) : T
    alpha=T<:Complex ? :(Ref(alpha)) : :alpha
    beta=T<:Complex ? :(Ref(beta)) : :beta
    R=real(T)
    @eval function cblas271_rankk(
        two,
        hermitian,
        order,
        u,
        t,
        n,
        k,
        alpha::$T,
        a,
        la,
        b,
        lb,
        beta::$T,
        c,
        lc,
        ::Type{$J},
        route,
    )
        name="cblas_"*$prefix*(hermitian ? "her" : "syr")*(two ? "2k" : "k")
        if two
            if hermitian
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
                    $R,
                    Ptr{$T},
                    $J,
                ) order u t $J(n) $J(k) $alpha a $J(la) b $J(lb) $R(real(beta)) c $J(lc)
            else
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
                ) order u t $J(n) $J(k) $alpha a $J(la) b $J(lb) $beta c $J(lc)
            end
        elseif hermitian
            @cblas271_call name $J route Cvoid (
                Cint, Cint, Cint, $J, $J, $R, Ptr{$T}, $J, $R, Ptr{$T}, $J
            ) order u t $J(n) $J(k) $R(real(alpha)) a $J(la) $R(real(beta)) c $J(lc)
        else
            @cblas271_call name $J route Cvoid (
                Cint,
                Cint,
                Cint,
                $J,
                $J,
                $coefficient,
                Ptr{$T},
                $J,
                $coefficient,
                Ptr{$T},
                $J,
            ) order u t $J(n) $J(k) $alpha a $J(la) $beta c $J(lc)
        end
        return nothing
    end
end

@testset "CBLAS SYRK/SYR2K/HERK/HER2K independent rank products" begin
    n=3
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        route in (:dll, :lbt_trampoline)

        tolerance=real(T)===Float32 ? 4e-5 : 1e-11
        for two in (false, true),
            hermitian in (T<:Complex ? (false, true) : (false,)), order in (101, 102),
            u in (121, 122)

            modes=if T<:Real
                (111, 112, 113, 114)
            elseif hermitian
                (111, 113)
            else
                (111, 112)
            end
            for t in modes,
                k in (0, 4), zero_alpha in (false, true),
                zero_beta in (false, true)

                rows, cols=t in (111, 114) ? (n, k) : (k, n)
                logical_a=T[cblas271_value(T, i, j, 1) for i in 1:rows, j in 1:cols]
                logical_b=T[cblas271_value(T, i, j, 3) for i in 1:rows, j in 1:cols]
                la=max(1, order==101 ? cols : rows)+2
                lb=la+1
                lc=n+2
                a=cblas271_pack(logical_a, order, la)
                b=cblas271_pack(logical_b, order, lb)
                oa=cblas271_op(logical_a, t)
                ob=cblas271_op(logical_b, t)
                alpha=if zero_alpha
                    zero(T)
                elseif hermitian&&!two
                    T(0.75)
                else
                    cblas271_value(T, 1, 2, 1)
                end
                beta=if zero_beta
                    zero(T)
                elseif hermitian
                    T(-0.25)
                else
                    cblas271_value(T, 2, 1, -5)
                end
                initial=fill(T(NaN), lc*n+3)
                expected=copy(initial)
                written=falses(length(initial))
                for i in 1:n, j in 1:n
                    if (u==121&&i<=j)||(u==122&&i>=j)
                        position=cblas271_matrix_index(order, lc, i, j)
                        written[position]=true
                        old=cblas271_value(T, i, j, 4)
                        if hermitian&&i==j
                            old=T(real(old))
                        end
                        initial[position]=if zero_beta
                            T(NaN)
                        elseif hermitian&&i==j
                            cblas271_poison_diagonal(old)
                        else
                            old
                        end
                        product=zero(T)
                        if !zero_alpha
                            for l in 1:k
                                right=two ? ob[j, l] : oa[j, l]
                                product+=alpha*oa[i, l]*(hermitian ? conj(right) : right)
                                if two
                                    product+=(hermitian ? conj(alpha) : alpha)*ob[i, l]*(
                                        hermitian ? conj(oa[j, l]) : oa[j, l]
                                    )
                                end
                            end
                        end
                        value=product+(zero_beta ? zero(T) : beta*old)
                        expected[position]=hermitian&&i==j ? T(real(value)) : value
                    end
                end
                actual=copy(initial)
                cblas271_rankk(
                    two,
                    hermitian,
                    order,
                    u,
                    t,
                    n,
                    k,
                    alpha,
                    zero_alpha||k==0 ? Ptr{T}(C_NULL) : a,
                    la,
                    zero_alpha||k==0 ? Ptr{T}(C_NULL) : b,
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
                        isequal(actual[index], initial[index])
                    end
                end
            end
        end
    end
end

@testset "Rank-k true no-ops preserve output and need no arrays" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        route in (:dll, :lbt_trampoline),
        two in (false, true),
        hermitian in (T<:Complex ? (false, true) : (false,)),
        order in (101, 102),
        u in (121, 122)

        for k in (0, 3)
            ccall(Libdl.dlsym(CBLAS_271_NATIVE, "colmat_blas_last_error"), Cint, ())
            n=2
            la=(order==101 ? max(1, k) : n)+2
            lb=la+1
            lc=n+2
            alpha=k==0 ? one(T) : zero(T)
            original=fill(T(NaN), lc*n+3)
            actual=copy(original)
            cblas271_rankk(
                two,
                hermitian,
                order,
                u,
                111,
                n,
                k,
                alpha,
                Ptr{T}(C_NULL),
                la,
                Ptr{T}(C_NULL),
                lb,
                one(T),
                actual,
                lc,
                J,
                route,
            )
            @test isequal(actual, original)
            @test ccall(Libdl.dlsym(CBLAS_271_NATIVE, "colmat_blas_last_error"), Cint, ())==0
            cblas271_rankk(
                two,
                hermitian,
                order,
                u,
                111,
                n,
                k,
                alpha,
                Ptr{T}(C_NULL),
                la,
                Ptr{T}(C_NULL),
                lb,
                one(T),
                Ptr{T}(C_NULL),
                lc,
                J,
                route,
            )
            @test ccall(Libdl.dlsym(CBLAS_271_NATIVE, "colmat_blas_last_error"), Cint, ())==0
        end
    end
end

const CBLAS_271_RANKK_TARGETS=split(raw"""
cblas_ssyrk cblas_dsyrk cblas_csyrk cblas_zsyrk
cblas_ssyr2k cblas_dsyr2k cblas_csyr2k cblas_zsyr2k
cblas_cherk cblas_zherk cblas_cher2k cblas_zher2k
""")
@testset "Every rank-k target completed through every ABI and route" begin
    @test length(CBLAS_271_RANKK_TARGETS)==12
    @test Set(CBLAS_271_RANKK_TARGETS)⊆Set(CBLAS_271_TARGETS)
    for bits in (32, 64), route in (:dll, :lbt_trampoline)
        observed=Set(
            name for (name, width, actual_route) in CBLAS_271_CALLS if
            width==bits&&actual_route==route&&name in CBLAS_271_RANKK_TARGETS
        )
        @test observed==Set(CBLAS_271_RANKK_TARGETS)
    end
end

for J in (Int32, Int64)
    @eval function cblas271_vector_noops(::Type{$J}, route)
        zero_value=Ref(ComplexF64(0))
        one_value=Ref(ComplexF64(1))
        ccall(Libdl.dlsym(CBLAS_271_NATIVE, "colmat_blas_last_error"), Cint, ())
        @cblas271_call "cblas_zgemv" $J route Cvoid (
            Cint,
            Cint,
            $J,
            $J,
            Ref{ComplexF64},
            Ptr{ComplexF64},
            $J,
            Ptr{ComplexF64},
            $J,
            Ref{ComplexF64},
            Ptr{ComplexF64},
            $J,
        ) 102 111 $J(2) $J(3) zero_value C_NULL $J(2) C_NULL $J(-2) one_value C_NULL $J(-3)
        @test ccall(Libdl.dlsym(CBLAS_271_NATIVE, "colmat_blas_last_error"), Cint, ())==0
        @cblas271_call "cblas_zhemv" $J route Cvoid (
            Cint,
            Cint,
            $J,
            Ref{ComplexF64},
            Ptr{ComplexF64},
            $J,
            Ptr{ComplexF64},
            $J,
            Ref{ComplexF64},
            Ptr{ComplexF64},
            $J,
        ) 101 121 $J(2) zero_value C_NULL $J(2) C_NULL $J(-2) one_value C_NULL $J(-3)
        @test ccall(Libdl.dlsym(CBLAS_271_NATIVE, "colmat_blas_last_error"), Cint, ())==0
    end
end
@testset "GEMV/HEMV true no-ops need no negative-stride buffers" begin
    for J in (Int32, Int64), route in (:dll, :lbt_trampoline)
        cblas271_vector_noops(J, route)
    end
end
