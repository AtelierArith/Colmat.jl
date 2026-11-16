using Test, LinearAlgebra, Colmat

for (T, prefix) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    J in (Int32, Int64)

    R=real(T)
    coefficient=T<:Complex ? :(Ref{$T}) : T
    alpha=T<:Complex ? :(Ref(alpha)) : :alpha
    beta=T<:Complex ? :(Ref(beta)) : :beta
    norm_name=if T<:Complex
        "cblas_"*(T===ComplexF32 ? "scnrm2" : "dznrm2")
    else
        "cblas_"*prefix*"nrm2"
    end
    asum_name=if T<:Complex
        "cblas_"*(T===ComplexF32 ? "scasum" : "dzasum")
    else
        "cblas_"*prefix*"asum"
    end
    @eval function cblas271_vector_call(
        op,
        x::Vector{$T},
        ix,
        y::Vector{$T},
        iy,
        ::Type{$J},
        route;
        alpha=($T(0.75)),
        beta=($T(-0.25)),
    )
        n=$J(3)
        ix=$J(ix)
        iy=$J(iy)
        name="cblas_"*$prefix*op
        if op=="copy" || op=="swap"
            @cblas271_call name $J route Cvoid ($J, Ptr{$T}, $J, Ptr{$T}, $J) n x ix y iy
        elseif op=="scal"
            @cblas271_call name $J route Cvoid ($J, $coefficient, Ptr{$T}, $J) n $alpha x ix
        elseif op=="axpy"
            @cblas271_call name $J route Cvoid ($J, $coefficient, Ptr{$T}, $J, Ptr{$T}, $J) n $alpha x ix y iy
        elseif op=="axpby"
            @cblas271_call name $J route Cvoid (
                $J, $coefficient, Ptr{$T}, $J, $coefficient, Ptr{$T}, $J
            ) n $alpha x ix $beta y iy
        elseif op=="nrm2"
            return @cblas271_call $norm_name $J route $R ($J, Ptr{$T}, $J) n x ix
        elseif op=="asum"
            return @cblas271_call $asum_name $J route $R ($J, Ptr{$T}, $J) n x ix
        elseif op=="iamax" || op=="iamin"
            name="cblas_i"*$prefix*(op=="iamax" ? "amax" : "amin")
            return @cblas271_call name $J route Csize_t ($J, Ptr{$T}, $J) n x ix
        elseif op=="dotu" || op=="dotc"
            return @cblas271_call name $J route $T ($J, Ptr{$T}, $J, Ptr{$T}, $J) n x ix y iy
        elseif op=="rot"
            name="cblas_"*$(T<:Complex ? prefix*(T===ComplexF32 ? "s" : "d") : prefix)*"rot"
            @cblas271_call name $J route Cvoid ($J, Ptr{$T}, $J, Ptr{$T}, $J, $R, $R) n x ix y iy $R(
                0.6
            ) $R(0.8)
        elseif op=="real_scal"
            name="cblas_"*$(T===ComplexF32 ? "csscal" : "zdscal")
            @cblas271_call name $J route Cvoid ($J, $R, Ptr{$T}, $J) n $R(0.75) x ix
        else
            error("unrecognized level-one operation")
        end
        return nothing
    end
    @eval function cblas271_rotg(a::$T, b::$T, ::Type{$J}, route)
        ar=Ref(a)
        br=Ref(b)
        c=Ref{$R}(-97)
        s=Ref{$T}(-97)
        @cblas271_call $("cblas_"*prefix*"rotg") $J route Cvoid (
            Ref{$T}, Ref{$T}, Ref{$R}, Ref{$T}
        ) ar br c s
        return (; a=ar[], b=br[], c=c[], s=s[])
    end
    if T<:Real
        @eval function cblas271_reduction(op, x::Vector{$T}, inc, ::Type{$J}, route)
            return @cblas271_call ("cblas_"*$prefix*op) $J route $T ($J, Ptr{$T}, $J) $J(3) x $J(
                inc
            )
        end
    end
end

@testset "CBLAS Level 1 arithmetic through DLL and LBT" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        route in (:dll, :lbt_trampoline)

        R=real(T)
        atol=R===Float32 ? 2e-5 : 1e-12
        logical_x=T<:Complex ? T[1 + 2im, -3 + im, 2 - 4im] : T[1, -3, 4]
        logical_y=T<:Complex ? T[2 - im, 1 + 3im, -2 + im] : T[2, -1, 5]
        alpha=T<:Complex ? T(0.75, 0.25) : T(0.75)
        beta=T<:Complex ? T(-0.25, 0.125) : T(-0.25)
        for ix in (-2, 1, 2), iy in (-3, 1, 3)
            xi=cblas271_indices(3, ix)
            yi=cblas271_indices(3, iy)
            source=cblas271_storage(logical_x, ix)
            destination=cblas271_storage(logical_y, iy)
            for op in ("copy", "swap", "axpy", "axpby")
                x=copy(source)
                y=copy(destination)
                expected_x=copy(x)
                expected_y=copy(y)
                if op=="copy"
                    expected_y[yi]=logical_x
                elseif op=="swap"
                    expected_x[xi]=logical_y
                    expected_y[yi]=logical_x
                elseif op=="axpy"
                    expected_y[yi]=alpha .* logical_x .+ logical_y
                else
                    expected_y[yi]=alpha .* logical_x .+ beta .* logical_y
                end
                cblas271_vector_call(op, x, ix, y, iy, J, route; alpha, beta)
                @test x≈expected_x atol=atol rtol=atol
                @test y≈expected_y atol=atol rtol=atol
            end
            if T<:Complex
                for op in ("dotu", "dotc")
                    x=copy(source)
                    y=copy(destination)
                    expected=sum(
                        (op=="dotc" ? conj(logical_x[i]) : logical_x[i])*logical_y[i] for
                        i in 1:3
                    )
                    @test cblas271_vector_call(op, x, ix, y, iy, J, route)≈expected atol=atol rtol=atol
                end
            end
            rotation_name=if T===ComplexF32
                "cblas_csrot"
            elseif T===ComplexF64
                "cblas_zdrot"
            else
                "cblas_"*(T===Float32 ? "s" : "d")*"rot"
            end
            if rotation_name in CBLAS_271_TARGETS
                x=copy(source)
                y=copy(destination)
                expected_x=copy(x)
                expected_y=copy(y)
                expected_x[xi]=R(0.6) .* logical_x .+ R(0.8) .* logical_y
                expected_y[yi]=R(0.6) .* logical_y .- R(0.8) .* logical_x
                cblas271_vector_call("rot", x, ix, y, iy, J, route)
                @test x≈expected_x atol=atol rtol=atol
                @test y≈expected_y atol=atol rtol=atol
            end
        end
        for inc in (-2, 1, 2)
            source=cblas271_storage(logical_x, inc)
            indices=cblas271_indices(3, inc)
            for op in (T<:Complex ? ("scal", "real_scal") : ("scal",))
                x=copy(source)
                expected=copy(x)
                if inc>0
                    expected[indices]=logical_x .* (op=="real_scal" ? R(0.75) : alpha)
                end
                cblas271_vector_call(op, x, inc, T[], 1, J, route; alpha)
                @test x≈expected atol=atol rtol=atol
            end
            expected_norm=inc>0 ? sqrt(sum(abs2, logical_x)) : zero(R)
            expected_asum=if inc>0
                sum(abs(real(x))+abs(imag(x)) for x in logical_x)
            else
                zero(R)
            end
            @test cblas271_vector_call("nrm2", source, inc, T[], 1, J, route)≈expected_norm atol=atol rtol=atol
            @test cblas271_vector_call("asum", source, inc, T[], 1, J, route)≈expected_asum atol=atol rtol=atol
            magnitudes=[abs(real(x))+abs(imag(x)) for x in logical_x]
            @test cblas271_vector_call("iamax", source, inc, T[], 1, J, route)==(
                inc>0 ? argmax(magnitudes)-1 : 0
            )
            @test cblas271_vector_call("iamin", source, inc, T[], 1, J, route)==(
                inc>0 ? argmin(magnitudes)-1 : 0
            )
            if T<:Real
                for op in ("amax", "amin", "sum")
                    expected=if inc<=0
                        zero(T)
                    elseif op=="amax"
                        maximum(abs, logical_x)
                    elseif op=="amin"
                        minimum(abs, logical_x)
                    else
                        sum(logical_x)
                    end
                    @test cblas271_reduction(op, source, inc, J, route)≈expected atol=atol rtol=atol
                end
            end
        end
        prefix=if T===Float32
            "s"
        elseif T===Float64
            "d"
        elseif T===ComplexF32
            "c"
        else
            "z"
        end
        if "cblas_"*prefix*"rotg" in CBLAS_271_TARGETS
            a, b=logical_x[1], logical_y[1]
            rotation=cblas271_rotg(a, b, J, route)
            @test rotation.c*a+rotation.s*b≈rotation.a atol=atol rtol=atol
            @test -conj(rotation.s)*a+rotation.c*b≈zero(T) atol=atol rtol=atol
            @test abs2(rotation.c)+abs2(rotation.s)≈one(R) atol=atol rtol=atol
        end
    end
end

# Independent frozen Level-1 subset of #271, required for each ABI/route.
const CBLAS_271_LEVEL1 = split(raw"""
cblas_caxpby cblas_caxpy cblas_ccopy cblas_cdotc cblas_cdotu cblas_crotg
cblas_cscal cblas_csrot cblas_csscal cblas_cswap cblas_damax cblas_damin
cblas_dasum cblas_daxpby cblas_daxpy cblas_dcopy cblas_dnrm2 cblas_drot
cblas_drotg cblas_dscal cblas_dsum cblas_dswap cblas_dzasum cblas_dznrm2
cblas_icamax cblas_icamin cblas_idamax cblas_idamin cblas_isamax cblas_isamin
cblas_izamax cblas_izamin cblas_samax cblas_samin cblas_sasum cblas_saxpby
cblas_saxpy cblas_scasum cblas_scnrm2 cblas_scopy cblas_snrm2 cblas_srot
cblas_srotg cblas_sscal cblas_ssum cblas_sswap cblas_zaxpby cblas_zaxpy
cblas_zcopy cblas_zdotc cblas_zdotu cblas_zdscal cblas_zscal cblas_zswap
""")
@testset "Every Level-1 target completed through every ABI and route" begin
    @test length(CBLAS_271_LEVEL1)==54
    @test Set(CBLAS_271_LEVEL1) ⊆ Set(CBLAS_271_TARGETS)
    for bits in (32, 64), route in (:dll, :lbt_trampoline)
        observed=Set(
            name for (name, width, actual_route) in CBLAS_271_CALLS if
            width==bits&&actual_route==route&&name in CBLAS_271_LEVEL1
        )
        @test observed==Set(CBLAS_271_LEVEL1)
    end
    for (_, _, _, file, line, signature) in CBLAS_271_CALL_SITES
        @test isfile(file)
        @test line>0
        @test !isempty(signature)
    end
end
