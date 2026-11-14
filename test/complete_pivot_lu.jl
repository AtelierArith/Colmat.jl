using Test, LinearAlgebra, Random, Libdl, Colmat

for (T, prefix) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    J in (Int32, Int64)

    R=typeof(real(zero(T)))
    suffix=J===Int64 ? "64_" : ""
    getc2=Symbol(prefix*"getc2_"*suffix)
    gesc2=Symbol(prefix*"gesc2_"*suffix)
    @eval function direct_complete_factor!(
        a::Matrix{$T},
        ::Type{$J};
        n=size(a, 2),
        lda=max(1, size(a, 1)),
        library=Colmat.backend_library(),
    )
        ipiv=fill($J(777), size(a, 2)+2)
        jpiv=copy(ipiv)
        info=Ref{$J}(-777)
        f=Libdl.dlsym(Libdl.dlopen(library), $(QuoteNode(getc2)))
        ccall(
            f,
            Cvoid,
            (Ref{$J}, Ptr{$T}, Ref{$J}, Ptr{$J}, Ptr{$J}, Ref{$J}),
            n,
            a,
            lda,
            view(ipiv, 2:(length(ipiv) - 1)),
            view(jpiv, 2:(length(jpiv) - 1)),
            info,
        )
        (;
            info=info[],
            ipiv=ipiv[2:(end - 1)],
            jpiv=jpiv[2:(end - 1)],
            iguard=ipiv[[1, end]],
            jguard=jpiv[[1, end]],
        )
    end
    @eval function direct_complete_solve!(
        a::Matrix{$T},
        rhs::Vector{$T},
        ipiv::Vector{$J},
        jpiv::Vector{$J},
        ::Type{$J};
        n=size(a, 2),
        lda=max(1, size(a, 1)),
        library=Colmat.backend_library(),
    )
        f=Libdl.dlsym(Libdl.dlopen(library), $(QuoteNode(gesc2)))
        scale=fill($R(777), 3)
        ccall(
            f,
            Cvoid,
            (Ref{$J}, Ptr{$T}, Ref{$J}, Ptr{$T}, Ptr{$J}, Ptr{$J}, Ptr{$R}),
            n,
            a,
            lda,
            view(rhs, 2:(length(rhs) - 1)),
            ipiv,
            jpiv,
            view(scale, 2:2),
        )
        (; scale=scale[2], guard=scale[[1, end]])
    end
end
function complete_matrix(T, n)
    a=randn(T, n, n)
    for i in 1:n
        a[i, i]+=T(2)
    end
    if n>=3
        a[n, 2]+=T(20)
        a[1, n]+=T(10)
    end
    return a
end
function complete_storage(a)
    result=fill(eltype(a)(777), size(a, 1)+3, size(a, 2))
    result[1:size(a, 1), :]=a
    return result
end
function complete_effective(a, ipiv, jpiv)
    n=size(a, 2)
    factors=a[1:n, 1:n]
    lower=Matrix(UnitLowerTriangular(factors))
    upper=triu(factors)
    rows=collect(1:n)
    columns=copy(rows)
    for i in 1:n
        rows[i], rows[ipiv[i]]=rows[ipiv[i]], rows[i]
        columns[i], columns[jpiv[i]]=columns[jpiv[i]], columns[i]
    end
    effective=zeros(eltype(a), n, n)
    effective[rows, columns]=lower*upper
    return effective
end
function complete_big_residual(a, x, rhs, scale)
    B=eltype(a)<:Real ? BigFloat : Complex{BigFloat}
    ab, xb, bb=B.(a), B.(x), B.(rhs)
    residual=norm(ab*xb-BigFloat(scale)*bb)
    denominator=norm(ab)*norm(xb)+abs(BigFloat(scale))*norm(bb)
    return denominator==0 ? residual : residual/denominator
end
@testset "Complete-pivot native LU representation and scaled interoperability" begin
    Random.seed!(272)
    reference=only(
        filter(
            l->l.interface==:ilp64&&occursin("openblas", lowercase(basename(l.libname))),
            BLAS.get_config().loaded_libs,
        ),
    ).libname
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        n in (0, 1, 2, 5, 16, 40)

        R=typeof(real(zero(T)))
        tolerance=R===Float32 ? 8e-5 : 8e-13
        original=complete_matrix(T, n)
        a=complete_storage(original)
        prior=copy(a)
        result=direct_complete_factor!(a, J)
        @test result.info==0
        @test result.iguard==J[777, 777]
        @test result.jguard==J[777, 777]
        @test a[(n + 1):end, :]==prior[(n + 1):end, :]
        @test all(i->i<=result.ipiv[i]<=n, 1:n)
        @test all(i->i<=result.jpiv[i]<=n, 1:n)
        if n>0
            @test result.ipiv[end]==n
            @test result.jpiv[end]==n
        end
        @test complete_effective(a, result.ipiv, result.jpiv) ≈ original atol=tolerance
        rhs=randn(T, n)
        work=vcat(T(777), rhs, T(777))
        before=copy(a)
        solved=direct_complete_solve!(a, work, result.ipiv, result.jpiv, J)
        @test solved.scale==one(R)
        @test solved.guard==R[777, 777]
        @test work[[1, end]]==T[777, 777]
        @test original*work[2:(end - 1)] ≈ solved.scale*rhs atol=tolerance
        @test a==before
        if n>0
            external=complete_storage(original)
            metadata=direct_complete_factor!(external, Int64; library=reference)
            @test metadata.info==0
            work=vcat(T(777), rhs, T(777))
            solved=direct_complete_solve!(
                external, work, J.(metadata.ipiv), J.(metadata.jpiv), J
            )
            @test complete_big_residual(original, work[2:(end - 1)], rhs, solved.scale)<tolerance
            work=vcat(T(777), rhs, T(777))
            solved=direct_complete_solve!(
                a, work, Int64.(result.ipiv), Int64.(result.jpiv), Int64; library=reference
            )
            @test complete_big_residual(original, work[2:(end - 1)], rhs, solved.scale)<tolerance
        end
    end
end
@testset "Complete-pivot perturbation and subnormal pivots" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
        R=typeof(real(zero(T)))
        small=floatmin(R)/eps(R)
        tolerance=R===Float32 ? 2e-5 : 2e-12
        fixtures=(
            zeros(T, 3, 3),
            ones(T, 3, 3),
            Matrix(Diagonal(T[1, eps(R) / 4, 0])),
            fill(T(nextfloat(zero(R))), 3, 3),
        )
        for original in fixtures
            a=complete_storage(original)
            metadata=direct_complete_factor!(a, J)
            @test metadata.info==3
            @test all(isfinite, a)
            @test metadata.iguard==J[777, 777]
            @test metadata.jguard==J[777, 777]
            effective=complete_effective(a, metadata.ipiv, metadata.jpiv)
            difference=norm(BigFloat.(real.(effective-original)))+norm(
                BigFloat.(imag.(effective-original))
            )
            @test difference<=64BigFloat(eps(R))*norm(Complex{BigFloat}.(original))+8BigFloat(
                small
            )
            rhs=T[1, -2, 3]
            work=vcat(T(777), rhs, T(777))
            solved=direct_complete_solve!(a, work, metadata.ipiv, metadata.jpiv, J)
            @test 0<solved.scale<=1
            @test all(isfinite, work)
            @test complete_big_residual(effective, work[2:(end - 1)], rhs, solved.scale)<tolerance
        end
        for value in (zero(R), prevfloat(small), small, nextfloat(small), -small/2)
            a=complete_storage(fill(T(value), 1, 1))
            metadata=direct_complete_factor!(a, J)
            @test metadata.info==(abs(value)<small ? 1 : 0)
            @test a[1, 1]==(abs(value)<small ? T(small) : T(value))
            @test metadata.ipiv==J[1]
            @test metadata.jpiv==J[1]
        end
    end
end
@testset "Complete-pivot overflow guards and extreme finite scales" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
        R=typeof(real(zero(T)))
        tolerance=R===Float32 ? 4e-5 : 4e-12
        small=floatmin(R)/eps(R)
        extreme=R===Float32 ? R(1e20) : R(1e200)
        for matrix_scale in (inv(extreme), one(R), extreme),
            rhs_scale in (one(R), floatmax(R)/R(64))

            original=T[1 0.25 -0.125; 0.5 2 0.125; -0.25 0.5 3]*matrix_scale
            if T<:Complex
                original[1, 2]+=T(0, 0.25)*matrix_scale
            end
            a=complete_storage(original)
            metadata=direct_complete_factor!(a, J)
            @test metadata.info==0
            rhs=T[1, -0.5, 0.25]*rhs_scale
            work=vcat(T(777), rhs, T(777))
            solved=direct_complete_solve!(a, work, metadata.ipiv, metadata.jpiv, J)
            @test 0<solved.scale<=1
            @test all(isfinite, a)
            @test all(isfinite, work)
            @test complete_big_residual(original, work[2:(end - 1)], rhs, solved.scale)<tolerance
            @test work[[1, end]]==T[777, 777]
        end
        a=complete_storage(fill(T(small), 1, 1))
        metadata=direct_complete_factor!(a, J)
        rhs=T[floatmax(R) / R(64)]
        work=vcat(T(777), rhs, T(777))
        solved=direct_complete_solve!(a, work, metadata.ipiv, metadata.jpiv, J)
        @test 0<solved.scale<1
        @test all(isfinite, work)
        @test complete_big_residual(
            fill(T(small), 1, 1), work[2:(end - 1)], rhs, solved.scale
        )<tolerance
    end
end

@testset "Scaled upper solve preserves a representable nonzero solution" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
        R=typeof(real(zero(T)))
        n=112
        big=floatmax(R)/R(1024)
        upper=fill(zero(T), n, n)
        for j in 1:n, i in 1:j
            upper[i, j]=i==j ? T(big) : T(-big)
        end
        upper[n, n]=T(2eps(R)*big)
        a=complete_storage(upper)
        ipiv=J.(1:n)
        jpiv=copy(ipiv)
        rhs=fill(T(big/R(1024)), n)
        work=vcat(T(777), rhs, T(777))
        result=direct_complete_solve!(a, work, ipiv, jpiv, J)
        @test 0<result.scale<=1
        @test all(isfinite, work)
        B=T<:Real ? BigFloat : Complex{BigFloat}
        expected=UpperTriangular(B.(upper))\(BigFloat(result.scale)*B.(rhs))
        @test norm(B.(work[2:(end - 1)])-expected)/norm(expected)<(
            R===Float32 ? 2e-4 : 2e-12
        )
    end
end

@testset "Cumulative scale underflow returns a consistent zero solution" begin
    for T in (Float32, ComplexF32), J in (Int32, Int64)
        R=Float32
        n=300
        big=floatmax(R)/R(1024)
        upper=zeros(T, n, n)
        for j in 1:n, i in 1:j
            upper[i, j]=i==j ? T(big) : T(-big)
        end
        upper[n, n]=T(2eps(R)*big)
        rhs=fill(T(big/R(1024)), n)
        work=vcat(T(777), rhs, T(777))
        result=direct_complete_solve!(complete_storage(upper), work, J.(1:n), J.(1:n), J)
        @test result.scale==0
        @test all(iszero, work[2:(end - 1)])
        @test work[[1, end]]==T[777, 777]
        B=T<:Real ? BigFloat : Complex{BigFloat}
        smallest=BigFloat(nextfloat(zero(R)))
        expected=UpperTriangular(B.(upper))\(smallest*B.(rhs))
        @test maximum(abs, expected)>BigFloat(floatmax(R))
    end
end
@testset "Complete-pivot argument widths, ignored pivots and LBT calls" begin
    handle=Libdl.dlopen(Colmat.backend_library())
    last_error=Libdl.dlsym(handle, :colmat_blas_last_error)
    for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
        R=typeof(real(zero(T)))
        a=complete_storage(T[1 2; 3 5])
        before=copy(a)
        @test direct_complete_factor!(a, J; n=-1).info==-1
        @test direct_complete_factor!(a, J; lda=1).info==-3
        @test a==before
        for (options, arg) in (((; n=-1), 1), ((; lda=1), 3))
            ccall(last_error, Cint, ())
            rhs=T[777, 1, 2, 777]
            prior=copy(rhs)
            result=direct_complete_solve!(a, rhs, J[1, 2], J[1, 2], J; options...)
            @test result.scale==1
            @test ccall(last_error, Cint, ())==arg
            @test rhs==prior
            @test result.guard==R[777, 777]
        end
        if J===Int64
            @test direct_complete_factor!(a, J; n=(-(Int64(1)<<32))).info==-1
            empty=Matrix{T}(undef, 3, 0)
            @test direct_complete_factor!(empty, J; lda=(Int64(1)<<32)).info==0
            ccall(last_error, Cint, ())
            rhs=T[777, 777]
            @test direct_complete_solve!(empty, rhs, J[], J[], J; lda=(Int64(1)<<32)).scale==1
            @test ccall(last_error, Cint, ())==0
        end
        # GESC2 never uses the final pivot slots, including all slots for N=1.
        rhs=T[777, 4, 777]
        result=direct_complete_solve!(
            fill(T(2), 1, 1), rhs, J[typemin(J)], J[typemin(J)], J
        )
        @test result.scale==1
        @test rhs==T[777, 2, 777]
        a=complete_storage(T[1 2; 3 5])
        original=copy(a[1:2, :])
        metadata=direct_complete_factor!(a, J; library=BLAS.libblastrampoline)
        @test metadata.info==0
        rhs=T[777, 1, 2, 777]
        result=direct_complete_solve!(
            a, rhs, metadata.ipiv, metadata.jpiv, J; library=BLAS.libblastrampoline
        )
        @test original*rhs[2:3] ≈ result.scale*T[1, 2]
    end
    for prefix in ("s", "d", "c", "z"),
        family in (:getc2, :gesc2),
        interface in (:lp64, :ilp64)

        library=BLAS.lbt_find_backing_library(prefix*String(family)*"_", interface)
        @test library!==nothing
        @test realpath(library.libname)==realpath(Colmat.backend_library())
    end
end

@testset "Perturbed complete-pivot factors interoperate in both directions" begin
    reference=only(
        filter(
            l->l.interface==:ilp64&&occursin("openblas", lowercase(basename(l.libname))),
            BLAS.get_config().loaded_libs,
        ),
    ).libname
    for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
        R=typeof(real(zero(T)))
        tolerance=R===Float32 ? 8e-5 : 8e-13
        for original in
            (zeros(T, 3, 3), ones(T, 3, 3), Matrix(Diagonal(T[1, eps(R) / 4, 0])))
            own=complete_storage(original)
            metadata=direct_complete_factor!(own, J)
            external=complete_storage(original)
            other=direct_complete_factor!(external, Int64; library=reference)
            @test metadata.info==other.info==3
            for magnitude in (one(R), floatmax(R)/R(64))
                rhs=T[1, -1, 0.5]*magnitude
                work=vcat(T(777), rhs, T(777))
                result=direct_complete_solve!(
                    external, work, J.(other.ipiv), J.(other.jpiv), J
                )
                @test 0<result.scale<=1
                @test all(isfinite, work)
                @test complete_big_residual(
                    complete_effective(external, other.ipiv, other.jpiv),
                    work[2:(end - 1)],
                    rhs,
                    result.scale,
                )<tolerance
                work=vcat(T(777), rhs, T(777))
                result=direct_complete_solve!(
                    own,
                    work,
                    Int64.(metadata.ipiv),
                    Int64.(metadata.jpiv),
                    Int64;
                    library=reference,
                )
                @test 0<result.scale<=1
                @test all(isfinite, work)
                @test complete_big_residual(
                    complete_effective(own, metadata.ipiv, metadata.jpiv),
                    work[2:(end - 1)],
                    rhs,
                    result.scale,
                )<tolerance
            end
        end
    end
end
@testset "Complete-pivot logical submatrices and unused pivot slots" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
        n=3
        original=T[1 2 3; 4 6 -1; 2 -3 5]
        a=fill(T(777), n+3, n+2)
        a[1:n, 1:n]=original
        before=copy(a)
        metadata=direct_complete_factor!(a, J; n)
        @test metadata.info==0
        @test a[(n + 1):end, :]==before[(n + 1):end, :]
        @test a[:, (n + 1):end]==before[:, (n + 1):end]
        @test metadata.ipiv[(n + 1):end]==J[777, 777]
        @test metadata.jpiv[(n + 1):end]==J[777, 777]
        metadata.ipiv[n]=typemin(J)
        metadata.jpiv[n]=typemin(J)
        rhs=T[1, 2, 3]
        work=vcat(T(777), rhs, T[777, 777, 777])
        result=direct_complete_solve!(a, work, metadata.ipiv, metadata.jpiv, J; n)
        @test result.scale==1
        @test original*work[2:(n + 1)] ≈ rhs
        @test all(==(T(777)), work[(n + 2):end])
        @test work[1]==T(777)
    end
end
