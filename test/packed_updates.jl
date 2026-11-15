using Test, LinearAlgebra, Libdl, Colmat

for (T, R, p) in (
        (Float32, Float32, "s"),
        (Float64, Float64, "d"),
        (ComplexF32, Float32, "c"),
        (ComplexF64, Float64, "z"),
    ),
    J in (Int32, Int64)

    suffix=J===Int64 ? "64_" : ""
    @eval function direct_packed_update!(
        family,
        ap::Vector{$T},
        x::Vector{$T},
        y::Vector{$T},
        ::Type{$J};
        n,
        u,
        alpha,
        ix,
        iy=1,
        library=Colmat.backend_library(),
    )
        f=Libdl.dlsym(Libdl.dlopen(library), Symbol($p*family*"_"*$suffix))
        if family=="hpr"
            ccall(
                f,
                Cvoid,
                (Ref{UInt8}, Ref{$J}, Ref{$R}, Ptr{$T}, Ref{$J}, Ptr{$T}),
                u,
                n,
                alpha,
                pointer(x, 2),
                ix,
                pointer(ap, 2),
            )
        elseif family=="spr"
            ccall(
                f,
                Cvoid,
                (Ref{UInt8}, Ref{$J}, Ref{$T}, Ptr{$T}, Ref{$J}, Ptr{$T}),
                u,
                n,
                alpha,
                pointer(x, 2),
                ix,
                pointer(ap, 2),
            )
        else
            ccall(
                f,
                Cvoid,
                (Ref{UInt8}, Ref{$J}, Ref{$T}, Ptr{$T}, Ref{$J}, Ptr{$T}, Ref{$J}, Ptr{$T}),
                u,
                n,
                alpha,
                pointer(x, 2),
                ix,
                pointer(y, 2),
                iy,
                pointer(ap, 2),
            )
        end
    end
end
function packed_update_reference()
    return only(
        filter(
            l->l.interface==:ilp64 && occursin("openblas", lowercase(basename(l.libname))),
            BLAS.get_config().loaded_libs,
        ),
    ).libname
end
function packed_update_indices(n, inc)
    return if n==0
        Int[]
    else
        collect(inc>0 ? (2:inc:(2 + (n - 1) * inc)) : ((2 + (n - 1) * abs(inc)):inc:2))
    end
end
function packed_update_fixture(T, family, n, u)
    hermitian=startswith(family, "h")
    ap=fill(T(777), n*(n+1)÷2+2)
    positions=Tuple{Int,Int}[]
    for j in 1:n
        rows=u=='U' ? (1:j) : (j:n)
        for i in rows
            push!(positions, (i, j))
            value=T <: Complex ? T((i+2j)/8, (i-j+2)/16) : T((i+2j)/8)
            ap[length(positions) + 1]=hermitian&&i==j ? T(real(value), NaN) : value
        end
    end
    return ap, positions
end
function packed_update_oracle(family, before, positions, x, y, alpha)
    T=eltype(before)
    hermitian=startswith(family, "h")
    two=endswith(family, "2")
    expected=copy(before)
    alpha==0 && return expected
    for (k, (i, j)) in enumerate(positions)
        first=alpha*x[i]*(hermitian ? conj(two ? y[j] : x[j]) : (two ? y[j] : x[j]))
        increment=if two
            first+(hermitian ? conj(alpha) : alpha)*y[i]*(hermitian ? conj(x[j]) : x[j])
        else
            first
        end
        expected[k + 1]=if hermitian&&i==j
            T(real(before[k + 1])+real(increment), 0)
        else
            before[k + 1]+increment
        end
    end
    return expected
end

@testset "Packed native rank updates coefficients conjugation and strides" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        family in (T <: Complex ? ("spr", "spr2", "hpr", "hpr2") : ("spr", "spr2")),
        n in 0:9,
        u in ('U', 'L'),
        ix in (-3, -1, 1, 2),
        iy in (endswith(family, "2") ? (-2, -1, 1, 3) : (1,)),
        scenario in 1:4

        R=typeof(real(zero(T)))
        alpha=if scenario==4
            zero(T)
        elseif scenario==2
            (T <: Complex ? T(-0.5, 0.25) : T(-0.5))
        else
            (T <: Complex ? T(0.5, -1) : T(0.5))
        end
        if family=="hpr"
            alpha=if scenario==4
                R(0)
            elseif scenario==2
                R(-0.5)
            else
                R(0.5)
            end
        end
        ap, positions=packed_update_fixture(T, family, n, u)
        xi=packed_update_indices(n, ix)
        yi=packed_update_indices(n, iy)
        x=fill(T(NaN), max(1, (n-1)*abs(ix)+1)+2)
        y=fill(T(NaN), max(1, (n-1)*abs(iy)+1)+2)
        for (i, idx) in enumerate(xi)
            x[idx]=if scenario==4
                T(NaN)
            elseif scenario==3 || scenario==2&&iseven(i)
                zero(T)
            elseif T <: Complex
                T(i/4, (i+1)/8)
            else
                T(i/4)
            end
        end
        for (i, idx) in enumerate(yi)
            y[idx]=if scenario==4
                T(NaN)
            elseif scenario==2&&isodd(i)
                zero(T)
            elseif T <: Complex
                T(-i/8, (i+2)/4)
            else
                T(-i/8)
            end
        end
        before=copy(ap)
        beforex=copy(x)
        beforey=copy(y)
        expected=packed_update_oracle(family, before, positions, x[xi], y[yi], alpha)
        direct_packed_update!(family, ap, x, y, J; n, u, alpha, ix, iy)
        @test isapprox(
            ap[2:(end - 1)],
            expected[2:(end - 1)];
            rtol=100eps(R),
            atol=100eps(R),
            nans=true,
        )
        @test ap[1]==ap[end]==777
        @test isequal(x, beforex)
        @test isequal(y, beforey)
        if alpha==0 || n==0
            @test isequal(ap, before)
        elseif startswith(family, "h")
            for (k, (i, j)) in enumerate(positions)
                if i==j
                    @test isequal(imag(ap[k + 1]), R(0))
                end
            end
        end
        if J===Int64
            external=copy(before)
            # OpenBLAS 0.3.30's complex SPR/SPR2 wrappers advance a FLOAT*
            # by complex element counts for negative strides. Compare the
            # same logical vectors using positive contiguous reference input.
            # Colmat above still executes every original signed stride.
            normalize=T <: Complex && family in ("spr", "spr2") && (ix<0||iy<0)
            reference_x=normalize ? vcat(T(777), x[xi], T(777)) : x
            reference_y=normalize ? vcat(T(777), y[yi], T(777)) : y
            direct_packed_update!(
                family,
                external,
                reference_x,
                reference_y,
                J;
                n,
                u,
                alpha,
                ix=normalize ? 1 : ix,
                iy=normalize ? 1 : iy,
                library=packed_update_reference(),
            )
            @test isapprox(
                ap[2:(end - 1)],
                external[2:(end - 1)];
                rtol=100eps(R),
                atol=100eps(R),
                nans=true,
            )
            @test external[1]==external[end]==777
        end
    end
end

@testset "All 12 packed rank symbols numerically through LBT" begin
    for (T, p) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
        J in (Int32, Int64),
        family in (T <: Complex ? ("spr", "spr2", "hpr", "hpr2") : ("spr", "spr2"))

        ap, positions=packed_update_fixture(T, family, 4, 'L')
        x=T[777, 1, 2, 3, 4, 777]
        y=T[777, 2, 1, -1, 3, 777]
        alpha=family=="hpr" ? typeof(real(zero(T)))(0.5) : T(0.5)
        expected=packed_update_oracle(family, ap, positions, x[2:5], y[2:5], alpha)
        backing=BLAS.lbt_find_backing_library(p*family*"_", J===Int32 ? :lp64 : :ilp64)
        @test backing!==nothing
        @test realpath(backing.libname)==realpath(Colmat.backend_library())
        direct_packed_update!(
            family,
            ap,
            x,
            y,
            J;
            n=4,
            u=('L'),
            alpha,
            ix=1,
            iy=1,
            library=BLAS.libblastrampoline,
        )
        @test ap≈expected
    end
end
