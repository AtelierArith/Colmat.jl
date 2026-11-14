using Test, LinearAlgebra, Libdl, Colmat

for (T, R, p) in (
        (Float32, Float32, "s"),
        (Float64, Float64, "d"),
        (ComplexF32, Float32, "c"),
        (ComplexF64, Float64, "z"),
    ),
    J in (Int32, Int64)

    suffix=J===Int64 ? "64_" : ""
    @eval function direct_lascl!(
        a::Matrix{$T},
        ::Type{$J};
        kind,
        m,
        n,
        kl=0,
        ku=0,
        from,
        to,
        library=Colmat.backend_library(),
    )
        f=Libdl.dlsym(Libdl.dlopen(library), Symbol($p*"lascl_"*$suffix))
        info=fill($J(-777), 3)
        ccall(
            f,
            Cvoid,
            (
                Ref{UInt8},
                Ref{$J},
                Ref{$J},
                Ref{$R},
                Ref{$R},
                Ref{$J},
                Ref{$J},
                Ptr{$T},
                Ref{$J},
                Ptr{$J},
            ),
            kind,
            kl,
            ku,
            from,
            to,
            m,
            n,
            a,
            size(a, 1),
            pointer(info, 2),
        )
        @test info[1]==info[3]==-777
        if info[2]<0
            diagnostic=Libdl.dlsym(
                Libdl.dlopen(Colmat.backend_library()), :colmat_blas_last_error
            )
            @test ccall(diagnostic, Cint, ()) == -info[2]
        end
        return info[2]
    end
end

function scaling_locations(kind, m, n, kl, ku)
    positions=Tuple{Int,Int}[]
    for j in 1:n, i in 1:m
        selected=kind=='G' ||
                 kind=='L' && i>=j ||
                 kind=='U' && i<=j ||
                 kind=='H' && i<=j+1 ||
                 kind=='B' && j<=i<=j+kl ||
                 kind=='Q' && j-ku<=i<=j ||
                 kind=='Z' && j-ku<=i<=j+kl
        selected || continue
        row=if kind=='B'
            i-j+1
        elseif kind=='Q'
            ku+1+i-j
        elseif kind=='Z'
            kl+ku+1+i-j
        else
            i
        end
        push!(positions, (row, j))
    end
    return positions
end

@testset "LASCL selected storage and safe scaling" begin
    reference=only(
        filter(
            l->l.interface==:ilp64 && occursin("openblas", lowercase(basename(l.libname))),
            BLAS.get_config().loaded_libs,
        ),
    ).libname
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64), kind in ('G', 'L', 'U', 'H', 'B', 'Q', 'Z'), m in 0:5,
        n0 in 0:5

        n=kind in ('B', 'Q') ? m : n0
        kl=max(0, min(2, m-1))
        ku=kind in ('B', 'Q') ? kl : max(0, min(1, n-1))
        ld=if kind=='B'
            kl+3
        elseif kind=='Q'
            ku+3
        elseif kind=='Z'
            2kl+ku+3
        else
            m+2
        end
        R=typeof(real(zero(T)))
        positions=scaling_locations(kind, m, n, kl, ku)
        for (from, to) in (
            (R(2), R(3)),
            (nextfloat(R(0)), R(1)),
            (R(1), nextfloat(R(0))),
            (floatmax(R), R(1)),
            (R(1), floatmax(R)),
        )
            a=fill(T(777), ld, n+1)
            expected=copy(a)
            for (index, (i, j)) in enumerate(positions)
                value=T <: Complex ? T(R(index)/16, -R(index)/32) : T(R(index)/16)
                # Include values that recover finite numbers when the ratio itself overflows.
                if from==nextfloat(R(0))
                    value=T <: Complex ? T(from, -from) : T(from)
                elseif to==nextfloat(R(0))
                    value=T <: Complex ? T(floatmax(R)/4, -floatmax(R)/8) : T(floatmax(R)/4)
                end
                a[i, j]=value
                re=R(BigFloat(real(value))*BigFloat(to)/BigFloat(from))
                im=R(BigFloat(imag(value))*BigFloat(to)/BigFloat(from))
                expected[i, j]=T <: Complex ? T(re, im) : T(re)
            end
            before=copy(a)
            @test direct_lascl!(a, J; kind, m, n, kl, ku, from, to)==0
            for idx in eachindex(a)
                @test isequal(a[idx], expected[idx]) ||
                    isapprox(a[idx], expected[idx]; rtol=4eps(R), atol=nextfloat(R(0)))
            end
            if J===Int64
                external=copy(before)
                @test direct_lascl!(
                    external, J; kind, m, n, kl, ku, from, to, library=reference
                )==0
                @test isequal(a, external)
            end
        end
    end
end

@testset "LASCL nonfinite and signed zero components" begin
    # Preserve the reference Fortran A(I,J)*MUL expression, including its
    # complex promotion: (0+Inf*im)*0 yields NaN+NaN*im and (1-0im)*0
    # yields 0+0im with gfortran -O2 and OpenBLAS 0.3.30 on this platform.
    # Componentwise real multiplication is not the reference at these edges.
    # https://github.com/Reference-LAPACK/lapack/blob/v3.12.1/SRC/zlascl.f
    reference=only(
        filter(
            l->l.interface==:ilp64 && occursin("openblas", lowercase(basename(l.libname))),
            BLAS.get_config().loaded_libs,
        ),
    ).libname
    for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
        R=typeof(real(zero(T)))
        for (from, to) in (
            (R(Inf), R(1)), (R(Inf), R(Inf)), (R(1), R(Inf)), (R(-1), R(-0.0)), (R(1), R(1))
        )
            values=R[0.0, -0.0, Inf, -Inf, NaN, 1, -1]
            a=reshape(
                T[T <: Complex ? T(x, y) : T(x) for x in values for y in values], 7, 7
            )
            external=copy(a)
            @test direct_lascl!(a, J; kind=('G'), m=7, n=7, from, to)==0
            @test direct_lascl!(
                external, Int64; kind=('G'), m=7, n=7, from, to, library=reference
            )==0
            @test isequal(a, external)
        end
    end
end

@testset "LASCL argument precedence and INFO width" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
        R=typeof(real(zero(T)))
        base=(kind=('G'), m=3, n=3, kl=0, ku=0, from=R(1), to=R(2))
        cases=[
            ((kind=('?'), from=R(0)), 1),
            ((from=R(0), m=-1), 4),
            ((from=R(NaN),), 4),
            ((to=R(NaN), m=-1), 5),
            ((m=-1,), 6),
            ((n=-1,), 7),
            ((kind=('B'), n=2, kl=-1), 7),
            ((kind=('Z'), kl=-1), 2),
            ((kind=('Z'), ku=-1), 3),
            ((kind=('Q'), kl=1, ku=0), 3),
            ((m=7,), 9),
            ((kind=('Z'), kl=2, ku=2), 9),
        ]
        for (overrides, arg) in cases
            a=fill(T(777), 5, 4)
            before=copy(a)
            @test direct_lascl!(a, J; merge(base, overrides)...) == -arg
            @test isequal(a, before)
        end
    end
end
