using Test, LinearAlgebra, Libdl, Colmat
for (T, R, p) in (
        (Float32, Float32, "s"),
        (Float64, Float64, "d"),
        (ComplexF32, Float32, "c"),
        (ComplexF64, Float64, "z"),
    ),
    J in (Int32, Int64)

    suffix=J===Int64 ? "64_" : ""
    @eval function direct_general_equil!(
        family,
        a::Matrix{$T},
        ::Type{$J};
        m,
        n,
        kl=1,
        ku=1,
        library=Colmat.backend_library(),
    )
        f=Libdl.dlsym(Libdl.dlopen(library), Symbol($p*family*"_"*$suffix))
        r=fill($R(777), max(0, m)+2)
        c=fill($R(777), max(0, n)+2)
        rc=fill($R(777), 3)
        cc=fill($R(777), 3)
        amax=fill($R(777), 3)
        info=fill($J(-777), 3)
        if startswith(family, "gb")
            ccall(
                f,
                Cvoid,
                (
                    Ref{$J},
                    Ref{$J},
                    Ref{$J},
                    Ref{$J},
                    Ptr{$T},
                    Ref{$J},
                    Ptr{$R},
                    Ptr{$R},
                    Ptr{$R},
                    Ptr{$R},
                    Ptr{$R},
                    Ptr{$J},
                ),
                m,
                n,
                kl,
                ku,
                a,
                size(a, 1),
                pointer(r, 2),
                pointer(c, 2),
                pointer(rc, 2),
                pointer(cc, 2),
                pointer(amax, 2),
                pointer(info, 2),
            )
        else
            ccall(
                f,
                Cvoid,
                (
                    Ref{$J},
                    Ref{$J},
                    Ptr{$T},
                    Ref{$J},
                    Ptr{$R},
                    Ptr{$R},
                    Ptr{$R},
                    Ptr{$R},
                    Ptr{$R},
                    Ptr{$J},
                ),
                m,
                n,
                a,
                size(a, 1),
                pointer(r, 2),
                pointer(c, 2),
                pointer(rc, 2),
                pointer(cc, 2),
                pointer(amax, 2),
                pointer(info, 2),
            )
        end
        @test r[1]==r[end]==c[1]==c[end]==777
        @test rc[1]==rc[3]==cc[1]==cc[3]==amax[1]==amax[3]==777
        @test info[1]==info[3]==-777
        if info[2]<0
            diagnostic=Libdl.dlsym(
                Libdl.dlopen(Colmat.backend_library()), :colmat_blas_last_error
            )
            @test ccall(diagnostic, Cint, ()) == -info[2]
        end
        return (;
            r=r[2:(end - 1)],
            c=c[2:(end - 1)],
            rc=rc[2],
            cc=cc[2],
            amax=amax[2],
            info=info[2],
        )
    end
end
function general_reference()
    return only(
        filter(
            l->l.interface==:ilp64 && occursin("openblas", lowercase(basename(l.libname))),
            BLAS.get_config().loaded_libs,
        ),
    ).libname
end

function general_oracle(a)
    m, n=size(a)
    R=typeof(real(zero(eltype(a))))
    r=fill(R(777), m)
    c=fill(R(777), n)
    rc=cc=amax=R(777)
    info=0
    if m==0||n==0
        return (; r, c, rc=R(1), cc=R(1), amax=R(0), info)
    end
    magnitude(x) = abs(real(x))+abs(imag(x))
    small=floatmin(R)
    big=inv(small)
    r=[maximum(magnitude(a[i, j]) for j in 1:n) for i in 1:m]
    amax=maximum(r)
    bad=findfirst(iszero, r)
    if bad!==nothing
        return (; r, c, rc, cc, amax, info=bad)
    end
    rc=max(min(big, minimum(r)), small)/min(maximum(r), big)
    r=inv.(clamp.(r, small, big))
    c=[maximum(magnitude(a[i, j])*r[i] for i in 1:m) for j in 1:n]
    bad=findfirst(iszero, c)
    if bad!==nothing
        return (; r, c, rc, cc, amax, info=m+bad)
    end
    cc=max(min(big, minimum(c)), small)/min(maximum(c), big)
    c=inv.(clamp.(c, small, big))
    return (; r, c, rc, cc, amax, info)
end

@testset "General and band equilibration row/column scaling and radix" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        family in ("geequ", "geequb", "gbequ", "gbequb"),
        m in 0:6,
        n in 0:6,
        width in (0, 1, 3),
        scenario in 1:5

        R=typeof(real(zero(T)))
        band=startswith(family, "gb")
        radix=endswith(family, "b")
        kl=width
        ku=width
        a=band ? fill(T(NaN), kl+ku+3, n+1) : fill(T(NaN), m+2, n+1)
        dense=zeros(T, m, n)
        extremes=R[nextfloat(R(0)), floatmin(R), 1 / 16, 1, 16, floatmax(R) / 16]
        for j in 1:n, i in 1:m
            if band && !(j-ku<=i<=j+kl)
                continue
            end
            value=scenario==5 ? extremes[mod1(i+j, 6)] : R(i+2j)/R(2)^(i+j)
            if scenario==4 || scenario==2 && i==cld(m, 2) || scenario==3 && j==cld(n, 2)
                value=R(0)
            end
            dense[i, j]=T <: Complex ? T(value, -value/2) : T(value)
            a[band ? ku+1+i-j : i, j]=dense[i, j]
        end
        before=copy(a)
        result=direct_general_equil!(family, a, J; m, n, kl, ku)
        @test isequal(a, before)
        if !radix
            expected=general_oracle(dense)
            @test isequal(result, expected)
        elseif result.info==0 && m>0 && n>0
            @test all(x->isfinite(x)&&x>0&&isinteger(log2(x)), result.r)
            @test all(x->isfinite(x)&&x>0&&isinteger(log2(x)), result.c)
            @test 0<=result.rc<=1
            @test 0<=result.cc<=1
        end
        if J===Int64
            external=direct_general_equil!(
                family, a, J; m, n, kl, ku, library=general_reference()
            )
            @test isequal(result, external)
        end
    end
end

@testset "General equilibration argument checks" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
        for (family, m, n, kl, ku, ld, arg) in (
            ("geequ", -1, 3, 0, 0, 1, 1),
            ("geequb", 3, -1, 0, 0, 3, 2),
            ("geequ", 3, 3, 0, 0, 1, 4),
            ("gbequ", -1, 3, 0, 0, 1, 1),
            ("gbequb", 3, -1, 0, 0, 1, 2),
            ("gbequ", 3, 3, -1, 0, 1, 3),
            ("gbequb", 3, 3, 0, -1, 1, 4),
            ("gbequ", 3, 3, 2, 2, 1, 6),
        )
            a=fill(T(NaN), ld, 4)
            result=direct_general_equil!(family, a, J; m, n, kl, ku)
            @test result.info == -arg
            @test all(==(777), result.r)
            @test all(==(777), result.c)
            @test result.rc==result.cc==result.amax==777
        end
    end
end

@testset "General equilibration uniform extreme magnitudes" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        family in ("geequ", "geequb", "gbequ", "gbequb")

        R=typeof(real(zero(T)))
        for value in (nextfloat(R(0)), floatmin(R), floatmax(R)/16, floatmax(R))
            a=fill(T(NaN), 3, 2)
            a[1, 1]=T(value)
            result=direct_general_equil!(family, a, J; m=1, n=1, kl=0, ku=0)
            @test result.info==0
            @test all(x->isfinite(x)&&x>0, result.r)
            @test all(x->isfinite(x)&&x>0, result.c)
            if J===Int64
                external=direct_general_equil!(
                    family, a, J; m=1, n=1, kl=0, ku=0, library=general_reference()
                )
                @test isequal(result, external)
            end
        end
    end
end
