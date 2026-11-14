using Test, LinearAlgebra, Libdl, Colmat
positive_input(a::Matrix) = a
positive_input(a::Vector) = view(a, 2:(length(a) - 1))
for (T, R, p) in (
        (Float32, Float32, "s"),
        (Float64, Float64, "d"),
        (ComplexF32, Float32, "c"),
        (ComplexF64, Float64, "z"),
    ),
    J in (Int32, Int64)

    suffix=J===Int64 ? "64_" : ""
    @eval function direct_positive_equil!(
        family, a::Array{$T}, ::Type{$J}; n, u=('U'), kd=2, library=Colmat.backend_library()
    )
        f=Libdl.dlsym(Libdl.dlopen(library), Symbol($p*family*"_"*$suffix))
        s=fill($R(777), max(0, n)+2)
        cond=fill($R(777), 3)
        amax=fill($R(777), 3)
        info=fill($J(-777), 3)
        source=positive_input(a)
        if family in ("poequ", "poequb")
            ccall(
                f,
                Cvoid,
                (Ref{$J}, Ptr{$T}, Ref{$J}, Ptr{$R}, Ptr{$R}, Ptr{$R}, Ptr{$J}),
                n,
                source,
                size(a, 1),
                pointer(s, 2),
                pointer(cond, 2),
                pointer(amax, 2),
                pointer(info, 2),
            )
        elseif family=="pbequ"
            ccall(
                f,
                Cvoid,
                (
                    Ref{UInt8},
                    Ref{$J},
                    Ref{$J},
                    Ptr{$T},
                    Ref{$J},
                    Ptr{$R},
                    Ptr{$R},
                    Ptr{$R},
                    Ptr{$J},
                ),
                u,
                n,
                kd,
                source,
                size(a, 1),
                pointer(s, 2),
                pointer(cond, 2),
                pointer(amax, 2),
                pointer(info, 2),
            )
        else
            ccall(
                f,
                Cvoid,
                (Ref{UInt8}, Ref{$J}, Ptr{$T}, Ptr{$R}, Ptr{$R}, Ptr{$R}, Ptr{$J}),
                u,
                n,
                source,
                pointer(s, 2),
                pointer(cond, 2),
                pointer(amax, 2),
                pointer(info, 2),
            )
        end
        @test s[1]==s[end]==777
        @test cond[1]==cond[3]==amax[1]==amax[3]==777
        @test info[1]==info[3]==-777
        if info[2]<0
            diagnostic=Libdl.dlsym(
                Libdl.dlopen(Colmat.backend_library()), :colmat_blas_last_error
            )
            @test ccall(diagnostic, Cint, ()) == -info[2]
        end
        return (; s=s[2:(end - 1)], cond=cond[2], amax=amax[2], info=info[2])
    end
end
function positive_reference()
    return only(
        filter(
            l->l.interface==:ilp64 && occursin("openblas", lowercase(basename(l.libname))),
            BLAS.get_config().loaded_libs,
        ),
    ).libname
end

@testset "Positive equilibration diagonal storage radix and INFO" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        family in ("poequ", "poequb", "pbequ", "ppequ"),
        n in 0:9,
        u in ('U', 'L'),
        scenario in 1:6

        R=typeof(real(zero(T)))
        kd=2
        values=if scenario==2
            R[nextfloat(R(0)), floatmin(R), 1 / 64, 1, 64, floatmax(R) / 16, floatmax(R)]
        elseif scenario==6
            R[
                prevfloat(R(4)),
                4,
                nextfloat(R(4)),
                prevfloat(R(0.25)),
                0.25,
                nextfloat(R(0.25)),
            ]
        else
            R[1, 2, 3, 4, 7, 16, 81]
        end
        diagonal=R[values[mod1(i, length(values))] for i in 1:n]
        if n>0 && scenario in (3, 4, 5)
            diagonal[cld(n, 2)]=if scenario==3
                R(0)
            elseif scenario==4
                R(-1)
            else
                R(-0.0)
            end
        end
        a=if family=="ppequ"
            fill(T(NaN), n*(n+1)÷2+2)
        elseif family=="pbequ"
            fill(T(NaN), kd+3, n+1)
        else
            fill(T(NaN), n+2, n+1)
        end
        for i in 1:n
            value=T <: Complex ? T(diagonal[i], NaN) : T(diagonal[i])
            if family=="ppequ"
                idx=u=='U' ? i*(i+1)÷2 : 1+(i-1)*n-(i-1)*(i-2)÷2
                a[idx + 1]=value
            elseif family=="pbequ"
                a[u=='U' ? kd+1 : 1, i]=value
            else
                a[i, i]=value
            end
        end
        before=copy(a)
        result=direct_positive_equil!(family, a, J; n, u, kd)
        @test isequal(a, before)
        bad=findfirst(x->x<=0, diagonal)
        @test result.info==something(bad, 0)
        if n==0
            @test result.cond==1
            @test result.amax==0
        else
            @test result.amax==maximum(diagonal)
            if bad!==nothing
                @test isequal(result.s, diagonal)
                @test result.cond==777
            else
                @test result.cond≈sqrt(minimum(diagonal))/sqrt(maximum(diagonal))
                if family=="poequb"
                    @test all(x->isfinite(x)&&x>0&&isinteger(log2(x)), result.s)
                    @test all(
                        i->BigFloat(1)/4<=BigFloat(diagonal[i])*BigFloat(result.s[i])^2<=4,
                        1:n,
                    )
                else
                    @test all(
                        i->isapprox(result.s[i], inv(sqrt(diagonal[i])); rtol=2eps(R)), 1:n
                    )
                end
            end
        end
        if J===Int64
            external=direct_positive_equil!(
                family, a, J; n, u, kd, library=positive_reference()
            )
            @test result.info==external.info
            @test isequal(result.s, external.s)
            @test isequal(result.cond, external.cond)
            @test isequal(result.amax, external.amax)
        end
    end
end

@testset "Positive equilibration invalid arguments and guarded outputs" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
        for (family, n, u, kd, ld, arg) in (
            ("poequ", -1, 'U', 0, 1, 1),
            ("poequb", 3, 'U', 0, 1, 3),
            ("pbequ", 3, '?', 0, 3, 1),
            ("pbequ", -1, 'U', 0, 1, 2),
            ("pbequ", 3, 'L', -1, 3, 3),
            ("pbequ", 3, 'U', 2, 1, 5),
            ("ppequ", 3, '?', 0, 1, 1),
            ("ppequ", -1, 'U', 0, 1, 2),
        )
            a=family=="ppequ" ? fill(T(NaN), 8) : fill(T(NaN), ld, 4)
            result=direct_positive_equil!(family, a, J; n, u, kd)
            @test result.info == -arg
            @test all(==(777), result.s)
            @test result.cond==result.amax==777
        end
    end
end
