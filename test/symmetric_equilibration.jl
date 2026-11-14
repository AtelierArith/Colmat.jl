using Test, LinearAlgebra, Libdl, Colmat, Random

for (T, R, p) in (
        (Float32, Float32, "s"),
        (Float64, Float64, "d"),
        (ComplexF32, Float32, "c"),
        (ComplexF64, Float64, "z"),
    ),
    J in (Int32, Int64)

    suffix=J===Int64 ? "64_" : ""
    @eval function direct_symmetric_equil!(
        family, a::Matrix{$T}, ::Type{$J}; n, u=('U'), library=Colmat.backend_library()
    )
        f=Libdl.dlsym(Libdl.dlopen(library), Symbol($p*family*"_"*$suffix))
        s=fill($R(777), max(0, n)+2)
        cond=fill($R(777), 3)
        amax=fill($R(777), 3)
        info=fill($J(-777), 3)
        work=fill($T(777), 2max(0, n)+2)
        ccall(
            f,
            Cvoid,
            (
                Ref{UInt8},
                Ref{$J},
                Ptr{$T},
                Ref{$J},
                Ptr{$R},
                Ptr{$R},
                Ptr{$R},
                Ptr{$T},
                Ptr{$J},
            ),
            u,
            n,
            a,
            size(a, 1),
            pointer(s, 2),
            pointer(cond, 2),
            pointer(amax, 2),
            pointer(work, 2),
            pointer(info, 2),
        )
        @test s[1]==s[end]==777
        @test cond[1]==cond[3]==amax[1]==amax[3]==777
        @test info[1]==info[3]==-777
        @test work[1]==work[end]==777
        if info[2]<0 && (!(u in ('U', 'L', 'u', 'l')) || n<0 || size(a, 1)<max(1, n))
            diagnostic=Libdl.dlsym(
                Libdl.dlopen(Colmat.backend_library()), :colmat_blas_last_error
            )
            @test ccall(diagnostic, Cint, ()) == -info[2]
        end
        return (; s=s[2:(end - 1)], cond=cond[2], amax=amax[2], info=info[2])
    end
end
function symmetric_reference()
    return only(
        filter(
            l->l.interface==:ilp64 && occursin("openblas", lowercase(basename(l.libname))),
            BLAS.get_config().loaded_libs,
        ),
    ).libname
end

@testset "BIN symmetric equilibration native connection and selected storage" begin
    rng=MersenneTwister(280)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        family in (T <: Complex ? ("syequb", "heequb") : ("syequb",)),
        n in 0:12,
        u in ('U', 'L'),
        scenario in 1:4

        R=typeof(real(zero(T)))
        b=randn(rng, T, n, n)
        core=family=="syequb" && T <: Complex ? b*transpose(b)+T(n+1)*I : b*b'+T(n+1)*I
        weights=R[R(2)^(mod(3i, 13)-6) for i in 1:n]
        dense=Diagonal(weights)*core*Diagonal(weights)
        if scenario==2
            dense=Matrix{T}(I, n, n)
        elseif scenario==3
            dense=ones(T, n, n)
        elseif scenario==4
            dense=Matrix(Diagonal(T[T(2)^(2*(i-1)-n) for i in 1:n]))
        end
        a=fill(T(NaN), n+3, n+1)
        for j in 1:n, i in 1:n
            if (u=='U'&&i<=j)||(u=='L'&&i>=j)
                a[i, j]=dense[i, j]
            end
        end
        before=copy(a)
        result=direct_symmetric_equil!(family, a, J; n, u)
        @test result.info==0
        @test isequal(a, before)
        if n==0
            @test result.cond==1
            @test result.amax==0
        else
            magnitude=abs.(real.(dense))+abs.(imag.(dense))
            @test result.amax==maximum(magnitude)
            @test all(x->isfinite(x)&&x>0&&isinteger(log2(x)), result.s)
            @test result.cond≈maximum((minimum(result.s), floatmin(R)))/minimum((
                maximum(result.s), inv(floatmin(R))
            ))
            scaled=Diagonal(Float64.(result.s))*Float64.(magnitude)*Diagonal(
                Float64.(result.s)
            )
            energies=vec(sum(scaled; dims=2))
            @test maximum(energies)/minimum(energies)<=100
            if scenario==4
                @test cond(scaled)<=16
            elseif scenario==1 && !(family=="syequb" && T <: Complex)
                equilibrated=Diagonal(result.s)*dense*Diagonal(result.s)
                @test cond(equilibrated)<=16max(1, n)*cond(core)
            end
        end
        if J===Int64
            external=direct_symmetric_equil!(
                family, a, J; n, u, library=symmetric_reference()
            )
            @test external.info==0
            @test result.amax==external.amax
            if n>0
                # Different native GEMV accumulation and safe normalization may move a
                # coefficient across a radix boundary; allow one binary exponent.
                @test all(i->0.5<=result.s[i]/external.s[i]<=2, 1:n)
            end
        end
    end
end

@testset "BIN extreme finite scales and zero rows" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        family in (T <: Complex ? ("syequb", "heequb") : ("syequb",)),
        u in ('U', 'L')

        R=typeof(real(zero(T)))
        for values in (
            R[nextfloat(R(0)), 1, floatmax(R)],
            fill(nextfloat(R(0)), 3),
            fill(floatmax(R), 3),
        )
            a=fill(T(NaN), 5, 4)
            for j in 1:3, i in 1:3
                if (u=='U'&&i<=j)||(u=='L'&&i>=j)
                    a[i, j]=i==j ? T(values[i]) : zero(T)
                end
            end
            result=direct_symmetric_equil!(family, a, J; n=3, u)
            @test result.info==0
            @test result.amax==maximum(values)
            @test all(x->isfinite(x)&&x>0, result.s)
            diagonal=BigFloat.(values) .* BigFloat.(result.s) .^ 2
            @test maximum(diagonal)/minimum(diagonal)<=16
        end
        a=zeros(T, 4, 3)
        a[1, 1]=1
        a[3, 3]=1
        result=direct_symmetric_equil!(family, a, J; n=3, u)
        @test result.info==2
        @test result.cond==777
    end
end

@testset "BIN argument validation and INFO width" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        family in (T <: Complex ? ("syequb", "heequb") : ("syequb",))

        for (n, u, ld, arg) in ((3, '?', 3, 1), (-1, 'U', 1, 2), (3, 'L', 1, 4))
            a=fill(T(NaN), ld, 4)
            result=direct_symmetric_equil!(family, a, J; n, u)
            @test result.info == -arg
            @test all(==(777), result.s)
            @test result.cond==result.amax==777
        end
    end
end
