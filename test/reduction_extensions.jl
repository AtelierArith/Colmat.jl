using Test, LinearAlgebra, Random, Libdl, Colmat

for (T, prefix) in ((Float32, "s"), (Float64, "d")), J in (Int32, Int64)
    @eval function direct_extension_value(
        family,
        x::Vector{$T},
        ::Type{$J};
        n=length(x)-2,
        inc=1,
        library=Colmat.backend_library(),
    )
        symbol=Symbol($prefix*family*"_"*($J===Int64 ? "64_" : ""))
        f=Libdl.dlsym(Libdl.dlopen(library), symbol)
        xp=view(x, 2:(length(x) - 1))
        ccall(f, $T, (Ref{$J}, Ptr{$T}, Ref{$J}), n, xp, inc)
    end
end
for (T, prefix) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    J in (Int32, Int64)

    symbol=Symbol("i"*prefix*"amin_"*(J===Int64 ? "64_" : ""))
    @eval function direct_extension_index(
        x::Vector{$T}, ::Type{$J}; n=length(x)-2, inc=1, library=Colmat.backend_library()
    )
        f=Libdl.dlsym(Libdl.dlopen(library), $(QuoteNode(symbol)))
        xp=view(x, 2:(length(x) - 1))
        ccall(f, $J, (Ref{$J}, Ptr{$T}, Ref{$J}), n, xp, inc)
    end
end
function reduction_storage(x, inc)
    T=eltype(x)
    n=length(x)
    out=fill(T(NaN), max(1, 1+(n-1)*max(1, abs(inc)))+2)
    out[1]=out[end]=T(777)
    if inc>0
        for i in 1:n
            out[2 + (i - 1) * inc]=x[i]
        end
    elseif n>0
        out[2]=x[1]
    end
    return out
end
function reduction_reference()
    return only(
        filter(
            l->l.interface==:ilp64 && occursin("openblas", lowercase(basename(l.libname))),
            BLAS.get_config().loaded_libs,
        ),
    ).libname
end
function scalar_value_reference(family, x, n, inc)
    T=eltype(x)
    n<=0 && return zero(T)
    inc<0 && return zero(T)
    inc==0 && return if family=="sum"
        zero(T)
    elseif family in ("amax", "amin")
        abs(x[1])
    else
        x[1]
    end
    family=="max" && return maximum(x)
    family=="min" && return minimum(x)
    family=="amax" && return maximum(abs, x)
    family=="amin" && return minimum(abs, x)
    total=zero(T)
    for value in x
        total+=value
    end
    return total
end

@testset "BLAS real value extensions and stride contracts" begin
    reference=reduction_reference()
    rng=MersenneTwister(277)
    for T in (Float32, Float64),
        J in (Int32, Int64),
        family in ("amax", "amin", "max", "min", "sum"),
        n in (-3, 0, 1, 2, 17, 128, 129, 257),
        inc in (-4, -1, 0, 1, 2, 5)

        values=randn(rng, T, max(1, n))
        if n>1
            values[end]=-values[1]
        end
        x=reduction_storage(values, inc)
        prior=copy(x)
        expected=scalar_value_reference(family, values, n, inc)
        actual=direct_extension_value(family, x, J; n, inc)
        tolerance=8eps(T)*sum(abs, values)
        @test isapprox(actual, expected; atol=tolerance, rtol=0)
        if inc>=0
            external=direct_extension_value(family, x, Int64; n, inc, library=reference)
            @test isapprox(actual, external; atol=tolerance, rtol=0)
        end
        @test isequal(x, prior)
        @test actual isa T
    end
end

@testset "BLAS IAMIN abs1, first ties and logical Fortran indices" begin
    reference=reduction_reference()
    rng=MersenneTwister(278)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64), n in (-3, 0, 1, 2, 17, 128, 129, 257),
        inc in (-4, -1, 0, 1, 2, 5)

        R=real(T)
        values=randn(rng, T, max(1, n))
        if n>2
            values[2]=T(R(0.01))
            values[end]=-values[2]
        end
        scores=R[abs(real(v))+abs(imag(v)) for v in values]
        expected=n<=0||inc<=0 ? 0 : argmin(scores)
        x=reduction_storage(values, inc)
        prior=copy(x)
        actual=direct_extension_index(x, J; n, inc)
        @test actual==expected
        if inc>=0
            external=direct_extension_index(x, Int64; n, inc, library=reference)
            @test actual==external
        end
        @test actual isa J
        @test isequal(x, prior)
    end
    for T in (ComplexF32, ComplexF64), J in (Int32, Int64), inc in (1, 2, 5)
        # abs1 selects the first entry; Euclidean modulus selects the second.
        x=reduction_storage(T[3 + 0im, 2 + 2im], inc)
        @test direct_extension_index(x, J; n=2, inc)==1
        R=real(T)
        x=reduction_storage(T[T(1, eps(R)/2), T(1, 0)], inc)
        @test direct_extension_index(x, J; n=2, inc)==1
        @test direct_extension_index(x, Int64; n=2, inc, library=reference)==1
    end
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        inc in (1, 2, 5)

        x=reduction_storage(T[5, 4, 1, -1], inc)
        @test direct_extension_index(x, J; n=4, inc)==3
    end
end

@testset "BLAS reduction empty inputs, integer width and nonfinite scalars" begin
    for T in (Float32, Float64),
        J in (Int32, Int64),
        family in ("amax", "amin", "max", "min", "sum")

        x=T[777, 7, 3, 777]
        if J===Int64
            @test direct_extension_value(family, x, J; n=(-(Int64(2)^32)+2))==0
        end
        for value in (T(NaN), T(Inf), T(-Inf), T(0), T(-0.0))
            x=T[777, value, 777]
            expected=family in ("amax", "amin") ? abs(value) : value
            actual=direct_extension_value(family, x, J; n=1)
            @test isequal(actual, expected) || (actual==expected && iszero(expected))
        end
    end
    for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
        x=T[777, 7, 3, 777]
        if J===Int64
            @test direct_extension_index(x, J; n=(-(Int64(2)^32)+2))==0
        end
        @test direct_extension_index(T[777, T(NaN), 777], J; n=1)==1
    end
end

@testset "BLAS reduction extension numerical LBT calls" begin
    for (T, prefix) in ((Float32, "s"), (Float64, "d")),
        J in (Int32, Int64),
        family in ("amax", "amin", "max", "min", "sum")

        backing=BLAS.lbt_find_backing_library(prefix*family*"_", J===Int32 ? :lp64 : :ilp64)
        @test backing!==nothing
        @test realpath(backing.libname)==realpath(Colmat.backend_library())
        values=T[-4, 2, 3]
        x=reduction_storage(values, 2)
        @test direct_extension_value(
            family, x, J; n=3, inc=2, library=BLAS.libblastrampoline
        ) == scalar_value_reference(family, values, 3, 2)
    end
    for (T, prefix) in
        ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
        J in (Int32, Int64)

        backing=BLAS.lbt_find_backing_library(
            "i"*prefix*"amin_", J===Int32 ? :lp64 : :ilp64
        )
        @test backing!==nothing
        @test realpath(backing.libname)==realpath(Colmat.backend_library())
        x=reduction_storage(T[5, 1, -1], 2)
        @test direct_extension_index(x, J; n=3, inc=2, library=BLAS.libblastrampoline)==2
    end
end
