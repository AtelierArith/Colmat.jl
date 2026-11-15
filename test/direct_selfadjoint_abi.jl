using Test, LinearAlgebra, Libdl, Random, Colmat
isdefined(@__MODULE__, :factor_abi_matrix) || include("direct_factor_abi.jl")
for J in (Int32, Int64),
    T in (Float32, Float64, ComplexF32, ComplexF64),
    family in (:ev, :evd, :evr)

    R=real(T)
    signature=Any[Ref{UInt8}]
    args=Any[:job]
    if family==:evr
        push!(signature, Ref{UInt8})
        push!(args, :range)
    end
    append!(signature, [Ref{UInt8}, Ref{J}, Ptr{T}, Ref{J}])
    append!(args, [:uplo, :n, :(x.a.data), :(stride(x.a.data, 2))])
    if family==:evr
        append!(signature, [Ref{R}, Ref{R}, Ref{J}, Ref{J}, Ref{R}, Ptr{J}])
        append!(args, [:vl, :vu, :il, :iu, zero(R), :(x.count)])
    end
    push!(signature, Ptr{R})
    push!(args, :(x.values))
    if family==:evr
        append!(signature, [Ptr{T}, Ref{J}, Ptr{J}])
        append!(args, [:(x.z.data), :(stride(x.z.data, 2)), :(x.support)])
    end
    append!(signature, [Ptr{T}, Ref{J}])
    append!(args, [:(x.work), :lw])
    if T<:Complex
        push!(signature, Ptr{R})
        push!(args, :(x.rwork))
        if family!=:ev
            push!(signature, Ref{J})
            push!(args, :lrw)
        end
    end
    if family!=:ev
        append!(signature, [Ptr{J}, Ref{J}])
        append!(args, [:(x.iwork), :liw])
    end
    append!(signature, [Ptr{J}, Clong, Clong])
    append!(args, [:(x.info), 1, 1])
    if family==:evr
        push!(signature, Clong)
        push!(args, 1)
    end
    @eval function direct_selfadjoint_call(
        f,
        ::Val{$(QuoteNode(family))},
        ::Type{$T},
        ::Type{$J},
        x,
        n,
        job,
        uplo,
        range,
        vl,
        vu,
        il,
        iu,
        lw,
        lrw,
        liw,
    )
        ccall(f, Cvoid, ($(signature...),), $(args...))
    end
end
function direct_selfadjoint_minima(T, family, n, job)
    if family==:ev
        return max(1, (T<:Complex ? 2n-1 : 3n-1)), max(1, 3n-2), 1
    elseif family==:evr
        return max(1, (T<:Complex ? 2n : 26n)), max(1, 24n), max(1, 10n)
    end
    n<=1 && return 1, 1, 1
    if T<:Complex
        return job=='V' ? (2n+n*n, 1+5n+2n*n, 3+5n) : (n+1, n, 1)
    end
    return job=='V' ? (1+6n+2n*n, 1, 3+5n) : (2n+1, 1, 1)
end
function direct_selfadjoint_fixture(T, J, original, family, job; selected=size(original, 1))
    n=size(original, 1)
    lw, lrw, liw=direct_selfadjoint_minima(T, family, n, job)
    a=factor_abi_matrix(original)
    z=factor_abi_matrix(
        zeros(
            T,
            family==:evr && job=='V' ? max(1, n) : 1,
            family==:evr && job=='V' ? max(1, selected) : 1,
        ),
    )
    wg, work=factor_abi_integer(T, lw)
    rg, rwork=factor_abi_integer(real(T), lrw)
    pg, iwork=factor_abi_integer(J, liw)
    sg, support=factor_abi_integer(J, family==:evr && job=='V' ? 2max(1, selected) : 1)
    vg, values=factor_abi_integer(real(T), max(1, n))
    mg, count=factor_abi_integer(J)
    ig, info=factor_abi_integer(J)
    return (;
        a, z, wg, work, rg, rwork, pg, iwork, sg, support, vg, values, mg, count, ig, info
    )
end
function direct_selfadjoint_guards(x)
    factor_abi_guards(x.a)
    factor_abi_guards(x.z)
    for array in (x.wg, x.rg, x.pg, x.sg, x.vg, x.mg, x.ig)
        factor_abi_integer_guards(array)
    end
end
@testset "Direct selfadjoint EVD integer ABI contracts" begin
    Random.seed!(31719)
    library=Libdl.dlopen(Colmat.backend_library())
    original_library=only(
        filter(
            l->l.interface==:ilp64 && occursin("openblas", lowercase(l.libname)),
            BLAS.get_config().loaded_libs,
        ),
    ).libname
    reference_library=Libdl.dlopen(original_library)
    for (T, prefix) in
        ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
        J in (Int32, Int64),
        route in (:dll, :lbt),
        family in (:ev, :evd, :evr),
        n in (0, 1, 5),
        job in ('N', 'V'),
        uplo in ('L', 'U'),
        range in (family==:evr ? ('A', 'I', 'V') : ('A',))

        @testset "$T/$J/$route/$family/$n/$job/$uplo/$range" begin
            symbol=prefix*(T<:Complex ? "he" : "sy")*string(family)
            f=if route==:dll
                Libdl.dlsym(library, symbol*(J==Int32 ? "_" : "_64_"))
            else
                Libdl.dlsym(
                    Libdl.dlopen(BLAS.libblastrampoline), symbol*(J==Int32 ? "_" : "_64_")
                )
            end
            reference=Libdl.dlsym(reference_library, symbol*"_64_")
            noise=randn(T, n, n)
            original=Matrix(Diagonal(T.(1:n)))+real(T)(0.05)*(noise+noise')
            y=direct_selfadjoint_fixture(T, Int64, original, family, job)
            il, iu=n<=1 ? (1, n) : (2, n-1)
            vl, vu=real(T)(0.5), real(T)(2.5)
            direct_selfadjoint_call(
                reference,
                Val(family),
                T,
                Int64,
                y,
                n,
                job,
                uplo,
                range,
                vl,
                vu,
                il,
                iu,
                length(y.work),
                length(y.rwork),
                length(y.iwork),
            )
            @test y.info[1]==0
            direct_selfadjoint_guards(y)
            selected=family==:evr ? Int(y.count[1]) : n
            0<=selected<=n || error("Reference returned invalid selected dimension")
            x=direct_selfadjoint_fixture(T, J, original, family, job; selected)
            saved_count, saved_support, saved_values=copy(x.count),
            copy(x.support),
            copy(x.values)
            lw, lrw, liw=length(x.work), length(x.rwork), length(x.iwork)
            call(lw, lrw, liw; j=job, u=uplo) = direct_selfadjoint_call(
                f, Val(family), T, J, x, n, j, u, range, vl, vu, il, iu, lw, lrw, liw
            )
            call(-1, lrw, liw)
            @test x.info[1]==0 && real(x.work[1])>=lw
            @test x.a.data==original
            @test x.count==saved_count &&
                x.support==saved_support &&
                x.values==saved_values &&
                all(iszero, x.z.data)
            if family!=:ev
                @test x.iwork[1]>=liw
                T<:Complex && @test x.rwork[1]>=lrw
            end
            direct_selfadjoint_guards(x)
            # Test each workspace length independently to preserve argument precedence.
            call(lw-1, lrw, liw)
            @test x.info[1]==-(family==:evr ? 18 : 8)
            @test factor_abi_diagnostic() == -x.info[1]
            @test x.a.data==original
            if family!=:ev
                call(lw, lrw, liw-1)
                @test x.info[1]==-(
                    family==:evr ? (T<:Complex ? 22 : 20) : (T<:Complex ? 12 : 10)
                )
                @test factor_abi_diagnostic() == -x.info[1]
                if T<:Complex
                    call(lw, lrw-1, liw)
                    @test x.info[1]==-(family==:evr ? 20 : 10)
                    @test factor_abi_diagnostic() == -x.info[1]
                end
            end
            call(lw, lrw, liw; j=('?'))
            @test x.info[1]==-1 && x.a.data==original
            @test factor_abi_diagnostic() == -x.info[1]
            call(lw, lrw, liw; u=('?'))
            @test x.info[1]==-(family==:evr ? 3 : 2) && x.a.data==original
            @test factor_abi_diagnostic() == -x.info[1]
            call(lw, lrw, liw)
            @test x.info[1]==0
            count=family==:evr ? Int(x.count[1]) : n
            @test 0<=count<=n
            family==:evr && @test count==y.count[1]
            @test x.values[1:count]≈y.values[1:count]
            @test issorted(x.values[1:count])
            range=='I' && @test count==max(0, iu-il+1)
            range=='V' && @test all(vl .< x.values[1:count] .<= vu)
            if job=='V' && count>0
                vectors=family==:evr ? x.z.data[1:n, 1:count] : x.a.data[:, 1:count]
                @test original*vectors≈vectors*Diagonal(x.values[1:count])
                @test vectors'*vectors≈Matrix{T}(I, count, count)
                if family==:evr && range=='A'
                    @test all(1 .<= x.support[1:2count] .<= n)
                end
            end
            direct_selfadjoint_guards(x)
        end
    end
end
