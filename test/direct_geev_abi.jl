using Test, LinearAlgebra, Libdl, Random, Colmat
isdefined(@__MODULE__, :factor_abi_matrix) || include("direct_factor_abi.jl")
isdefined(@__MODULE__, :geevx_unpack) || include("geevx_expert_helpers.jl")
for J in (Int32, Int64), T in (Float32, Float64, ComplexF32, ComplexF64)
    R=real(T)
    signature=Any[Ref{UInt8}, Ref{UInt8}, Ref{J}, Ptr{T}, Ref{J}, Ptr{T}]
    args=Any[:left, :right, :n, :(x.a.data), :(stride(x.a.data, 2)), :(x.w)]
    if T<:Real
        push!(signature, Ptr{R})
        push!(args, :(x.wi))
    end
    append!(signature, [Ptr{T}, Ref{J}, Ptr{T}, Ref{J}, Ptr{T}, Ref{J}])
    append!(
        args,
        [
            :(x.left.data),
            :(stride(x.left.data, 2)),
            :(x.right.data),
            :(stride(x.right.data, 2)),
            :(x.work),
            :lw,
        ],
    )
    if T<:Complex
        push!(signature, Ptr{R})
        push!(args, :(x.rwork))
    end
    append!(signature, [Ptr{J}, Clong, Clong])
    append!(args, [:(x.info), 1, 1])
    @eval function direct_geev_call(f, ::Type{$T}, ::Type{$J}, x, n, left, right, lw)
        ccall(f, Cvoid, ($(signature...),), $(args...))
    end
end
function direct_geev_fixture(T, J, original, left, right)
    n=size(original, 1)
    minimum=max(
        1,
        if T<:Complex
            2n
        elseif left=='V'||right=='V'
            4n
        else
            3n
        end,
    )
    a=factor_abi_matrix(original)
    l=factor_abi_matrix(zeros(T, max(1, n), max(1, n)))
    r=factor_abi_matrix(zeros(T, max(1, n), max(1, n)))
    wg, work=factor_abi_integer(T, minimum)
    vg, w=factor_abi_integer(T, max(1, n))
    ig, wi=factor_abi_integer(real(T), max(1, n))
    rg, rwork=factor_abi_integer(real(T), max(1, 2n))
    fg, info=factor_abi_integer(J)
    return (; a, left=l, right=r, wg, work, vg, w, ig, wi, rg, rwork, fg, info)
end
function direct_geev_guards(x)
    for a in (x.a, x.left, x.right)
        factor_abi_guards(a)
    end
    for v in (x.wg, x.vg, x.ig, x.rg, x.fg)
        factor_abi_integer_guards(v)
    end
end
@testset "Direct GEEV integer ABI contracts" begin
    library=Libdl.dlopen(Colmat.backend_library())
    reference_library=Libdl.dlopen(geevx_reference())
    for (T, prefix) in
        ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
        J in (Int32, Int64),
        route in (:dll, :lbt),
        n in (0, 1, 2, 4, 7),
        left in ('N', 'V'),
        right in ('N', 'V')

        @testset "$T/$J/$route/$n/$left/$right" begin
            symbol=prefix*"geev"
            f=if route==:dll
                Libdl.dlsym(library, symbol*(J==Int32 ? "_" : "_64_"))
            else
                Libdl.dlsym(
                    Libdl.dlopen(BLAS.libblastrampoline), symbol*(J==Int32 ? "_" : "_64_")
                )
            end
            reference=Libdl.dlsym(reference_library, symbol*"_64_")
            original=geevx_fixture(T, n)
            x=direct_geev_fixture(T, J, original, left, right)
            y=direct_geev_fixture(T, Int64, original, left, right)
            minimum=length(x.work)
            direct_geev_call(f, T, J, x, n, left, right, -1)
            @test x.info[1]==0 && real(x.work[1])>=minimum
            @test x.a.data==original
            direct_geev_guards(x)
            direct_geev_call(f, T, J, x, n, left, right, minimum-1)
            @test x.info[1]==-(T<:Complex ? 12 : 13) && x.a.data==original
            @test factor_abi_diagnostic() == -x.info[1]
            direct_geev_call(f, T, J, x, n, '?', right, minimum)
            @test x.info[1]==-1 && x.a.data==original
            @test factor_abi_diagnostic() == -x.info[1]
            direct_geev_call(f, T, J, x, n, left, '?', minimum)
            @test x.info[1]==-2 && x.a.data==original
            @test factor_abi_diagnostic() == -x.info[1]
            direct_geev_call(f, T, J, x, n, left, right, minimum)
            @test x.info[1]==0
            direct_geev_call(reference, T, Int64, y, n, left, right, length(y.work))
            @test y.info[1]==0
            values=T<:Real ? complex.(x.w[1:n], x.wi[1:n]) : x.w[1:n]
            expected=T<:Real ? complex.(y.w[1:n], y.wi[1:n]) : y.w[1:n]
            tol=2000eps(real(T))*max(1, n)
            @test sort(values; by=z->(real(z), imag(z)))≈sort(
                expected; by=z->(real(z), imag(z))
            ) atol=tol rtol=tol
            for (job, output, op, eigenvalues) in (
                (left, x.left.data, original', conj.(values)),
                (right, x.right.data, original, values),
            )
                if job=='V' && n>0
                    vectors=if T<:Real
                        geevx_unpack(output[1:n, 1:n], x.wi[1:n])
                    else
                        output[1:n, 1:n]
                    end
                    @test norm(op*vectors-vectors*Diagonal(eigenvalues))<=tol*max(
                        1, norm(original)*norm(vectors)
                    )
                    @test all(
                        isapprox(norm(v), 1; atol=tol, rtol=tol) for v in eachcol(vectors)
                    )
                end
            end
            direct_geev_guards(x)
        end
    end
end
