using Test, LinearAlgebra, Libdl
mode=only(ARGS[1:1])
lp64_reference=nothing
lp64_original64=Dict{String,Ptr{Cvoid}}()
if mode=="native_lp64"
    @eval using OpenBLAS32_jll
    lp64_reference=OpenBLAS32_jll.libopenblas_path
    BLAS.lbt_forward(lp64_reference; clear=false)
    for prefix in ("s", "d", "c", "z")
        provider=BLAS.lbt_find_backing_library(prefix*"geevx_", :lp64)
        @assert provider!==nothing && realpath(provider.libname)==realpath(lp64_reference)
        lp64_original64[prefix]=BLAS.lbt_get_forward(prefix*"geevx_", :ilp64)
    end
end
if mode=="first_missing"
    prefix=ARGS[2]
    id=findfirst(==(prefix), ("s", "d", "c", "z"))-1
    symbol=prefix*"geevx_"
    BLAS.lbt_set_forward(symbol, C_NULL, :ilp64)
    before=Dict(
        (name, abi)=>BLAS.lbt_get_forward(name, abi) for
        name in ("dgemm_", "sgemm_", "sgeevx_", "dgeevx_", "cgeevx_", "zgeevx_") for
        abi in (:lp64, :ilp64)
    )
    @testset "GEEVX first registration rejects a missing $prefix ILP64 implementation" begin
        @test BLAS.lbt_find_backing_library(symbol, :ilp64)===nothing
        err=try
            @eval using Colmat
            nothing
        catch e
            e
        end
        @test err isa InitError
        @test occursin(
            "GEEVX requires an existing ILP64 expert backend", sprint(showerror, err)
        )
        @test all(BLAS.lbt_get_forward(name, abi)==p for ((name, abi), p) in before)
        library=joinpath(
            @__DIR__, "..", "deps", "usr", "lib", "libcolmat_blas."*Libdl.dlext
        )
        getter=Libdl.dlsym(Libdl.dlopen(library), :colmat_lapack_has_expert_fallback)
        @test ccall(getter, Cint, (Cint, Cint), id, 64)==0
    end
    exit(0)
end
using Colmat
include("geevx_expert_helpers.jl")
Random.seed!(3163)
handle=Libdl.dlopen(Colmat.backend_library())
setter=Libdl.dlsym(handle, :colmat_lapack_set_expert_fallback)
getter=Libdl.dlsym(handle, :colmat_lapack_has_expert_fallback)
reference=geevx_reference()
if mode=="bridge"
    @testset "GEEVX fresh process forces LP64 expert calls through ILP64" begin
        for (id, (T, prefix)) in enumerate((
            (Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")
        ))
            ccall(setter, Cvoid, (Cint, Cint, Ptr{Cvoid}), id-1, 32, C_NULL)
            @test ccall(getter, Cint, (Cint, Cint), id-1, 32)==1
            @test ccall(getter, Cint, (Cint, Cint), id-1, 64)==1
            @test Colmat.load_backend()===nothing
            # Neither integer ABI may be saved as its own original backend.
            for suffix in ("", "64_")
                own=Libdl.dlsym(handle, Symbol(prefix*"geevx_"*suffix))
                ccall(setter, Cvoid, (Cint, Cint, Ptr{Cvoid}), id-1, 64, own)
            end
            for J in (Int32, Int64),
                n in (0, 1, 4), sense in ('N', 'E', 'V', 'B', '?'),
                query in (false, true)

                original=geevx_fixture(T, n)
                a=vcat(original, fill(T(777), 2, n))
                pointer=BLAS.lbt_get_forward(prefix*"geevx_", J===Int32 ? :lp64 : :ilp64)
                result=direct_geevx!(
                    a,
                    J;
                    n,
                    bal=('B'),
                    sense,
                    lwork=query ? -1 : geevx_minimum(T, n, 'V', 'V', sense),
                    pointer,
                )
                if sense=='?'
                    @test result.info == -4 && result.diagnostic==4
                else
                    @test result.info==result.diagnostic==0
                    if !query
                        w=if T<:Real
                            complex.(result.values, result.imaginary)
                        else
                            result.values
                        end
                        x=geevx_unpack(result.right, result.imaginary)
                        tol=2000eps(real(T))*max(1, n)
                        @test norm(original*x-x*Diagonal(w))<=tol*max(
                            1, norm(original)*norm(x)
                        )
                        @test all(
                            isapprox(norm(col), 1; atol=tol, rtol=tol) for col in eachcol(x)
                        )
                        if sense!='N'
                            refa=vcat(original, fill(T(777), 2, n))
                            oracle=direct_geevx!(
                                refa, Int64; n, bal=('B'), sense, library=reference
                            )
                            @test isequal(result.rconde, oracle.rconde) &&
                                isequal(result.rcondv, oracle.rcondv)
                        end
                    end
                end
            end
        end
    end
elseif mode=="native_lp64"
    @testset "GEEVX existing LP64 backend remains usable without an ILP64 expert slot" begin
        for (id, (T, prefix)) in enumerate((
            (Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")
        ))
            ccall(setter, Cvoid, (Cint, Cint, Ptr{Cvoid}), id-1, 64, C_NULL)
            @test ccall(getter, Cint, (Cint, Cint), id-1, 32)==1
            @test ccall(getter, Cint, (Cint, Cint), id-1, 64)==0
            absent=direct_geevx!(
                vcat(geevx_fixture(T, 1), fill(T(777), 2, 1)), Int64; sense=('E'), lwork=-1
            )
            @test absent.info==-777 && absent.diagnostic == -1
            for n in (0, 1, 4), sense in ('N', 'E', 'V', 'B', '?'), query in (false, true)
                original=geevx_fixture(T, n)
                a=vcat(original, fill(T(777), 2, n))
                refa=copy(a)
                pointer=BLAS.lbt_get_forward(prefix*"geevx_", :lp64)
                lw=query ? -1 : geevx_minimum(T, n, 'V', 'V', sense)
                result=direct_geevx!(a, Int32; n, bal=('B'), sense, lwork=lw, pointer)
                oracle=direct_geevx!(
                    refa, Int32; n, bal=('B'), sense, lwork=lw, library=lp64_reference
                )
                if sense=='?'
                    @test result.info==oracle.info==-4 && result.diagnostic==4
                else
                    @test result.info==oracle.info==result.diagnostic==0
                    if !query
                        w=if T<:Real
                            complex.(result.values, result.imaginary)
                        else
                            result.values
                        end
                        x=geevx_unpack(result.right, result.imaginary)
                        tol=2000eps(real(T))*max(1, n)
                        @test norm(original*x-x*Diagonal(w))<=tol*max(
                            1, norm(original)*norm(x)
                        )
                        @test all(
                            isapprox(norm(col), 1; atol=tol, rtol=tol) for col in eachcol(x)
                        )
                        if sense!='N'
                            @test isequal(result.rconde, oracle.rconde) &&
                                isequal(result.rcondv, oracle.rcondv)
                            @test isequal(result.values, oracle.values) &&
                                isequal(result.scale, oracle.scale)
                        end
                    end
                end
            end
            ccall(
                setter, Cvoid, (Cint, Cint, Ptr{Cvoid}), id-1, 64, lp64_original64[prefix]
            )
            @test Colmat.load_backend()===nothing
        end
    end
elseif mode=="missing_after"
    @testset "GEEVX missing saved backend reports configuration before any re-forwarding" begin
        for (id, (T, prefix)) in enumerate((
            (Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")
        ))
            for bits in (32, 64)
                ccall(setter, Cvoid, (Cint, Cint, Ptr{Cvoid}), id-1, bits, C_NULL)
            end
            for bits in (32, 64)
                @test ccall(getter, Cint, (Cint, Cint), id-1, bits)==0
            end
            before=Dict(
                (name, abi)=>BLAS.lbt_get_forward(name, abi) for
                name in ("dgemm_", "sgeevx_", "dgeevx_", "cgeevx_", "zgeevx_") for
                abi in (:lp64, :ilp64)
            )
            err=try
                Colmat.load_backend()
                nothing
            catch e
                e
            end
            @test err isa ErrorException
            @test occursin(
                "GEEVX requires an existing ILP64 expert backend", sprint(showerror, err)
            )
            @test all(BLAS.lbt_get_forward(name, abi)==p for ((name, abi), p) in before)
            for J in (Int32, Int64),
                sense in ('N', 'E', 'V', 'B', '?'),
                query in (false, true)

                initial=geevx_fixture(T, 3)
                a=vcat(initial, fill(T(777), 2, 3))
                saved=copy(a)
                result=direct_geevx!(
                    a, J; sense, lwork=query ? -1 : geevx_minimum(T, 3, 'V', 'V', sense)
                )
                if sense=='N'
                    @test result.info==result.diagnostic==0
                elseif sense=='?'
                    @test result.info == -4 && result.diagnostic==4
                else
                    # Uninitialized low-level entry: no invented LAPACK INFO.
                    @test result.info==-777 && result.diagnostic == -1
                    @test isequal(a, saved) && result.workspace==777
                    @test result.ilo==result.ihi==-777
                    @test all(==(T(777)), result.values) &&
                        all(==(T(777)), result.left) &&
                        all(==(T(777)), result.right)
                end
            end
            external=Libdl.dlsym(Libdl.dlopen(reference), Symbol(prefix*"geevx_64_"))
            ccall(setter, Cvoid, (Cint, Cint, Ptr{Cvoid}), id-1, 64, external)
            @test Colmat.load_backend()===nothing
            @test direct_geevx!(
                vcat(geevx_fixture(T, 1), fill(T(777), 2, 1)), Int32; sense=('E'), lwork=-1
            ).info==0
        end
    end
else
    error("Unknown GEEVX worker mode: $mode")
end
