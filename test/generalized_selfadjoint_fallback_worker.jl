using Test, LinearAlgebra, Libdl, Colmat

@testset "Generalized self-adjoint backend requirement and native dispatch" begin
    handle=Libdl.dlopen(Colmat.backend_library())
    setter=Libdl.dlsym(handle, :colmat_fallback_set)
    group=Cint(Colmat._fallback_group("generalized self-adjoint EVD").group)
    before=BLAS.lbt_get_forward("ssygv_", :ilp64)
    ccall(setter, Cvoid, (Cint, Cint, Ptr{Cvoid}), group, 0, C_NULL)
    err=try
        Colmat.load_backend()
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin(
        "generalized self-adjoint EVD requires an existing ILP64 LAPACK backend",
        sprint(showerror, err),
    )
    @test BLAS.lbt_get_forward("ssygv_", :ilp64)==before
    external=only(
        filter(
            l->l.interface==:ilp64&&occursin("openblas", lowercase(basename(l.libname))),
            BLAS.get_config().loaded_libs,
        ),
    )
    pointer=Libdl.dlsym(Libdl.dlopen(external.libname), :ssygv_64_)
    ccall(setter, Cvoid, (Cint, Cint, Ptr{Cvoid}), group, 0, pointer)
    @test Colmat.load_backend()===nothing
    for id in 0:11
        ccall(setter, Cvoid, (Cint, Cint, Ptr{Cvoid}), group, id, C_NULL)
    end
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        a=T[2 0.125 0; 0.125 3 0.25; 0 0.25 4]
        b=Matrix(Diagonal(T[2, 3, 4]))
        result=eigen(Hermitian(a), Hermitian(b))
        @test a*result.vectors ≈ b*result.vectors*Diagonal(result.values)
        @test result.vectors'*b*result.vectors ≈ Matrix{T}(I, 3, 3)
    end
end
