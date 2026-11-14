using Test, LinearAlgebra, Libdl, Colmat

@testset "GGEVX requires an expert backend before registration" begin
    handle=Libdl.dlopen(Colmat.backend_library())
    setter=Libdl.dlsym(handle, :colmat_ggevx_set_expert_fallback)
    before=BLAS.lbt_get_forward("sggevx_", :ilp64)
    # Inject a missing saved backend in an isolated process. Re-registration must
    # diagnose the configuration before changing the LBT table.
    ccall(setter, Cvoid, (Cint, Ptr{Cvoid}), 0, C_NULL)
    err=try
        Colmat.load_backend()
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("existing ILP64 expert backend", sprint(showerror, err))
    @test BLAS.lbt_get_forward("sggevx_", :ilp64)==before
    external=only(
        filter(
            l->l.interface==:ilp64 && occursin("openblas", lowercase(basename(l.libname))),
            BLAS.get_config().loaded_libs,
        ),
    )
    pointer=Libdl.dlsym(Libdl.dlopen(external.libname), :sggevx_64_)
    ccall(setter, Cvoid, (Cint, Ptr{Cvoid}), 0, pointer)
    @test Colmat.load_backend()===nothing
end
