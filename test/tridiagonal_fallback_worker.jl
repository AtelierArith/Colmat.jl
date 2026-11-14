using Test, LinearAlgebra, Libdl, Colmat

@testset "Tridiagonal backend requirement and native dispatch" begin
    handle=Libdl.dlopen(Colmat.backend_library())
    setter=Libdl.dlsym(handle, :colmat_tridiagonal_set_fallback)
    before=BLAS.lbt_get_forward("ssteqr_", :ilp64)
    ccall(setter, Cvoid, (Cint, Ptr{Cvoid}), 0, C_NULL)
    err=try
        Colmat.load_backend()
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("existing ILP64 LAPACK backend", sprint(showerror, err))
    @test BLAS.lbt_get_forward("ssteqr_", :ilp64)==before
    external=only(
        filter(
            l->l.interface==:ilp64&&occursin("openblas", lowercase(basename(l.libname))),
            BLAS.get_config().loaded_libs,
        ),
    )
    pointer=Libdl.dlsym(Libdl.dlopen(external.libname), :ssteqr_64_)
    ccall(setter, Cvoid, (Cint, Ptr{Cvoid}), 0, pointer)
    @test Colmat.load_backend()===nothing

    # Isolated execution proves ordinary finite calls do not delegate. A wrong
    # dispatch aborts only this worker because Rust requires a configured pointer.
    for id in 0:23
        ccall(setter, Cvoid, (Cint, Ptr{Cvoid}), id, C_NULL)
    end
    for R in (Float32, Float64)
        d=R[2, 3, 4]
        e=R[0.2, 0.3]
        a=SymTridiagonal(d, e)
        w, z=LAPACK.stegr!('V', copy(d), copy(e))
        @test a*z ≈ z*Diagonal(w)
        w, z=LAPACK.stev!('V', copy(d), copy(e))
        @test a*z ≈ z*Diagonal(w)
    end
end
