using Colmat, LinearAlgebra, Libdl

reference=only(
    filter(
        l->l.interface==:ilp64 && occursin("openblas", lowercase(l.libname)),
        BLAS.get_config().loaded_libs,
    ),
).libname
for library in (Colmat.backend_library(), reference)
    f=Libdl.dlsym(Libdl.dlopen(library), :dgeevx_64_)
    scratch=zeros(16)
    ilo=Ref{Int64}(-777)
    ihi=Ref{Int64}(-777)
    abnrm=Ref(777.0)
    info=Ref{Int64}(-777)
    iw=zeros(Int64, 1)
    ccall(
        f,
        Cvoid,
        (
            Ref{UInt8},
            Ref{UInt8},
            Ref{UInt8},
            Ref{UInt8},
            Ref{Int64},
            Ptr{Float64},
            Ref{Int64},
            Ptr{Float64},
            Ptr{Float64},
            Ptr{Float64},
            Ref{Int64},
            Ptr{Float64},
            Ref{Int64},
            Ptr{Int64},
            Ptr{Int64},
            Ptr{Float64},
            Ptr{Float64},
            Ptr{Float64},
            Ptr{Float64},
            Ptr{Float64},
            Ref{Int64},
            Ptr{Int64},
            Ptr{Int64},
            Clong,
            Clong,
            Clong,
            Clong,
        ),
        'N',
        'N',
        'N',
        'N',
        0,
        scratch,
        1,
        scratch,
        scratch,
        scratch,
        1,
        scratch,
        1,
        ilo,
        ihi,
        scratch,
        abnrm,
        scratch,
        scratch,
        scratch,
        1,
        iw,
        info,
        1,
        1,
        1,
        1,
    )
    println(basename(library), ": ", (; info=info[], ilo=ilo[], ihi=ihi[], abnrm=abnrm[]))
end
