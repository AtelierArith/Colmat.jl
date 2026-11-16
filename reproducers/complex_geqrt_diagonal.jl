using Colmat, LinearAlgebra, Libdl
reference=only(
    filter(
        l->l.interface==:ilp64&&occursin("openblas", lowercase(l.libname)),
        BLAS.get_config().loaded_libs,
    ),
).libname
for library in (Colmat.backend_library(), reference)
    p=Libdl.dlsym(Libdl.dlopen(library), :zgeqrt_64_)
    m=Int64(5)
    n=Int64(3)
    nb=Int64(2)
    a=ComplexF64[(i==j ? 2 : 0)+0.1*(i+j-1)+0.07*(i-j)*im for i in 1:m, j in 1:n]
    t=zeros(ComplexF64, nb, n)
    work=zeros(ComplexF64, nb*n)
    info=Ref{Int64}(-99)
    ccall(
        p,
        Cvoid,
        (
            Ref{Int64},
            Ref{Int64},
            Ref{Int64},
            Ptr{ComplexF64},
            Ref{Int64},
            Ptr{ComplexF64},
            Ref{Int64},
            Ptr{ComplexF64},
            Ptr{Int64},
        ),
        m,
        n,
        nb,
        a,
        m,
        t,
        nb,
        work,
        info,
    )
    println((library=basename(library), info=info[], diagonal=diag(a)))
end
