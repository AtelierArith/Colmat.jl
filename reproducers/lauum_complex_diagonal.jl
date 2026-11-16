using Colmat, LinearAlgebra, Libdl
reference=only(
    filter(
        l->l.interface==:ilp64&&occursin("openblas", lowercase(l.libname)),
        BLAS.get_config().loaded_libs,
    ),
).libname
for n in (1, 3),
    u in ('U', 'L'), family in ("lauum", "lauu2"),
    library in (Colmat.backend_library(), reference)

    a=fill(ComplexF64(NaN), n, n)
    for i in 1:n, j in 1:n
        if u=='U' ? i<=j : i>=j
            a[i, j]=i==j ? complex(i+1, 3) : 0.5+0.2im
        end
    end
    info=Ref{Int64}(-99)
    p=Libdl.dlsym(Libdl.dlopen(library), "z"*family*"_64_")
    ccall(
        p,
        Cvoid,
        (Ref{UInt8}, Ref{Int64}, Ptr{ComplexF64}, Ref{Int64}, Ptr{Int64}, Clong),
        u,
        n,
        a,
        n,
        info,
        1,
    )
    println((; n, uplo=u, family, library=basename(library), info=info[], diagonal=diag(a)))
end
