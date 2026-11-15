# Run with: julia --startup-file=no --project reproducers/openblas-tprfb-workspace.jl
# OpenBLAS_jll 0.3.30 / Reference-LAPACK 3.12.1: ZTPRFB, SIDE=L,
# DIRECT=F, STOREV=R uses LDB instead of LDWORK in an internal ZTRMM.
# Distinct legal strides produce an incorrect result and writes past the
# declared scratch extent. Physical over-allocation keeps this reproducer safe.
# A factor from ZTPLQT2 supplies a valid block reflector, and the dense
# H = I - W*T*W^H formula supplies an independent expected result.
# Expected here: LDWORK=3/LDB=7 -> relative error ~0.55, two scratch
# writes past its declared extent; LDWORK=LDB=7 -> error ~2e-16, zero writes.
# Source: https://github.com/Reference-LAPACK/lapack/blob/v3.12.1/SRC/ztprfb.f

using LinearAlgebra, Libdl
reference=only(
    filter(
        l->l.interface==:ilp64&&occursin("openblas", lowercase(basename(l.libname))),
        BLAS.get_config().loaded_libs,
    ),
).libname
println("Julia ", VERSION, "; ", Sys.MACHINE, "; ", reference)
handle=Libdl.dlopen(reference)
k, tail, n, l=3, 4, 3, 2
ldv, ldt, lda, ldb=k+2, k+2, k+2, tail+3
factor_a=fill(ComplexF64(NaN), k+2, k)
v=fill(ComplexF64(NaN), ldv, tail)
for j in 1:k, i in j:k
    factor_a[i, j]=complex((i+2j)/10, (i-j)/20)
end
for j in 1:tail, i in 1:k
    if j<=tail-l||i>=j-(tail-l)
        v[i, j]=complex((3i+j)/17, (i-2j)/23)
    end
end
t=fill(ComplexF64(777), ldt, k);
info=Ref{Int64}(-1)
ccall(
    Libdl.dlsym(handle, :ztplqt2_64_),
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
        Ref{Int64},
        Ref{Int64},
    ),
    k,
    tail,
    l,
    factor_a,
    k+2,
    v,
    ldv,
    t,
    ldt,
    info,
)
@assert info[]==0
logical_v=copy(v[1:k, :])
for j in 1:tail, i in 1:k
    if j>tail-l&&i<j-(tail-l)
        logical_v[i, j]=0
    end
end
w=vcat(Matrix{ComplexF64}(I, k, k), Matrix(logical_v'))
h=Matrix{ComplexF64}(I, k+tail, k+tail)-w*triu(t[1:k, :])*w'
@assert h'*h ≈ Matrix{ComplexF64}(I, k+tail, k+tail)
a=fill(ComplexF64(777), lda, n);
b=fill(ComplexF64(777), ldb, n)
for j in 1:n, i in 1:k
    a[i, j]=complex((i+j)/7, (2i-j)/19)
end
for j in 1:n, i in 1:tail
    b[i, j]=complex((2i+j)/11, (i-3j)/17)
end
expected=h*vcat(a[1:k, :], b[1:tail, :])
for ldwork in (k, ldb)
    output_a=copy(a)
    output_b=copy(b)
    # Deliberately overallocate physically so a write outside the declared
    # workspace is observable without corrupting the Julia allocator.
    work=fill(ComplexF64(777), max(ldb, k)*n+8)
    ccall(
        Libdl.dlsym(handle, :ztprfb_64_),
        Cvoid,
        (
            Ref{UInt8},
            Ref{UInt8},
            Ref{UInt8},
            Ref{UInt8},
            Ref{Int64},
            Ref{Int64},
            Ref{Int64},
            Ref{Int64},
            Ptr{ComplexF64},
            Ref{Int64},
            Ptr{ComplexF64},
            Ref{Int64},
            Ptr{ComplexF64},
            Ref{Int64},
            Ptr{ComplexF64},
            Ref{Int64},
            Ptr{ComplexF64},
            Ref{Int64},
        ),
        'L',
        'N',
        'F',
        'R',
        tail,
        n,
        k,
        l,
        v,
        ldv,
        t,
        ldt,
        output_a,
        lda,
        output_b,
        ldb,
        work,
        ldwork,
    )
    actual=vcat(output_a[1:k, :], output_b[1:tail, :])
    outside=count(!=(ComplexF64(777)), work[(ldwork * n + 1):end])
    println(
        "LDWORK=",
        ldwork,
        " LDB=",
        ldb,
        " relative error=",
        norm(actual-expected)/norm(expected),
        " writes beyond declared workspace=",
        outside,
    )
end
