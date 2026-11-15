using Test, LinearAlgebra, Libdl, Colmat
if !isdefined(@__MODULE__, :direct_tgexc!)
    include("qz_reorder.jl")
end

@testset "TGEXC conjugate-block positions match LAPACK" begin
    external=only(
        filter(
            l->l.interface==:ilp64 && occursin("openblas", lowercase(basename(l.libname))),
            BLAS.get_config().loaded_libs,
        ),
    )
    reference=Libdl.dlopen(external.libname)
    for (T, prefix) in ((Float32, "s"), (Float64, "d")),
        J in (Int32, Int64),
        n1 in (1, 2),
        n2 in (1, 2),
        offset1 in 0:(n1 - 1),
        offset2 in 0:(n2 - 1),
        upward in (false, true)

        tol=T===Float32 ? 1e-3 : 1e-10
        n=n1+n2+2
        j1=2
        j2=j1+n1
        a=Matrix(Diagonal(T.(1:n)))
        b=Matrix{T}(I, n, n)
        for (k, width, center) in ((j1, n1, -2), (j2, n2, 3))
            a[k, k]=center
            if width==2
                a[k + 1, k + 1]=center
                a[k + 1, k]=1
                a[k, k + 1]=-1
            end
        end
        from, to=upward ? (j2+offset2, j1+offset1) : (j1+offset1, j2+offset2)
        q=Matrix{T}(I, n, n)
        z=copy(q)
        olda=copy(a)
        oldb=copy(b)
        ar=copy(a)
        br=copy(b)
        qref=copy(q)
        zref=copy(z)
        result=direct_tgexc!(a, b, q, z, J; from, to)
        oracle=direct_tgexc!(
            ar,
            br,
            qref,
            zref,
            Int64;
            from,
            to,
            pointer=Libdl.dlsym(reference, Symbol(prefix*"tgexc_64_")),
        )
        @test result.info==result.diagnostic==oracle.info==0
        @test (result.first, result.last)==(oracle.first, oracle.last)
        @test q*a*z'≈olda rtol=tol atol=tol
        @test q*b*z'≈oldb rtol=tol atol=tol
        @test q'*q≈Matrix{T}(I, n, n) rtol=tol atol=tol
        @test z'*z≈Matrix{T}(I, n, n) rtol=tol atol=tol
        width=upward ? n2 : n1
        rows=result.last:(result.last + width - 1)
        moved=sort(eigvals(a[rows, rows], b[rows, rows]); by=imag)
        expected=sort(eigvals(ar[rows, rows], br[rows, rows]); by=imag)
        @test moved≈expected rtol=tol atol=tol
    end
end
