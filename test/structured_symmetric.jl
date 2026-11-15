using Test, LinearAlgebra, Libdl, Colmat

symmetric_blas_pointer(a::Matrix) = a
symmetric_blas_pointer(a::Vector) = view(a, 2:(length(a) - 1))
for (T, p) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    J in (Int32, Int64)

    suffix=J===Int64 ? "64_" : ""
    @eval function direct_symmetric_blas!(
        family,
        a::Array{$T},
        x::Vector{$T},
        y::Vector{$T},
        ::Type{$J};
        n,
        k=0,
        u,
        alpha,
        beta,
        ix,
        iy,
        library=Colmat.backend_library(),
    )
        f=Libdl.dlsym(Libdl.dlopen(library), Symbol($p*family*"_"*$suffix))
        source=symmetric_blas_pointer(a)
        if family in ("sbmv", "hbmv")
            ccall(
                f,
                Cvoid,
                (
                    Ref{UInt8},
                    Ref{$J},
                    Ref{$J},
                    Ref{$T},
                    Ptr{$T},
                    Ref{$J},
                    Ptr{$T},
                    Ref{$J},
                    Ref{$T},
                    Ptr{$T},
                    Ref{$J},
                ),
                u,
                n,
                k,
                alpha,
                source,
                size(a, 1),
                pointer(x, 2),
                ix,
                beta,
                pointer(y, 2),
                iy,
            )
        else
            ccall(
                f,
                Cvoid,
                (
                    Ref{UInt8},
                    Ref{$J},
                    Ref{$T},
                    Ptr{$T},
                    Ptr{$T},
                    Ref{$J},
                    Ref{$T},
                    Ptr{$T},
                    Ref{$J},
                ),
                u,
                n,
                alpha,
                source,
                pointer(x, 2),
                ix,
                beta,
                pointer(y, 2),
                iy,
            )
        end
    end
end
function symmetric_blas_reference()
    return only(
        filter(
            l->l.interface==:ilp64 && occursin("openblas", lowercase(basename(l.libname))),
            BLAS.get_config().loaded_libs,
        ),
    ).libname
end
function symmetric_blas_indices(n, inc)
    return if n==0
        Int[]
    else
        collect(inc>0 ? (2:inc:(2 + (n - 1) * inc)) : ((2 + (n - 1) * abs(inc)):inc:2))
    end
end
function symmetric_blas_fixture(T, family, n, k, u)
    band=family in ("sbmv", "hbmv")
    hermitian=family in ("hbmv", "hpmv")
    a=band ? fill(T(NaN), k+3, n+1) : fill(T(NaN), n*(n+1)÷2+2)
    if !band
        a[1]=a[end]=T(777)
    end
    dense=zeros(T, n, n)
    cursor=2
    for j in 1:n
        rows=u=='U' ? (1:j) : (j:n)
        for i in rows
            if band && abs(i-j)>k
                continue
            end
            value=T <: Complex ? T((i+3j)/8, (i-j+1)/4) : T((i+3j)/8)
            dense[i, j]=hermitian && i==j ? T(real(value), 0) : value
            dense[j, i]=hermitian ? conj(dense[i, j]) : dense[i, j]
            stored=hermitian && i==j ? T(real(value), NaN) : value
            if band
                a[u=='U' ? k+1+i-j : 1+i-j, j]=stored
            else
                a[cursor]=stored
                cursor+=1
            end
        end
    end
    return a, dense
end

@testset "Symmetric/Hermitian band and packed native vector products" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        family in (T <: Complex ? ("sbmv", "spmv", "hbmv", "hpmv") : ("sbmv", "spmv")),
        n in 0:8,
        k in (family in ("sbmv", "hbmv") ? (0, 1, 4, 10) : (0,)),
        u in ('U', 'L'),
        ix in (-3, -1, 1, 2),
        iy in (-2, -1, 1, 3),
        coefficients in 1:4

        R=typeof(real(zero(T)))
        alpha=if coefficients<=2
            zero(T)
        elseif T <: Complex
            T(0.5, -1)
        else
            T(0.5)
        end
        beta=if coefficients==1
            one(T)
        elseif coefficients<=3
            zero(T)
        elseif T <: Complex
            T(-0.25, 0.5)
        else
            T(-0.25)
        end
        a, dense=symmetric_blas_fixture(T, family, n, k, u)
        if alpha==0
            if a isa Matrix
                a.=T(NaN)
            else
                a[2:(end - 1)].=T(NaN)
            end
        end
        xi=symmetric_blas_indices(n, ix)
        yi=symmetric_blas_indices(n, iy)
        x=fill(T(NaN), max(1, (n-1)*abs(ix)+1)+2)
        y=fill(T(777), max(1, (n-1)*abs(iy)+1)+2)
        for (i, idx) in enumerate(xi)
            x[idx]=if alpha==0
                T(NaN)
            elseif T <: Complex
                T(i/4, -(i+1)/8)
            else
                T(i/4)
            end
        end
        for (i, idx) in enumerate(yi)
            y[idx]=if beta==0
                T(NaN)
            elseif T <: Complex
                T(i/8, -i/4)
            else
                T(i/8)
            end
        end
        beforea=copy(a)
        beforex=copy(x)
        beforey=copy(y)
        expected=copy(y)
        for i in 1:n
            product=alpha==0 ? zero(T) : alpha*sum(dense[i, j]*x[xi[j]] for j in 1:n)
            expected[yi[i]]=product+(beta==0 ? zero(T) : beta*y[yi[i]])
        end
        direct_symmetric_blas!(family, a, x, y, J; n, k, u, alpha, beta, ix, iy)
        @test isequal(a, beforea)
        @test isequal(x, beforex)
        @test isapprox(y[yi], expected[yi]; rtol=100eps(R), atol=100eps(R), nans=true)
        ignored=setdiff(eachindex(y), yi)
        @test isequal(y[ignored], beforey[ignored])
        if n==0 || (alpha==0&&beta==1)
            @test isequal(y, beforey)
        end
        if J===Int64
            external=copy(beforey)
            direct_symmetric_blas!(
                family,
                a,
                x,
                external,
                J;
                n,
                k,
                u,
                alpha,
                beta,
                ix,
                iy,
                library=symmetric_blas_reference(),
            )
            @test isapprox(y[yi], external[yi]; rtol=100eps(R), atol=100eps(R), nans=true)
            @test isequal(external[ignored], beforey[ignored])
        end
    end
end

@testset "All 12 symmetric band/packed symbols numerically through LBT" begin
    for (T, p) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
        J in (Int32, Int64),
        family in (T <: Complex ? ("sbmv", "spmv", "hbmv", "hpmv") : ("sbmv", "spmv"))

        a, dense=symmetric_blas_fixture(T, family, 4, 1, 'L')
        x=T[777, 1, 2, 3, 4, 777]
        y=fill(T(777), 6)
        backing=BLAS.lbt_find_backing_library(p*family*"_", J===Int32 ? :lp64 : :ilp64)
        @test backing!==nothing
        @test realpath(backing.libname)==realpath(Colmat.backend_library())
        direct_symmetric_blas!(
            family,
            a,
            x,
            y,
            J;
            n=4,
            k=1,
            u=('L'),
            alpha=one(T),
            beta=zero(T),
            ix=1,
            iy=1,
            library=BLAS.libblastrampoline,
        )
        expected=T[sum(dense[i, j]*x[j + 1] for j in 1:4) for i in 1:4]
        @test y[2:5]==expected
        @test y[1]==y[end]==777
    end
end
