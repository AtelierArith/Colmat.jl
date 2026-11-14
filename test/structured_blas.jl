using Test, LinearAlgebra, Libdl, Colmat

for (T, p) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    J in (Int32, Int64)

    suffix=J===Int64 ? "64_" : ""
    @eval function direct_gbmv!(
        a::Matrix{$T},
        x::Vector{$T},
        y::Vector{$T},
        ::Type{$J};
        tr,
        m,
        n,
        kl,
        ku,
        alpha,
        beta,
        ix,
        iy,
        library=Colmat.backend_library(),
    )
        f=Libdl.dlsym(Libdl.dlopen(library), Symbol($p*"gbmv_"*$suffix))
        ccall(
            f,
            Cvoid,
            (
                Ref{UInt8},
                Ref{$J},
                Ref{$J},
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
            tr,
            m,
            n,
            kl,
            ku,
            alpha,
            a,
            size(a, 1),
            pointer(x, 2),
            ix,
            beta,
            pointer(y, 2),
            iy,
        )
    end
end
function structured_reference()
    return only(
        filter(
            l->l.interface==:ilp64 && occursin("openblas", lowercase(basename(l.libname))),
            BLAS.get_config().loaded_libs,
        ),
    ).libname
end
function structured_indices(n, inc)
    return if n==0
        Int[]
    else
        collect(inc>0 ? (2:inc:(2 + (n - 1) * inc)) : ((2 + (n - 1) * abs(inc)):inc:2))
    end
end

@testset "GBMV native band intervals transpose and signed strides" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        (m, n) in ((0, 3), (3, 0), (1, 1), (3, 5), (5, 3), (6, 6)),
        (kl, ku) in ((0, 0), (1, 2), (3, 1), (6, 7)),
        tr in ('N', 'T', 'C'),
        ix in (-3, -1, 1, 2),
        iy in (-2, -1, 1, 3),
        coefficients in 1:3

        R=typeof(real(zero(T)))
        alpha=if coefficients==1
            zero(T)
        elseif T <: Complex
            T(0.5, -1)
        else
            T(0.5)
        end
        beta=if coefficients==2
            zero(T)
        elseif T <: Complex
            T(-0.25, 0.5)
        else
            T(-0.25)
        end
        dense=zeros(T, m, n)
        a=fill(T(NaN), kl+ku+3, n+1)
        for j in 1:n, i in 1:m
            if j-ku<=i<=j+kl
                value=T <: Complex ? T((i+3j)/8, (i-j)/4) : T((i+3j)/8)
                dense[i, j]=value
                a[ku + 1 + i - j, j]=alpha==0 ? T(NaN) : value
            end
        end
        nx, ny=tr=='N' ? (n, m) : (m, n)
        xi=structured_indices(nx, ix)
        yi=structured_indices(ny, iy)
        x=fill(T(NaN), max(1, (nx-1)*abs(ix)+1)+2)
        y=fill(T(777), max(1, (ny-1)*abs(iy)+1)+2)
        for (k, index) in enumerate(xi)
            x[index]=if alpha==0
                T(NaN)
            elseif T <: Complex
                T(k/4, (k+1)/8)
            else
                T(k/4)
            end
        end
        for (k, index) in enumerate(yi)
            y[index]=if beta==0
                T(NaN)
            elseif T <: Complex
                T(k/8, -k/4)
            else
                T(k/8)
            end
        end
        beforea=copy(a)
        beforex=copy(x)
        beforey=copy(y)
        expected=copy(y)
        if m>0&&n>0
            op=if tr=='N'
                dense
            elseif tr=='T'
                transpose(dense)
            else
                adjoint(dense)
            end
            for k in 1:ny
                product=alpha==0 ? zero(T) : alpha*sum(op[k, l]*x[xi[l]] for l in 1:nx)
                expected[yi[k]]=product+(beta==0 ? zero(T) : beta*y[yi[k]])
            end
        end
        direct_gbmv!(a, x, y, J; tr, m, n, kl, ku, alpha, beta, ix, iy)
        @test isequal(a, beforea)
        @test isequal(x, beforex)
        @test isapprox(y[yi], expected[yi]; rtol=50eps(R), atol=50eps(R), nans=true)
        ignored=setdiff(eachindex(y), yi)
        @test isequal(y[ignored], beforey[ignored])
        if J===Int64
            external=copy(beforey)
            direct_gbmv!(
                a,
                x,
                external,
                J;
                tr,
                m,
                n,
                kl,
                ku,
                alpha,
                beta,
                ix,
                iy,
                library=structured_reference(),
            )
            @test isapprox(y[yi], external[yi]; rtol=50eps(R), atol=50eps(R), nans=true)
            @test isequal(external[ignored], beforey[ignored])
        end
    end
end

@testset "GBMV public Julia and both numerical LBT interfaces" begin
    for (T, p) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
        J in (Int32, Int64)

        m, n, kl, ku=3, 4, 1, 1
        a=fill(T(NaN), 3, 4)
        dense=zeros(T, m, n)
        for j in 1:n, i in 1:m
            if j-ku<=i<=j+kl
                dense[i, j]=T <: Complex ? T(i+j, i-j) : T(i+j)
                a[ku + 1 + i - j, j]=dense[i, j]
            end
        end
        logical_x=T.(1:n)
        x=fill(T(777), n+2)
        x[2:(n + 1)]=reverse(logical_x)
        y=fill(T(777), m+2)
        backing=BLAS.lbt_find_backing_library(p*"gbmv_", J===Int32 ? :lp64 : :ilp64)
        @test backing!==nothing
        @test realpath(backing.libname)==realpath(Colmat.backend_library())
        direct_gbmv!(
            a,
            x,
            y,
            J;
            tr=('N'),
            m,
            n,
            kl,
            ku,
            alpha=one(T),
            beta=zero(T),
            ix=-1,
            iy=1,
            library=BLAS.libblastrampoline,
        )
        expected=T[sum(dense[i, j]*logical_x[j] for j in 1:n) for i in 1:m]
        @test y[2:(m + 1)]==expected
        @test y[1]==y[end]==777
        public_y=zeros(T, m)
        LinearAlgebra.BLAS.gbmv!('N', m, kl, ku, one(T), a, logical_x, zero(T), public_y)
        @test public_y==expected
    end
end
