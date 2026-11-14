using Test, LinearAlgebra, Random, Libdl, Colmat

for (T, prefix) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    J in (Int32, Int64)

    suffix=J===Int64 ? "64_" : ""
    @eval function direct_triangular_product!(
        family,
        a::Matrix{$T},
        b::Matrix{$T},
        c::Matrix{$T},
        ::Type{$J};
        uplo=('L'),
        ta=('N'),
        tb=('N'),
        n=size(c, 2),
        k=ta=='N' ? size(a, 2) : size(a, 1),
        alpha=one($T),
        beta=zero($T),
        lda=size(a, 1),
        ldb=size(b, 1),
        ldc=size(c, 1),
        library=Colmat.backend_library(),
    )
        f=Libdl.dlsym(Libdl.dlopen(library), Symbol($prefix*family*"_"*$suffix))
        ccall(
            f,
            Cvoid,
            (
                Ref{UInt8},
                Ref{UInt8},
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
            uplo,
            ta,
            tb,
            n,
            k,
            alpha,
            a,
            lda,
            b,
            ldb,
            beta,
            c,
            ldc,
        )
        c
    end
    @eval function direct_rank2!(
        family,
        a::Matrix{$T},
        x::Vector{$T},
        y::Vector{$T},
        ::Type{$J};
        uplo=('L'),
        n=size(a, 2),
        alpha=one($T),
        incx=1,
        incy=1,
        lda=size(a, 1),
        library=Colmat.backend_library(),
    )
        f=Libdl.dlsym(Libdl.dlopen(library), Symbol($prefix*family*"_"*$suffix))
        xp=view(x, 2:(length(x) - 1))
        yp=view(y, 2:(length(y) - 1))
        ccall(
            f,
            Cvoid,
            (
                Ref{UInt8},
                Ref{$J},
                Ref{$T},
                Ptr{$T},
                Ref{$J},
                Ptr{$T},
                Ref{$J},
                Ptr{$T},
                Ref{$J},
            ),
            uplo,
            n,
            alpha,
            xp,
            incx,
            yp,
            incy,
            a,
            lda,
        )
        a
    end
end

function extension_storage(a)
    m, n=size(a)
    out=fill(eltype(a)(777), max(1, m+3), n)
    out[1:m, :]=a
    return out
end
extension_op(a, t) =
    if t=='N'
        a
    elseif t=='T'
        transpose(a)
    else
        adjoint(a)
    end
function extension_vector(x, inc)
    n=length(x)
    out=fill(eltype(x)(777), max(1, 1+(n-1)*abs(inc))+2)
    for i in 1:n
        out[2 + (inc < 0 ? n - i : i - 1) * abs(inc)]=x[i]
    end
    return out
end
function extension_last_error()
    f=Libdl.dlsym(Libdl.dlopen(Colmat.backend_library()), :colmat_blas_last_error)
    return ccall(f, Cint, ())
end

@testset "GEMMT/GEMMTR selected triangle, four scalars and integer ABIs" begin
    rng=MersenneTwister(276)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        family in ("gemmt", "gemmtr"),
        uplo in ('L', 'U'),
        ta in ('N', 'T', 'C'),
        tb in ('N', 'T', 'C'),
        (n, k) in ((0, 3), (1, 0), (1, 1), (5, 0), (5, 3)),
        alpha in (zero(T), T(0.7)),
        beta in (zero(T), one(T), T(-0.3))

        alpha=T<:Complex && alpha!=0 ? T(0.7, -0.4) : alpha
        beta=T<:Complex && beta!=0 && beta!=1 ? T(-0.3, 0.2) : beta
        ar, ac=ta=='N' ? (n, k) : (k, n)
        br, bc=tb=='N' ? (k, n) : (n, k)
        aa=randn(rng, T, ar, ac)
        bb=randn(rng, T, br, bc)
        cc=randn(rng, T, n, n)
        a=extension_storage(aa)
        b=extension_storage(bb)
        c=extension_storage(cc)
        if alpha==0
            a[1:ar, :].=T(NaN)
            b[1:br, :].=T(NaN)
        end
        for j in 1:n, i in 1:n
            if (uplo=='L' && i<j)||(uplo=='U' && i>j)||beta==0
                c[i, j]=T(NaN)
            end
        end
        prior=copy(c)
        aprior=copy(a)
        bprior=copy(b)
        expected=if alpha==0||k==0
            zeros(T, n, n)
        else
            [
                alpha*sum(
                    (extension_op(aa, ta)[i, p]*extension_op(bb, tb)[p, j] for p in 1:k);
                    init=zero(T),
                ) for i in 1:n, j in 1:n
            ]
        end
        expected+=beta==0 ? zeros(T, n, n) : beta*cc
        extension_last_error()
        direct_triangular_product!(family, a, b, c, J; uplo, ta, tb, n, k, alpha, beta)
        @test extension_last_error()==0
        tolerance=real(T)===Float32 ? 2e-5 : 3e-13
        for j in 1:n, i in 1:n
            if (uplo=='L' && i>=j)||(uplo=='U' && i<=j)
                @test c[i, j]≈expected[i, j] atol=tolerance
            else
                @test isequal(c[i, j], prior[i, j])
            end
        end
        @test isequal(a, aprior)
        @test isequal(b, bprior)
        @test isequal(c[(n + 1):end, :], prior[(n + 1):end, :])
    end
end

@testset "SYR2/HER2 stride, conjugation and ignored diagonal components" begin
    rng=MersenneTwister(277)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        family in (T<:Complex ? ("syr2", "her2") : ("syr2",)),
        uplo in ('L', 'U'),
        n in (0, 1, 5),
        incx in (-3, -1, 1, 2),
        incy in (-2, -1, 1, 3),
        alpha in (zero(T), T(0.7))

        hermitian=family=="her2"
        alpha=T<:Complex && alpha!=0 ? T(0.7, -0.4) : alpha
        xv=randn(rng, T, n)
        yv=randn(rng, T, n)
        x=extension_vector(xv, incx)
        y=extension_vector(yv, incy)
        aa=randn(rng, T, n, n)
        a=extension_storage(aa)
        for j in 1:n, i in 1:n
            if (uplo=='L' && i<j)||(uplo=='U' && i>j)
                a[i, j]=T(NaN)
            elseif hermitian && i==j
                a[i, j]=T(real(aa[i, j]), NaN)
                aa[i, j]=real(aa[i, j])
            end
        end
        if alpha==0
            fill!(x, T(NaN))
            fill!(y, T(NaN))
        end
        prior=copy(a)
        xp=copy(x)
        yp=copy(y)
        expected=copy(aa)
        for j in 1:n, i in 1:n
            expected[i, j]+=if hermitian
                alpha*xv[i]*conj(yv[j])+conj(alpha)*yv[i]*conj(xv[j])
            else
                alpha*xv[i]*yv[j]+alpha*yv[i]*xv[j]
            end
        end
        extension_last_error()
        direct_rank2!(family, a, x, y, J; uplo, n, alpha, incx, incy)
        @test extension_last_error()==0
        tolerance=real(T)===Float32 ? 3e-5 : 3e-13
        for j in 1:n, i in 1:n
            if alpha==0 || (uplo=='L' && i<j)||(uplo=='U' && i>j)
                @test isequal(a[i, j], prior[i, j])
            else
                @test a[i, j]≈expected[i, j] atol=tolerance
                if hermitian && i==j
                    @test imag(a[i, j])==0
                end
            end
        end
        @test isequal(x, xp)
        @test isequal(y, yp)
        @test isequal(a[(n + 1):end, :], prior[(n + 1):end, :])
    end
end

for (T, prefix) in ((ComplexF32, "c"), (ComplexF64, "z")), J in (Int32, Int64)
    symbol=Symbol(prefix*"gemm3m_"*(J===Int64 ? "64_" : ""))
    @eval function direct_gemm3m!(
        a::Matrix{$T},
        b::Matrix{$T},
        c::Matrix{$T},
        ::Type{$J};
        m=size(c, 1),
        n=size(c, 2),
        k=size(a, 2),
        ta=('N'),
        tb=('N'),
        alpha=one($T),
        beta=zero($T),
        lda=size(a, 1),
        ldb=size(b, 1),
        ldc=size(c, 1),
        library=Colmat.backend_library(),
    )
        f=Libdl.dlsym(Libdl.dlopen(library), $(QuoteNode(symbol)))
        ccall(
            f,
            Cvoid,
            (
                Ref{UInt8},
                Ref{UInt8},
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
            ta,
            tb,
            m,
            n,
            k,
            alpha,
            a,
            lda,
            b,
            ldb,
            beta,
            c,
            ldc,
        )
        c
    end
end
@testset "GEMM3M ordinary native GEMM arithmetic compatibility" begin
    rng=MersenneTwister(278)
    for T in (ComplexF32, ComplexF64),
        J in (Int32, Int64),
        ta in ('N', 'T', 'C'),
        tb in ('N', 'T', 'C'),
        (m, n, k) in ((0, 4, 3), (3, 0, 4), (3, 5, 0), (3, 5, 2)),
        alpha in (zero(T), T(0.7, -0.4)),
        beta in (zero(T), one(T), T(-0.2, 0.3))

        ar, ac=ta=='N' ? (m, k) : (k, m)
        br, bc=tb=='N' ? (k, n) : (n, k)
        aa=randn(rng, T, ar, ac)
        bb=randn(rng, T, br, bc)
        cc=randn(rng, T, m, n)
        a=extension_storage(aa)
        b=extension_storage(bb)
        c=extension_storage(cc)
        if alpha==0
            a[1:ar, :].=T(NaN)
            b[1:br, :].=T(NaN)
        end
        if beta==0
            c[1:m, :].=T(NaN)
        end
        prior=copy(c)
        aprior=copy(a)
        bprior=copy(b)
        expected=zeros(T, m, n)
        for j in 1:n, i in 1:m
            value=zero(T)
            if alpha!=0
                for p in 1:k
                    value+=extension_op(aa, ta)[i, p]*extension_op(bb, tb)[p, j]
                end
            end
            expected[i, j]=alpha*value+(beta==0 ? zero(T) : beta*cc[i, j])
        end
        extension_last_error()
        direct_gemm3m!(a, b, c, J; m, n, k, ta, tb, alpha, beta)
        @test extension_last_error()==0
        @test c[1:m, :]≈expected atol=(real(T)===Float32 ? 3e-5 : 4e-13)
        @test isequal(a, aprior)
        @test isequal(b, bprior)
        @test isequal(c[(m + 1):end, :], prior[(m + 1):end, :])
    end
end

@testset "Dense extensions argument diagnostics and integer width" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
        a=fill(T(NaN), 6, 3)
        b=copy(a)
        c=fill(T(777), 6, 3)
        prior=copy(c)
        for family in ("gemmt", "gemmtr")
            for (options, status) in (
                ((; uplo=('?')), 1),
                ((; ta=('?')), 2),
                ((; tb=('?')), 3),
                ((; n=-1), 4),
                ((; k=-1), 5),
                ((; lda=2), 8),
                ((; ldb=2), 10),
                ((; ldc=2), 13),
            )
                extension_last_error()
                direct_triangular_product!(family, a, b, c, J; n=3, k=3, options...)
                @test extension_last_error()==status
                @test c==prior
            end
            if J===Int64
                extension_last_error()
                direct_triangular_product!(
                    family,
                    a,
                    b,
                    c,
                    J;
                    n=3,
                    k=1,
                    lda=Int64(2)^32+1,
                    alpha=zero(T),
                    beta=one(T),
                )
                @test extension_last_error()==0
                @test c==prior
            end
        end
        x=fill(T(NaN), 7)
        y=copy(x)
        for family in (T<:Complex ? ("syr2", "her2") : ("syr2",))
            for (options, status) in (
                ((; uplo=('?')), 1),
                ((; n=-1), 2),
                ((; incx=0), 5),
                ((; incy=0), 7),
                ((; lda=2), 9),
            )
                extension_last_error()
                direct_rank2!(family, c, x, y, J; options...)
                @test extension_last_error()==status
                @test c==prior
            end
        end
    end
end

@testset "Dense extensions numerical LBT entry points" begin
    for (T, prefix) in
        ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
        J in (Int32, Int64)

        abi=J===Int32 ? :lp64 : :ilp64
        families=if T<:Complex
            ("gemmt", "gemmtr", "syr2", "her2", "gemm3m")
        else
            ("gemmt", "gemmtr", "syr2")
        end
        for family in families
            backing=BLAS.lbt_find_backing_library(prefix*family*"_", abi)
            @test backing!==nothing
            @test realpath(backing.libname)==realpath(Colmat.backend_library())
            a=Matrix{T}(I, 3, 3)
            b=copy(a)
            c=zeros(T, 3, 3)
            if family in ("gemmt", "gemmtr")
                direct_triangular_product!(
                    family, a, b, c, J; library=BLAS.libblastrampoline
                )
                @test c==a
            elseif family=="gemm3m"
                direct_gemm3m!(a, b, c, J; library=BLAS.libblastrampoline)
                @test c==a
            else
                x=extension_vector(ones(T, 3), 1)
                y=extension_vector(fill(T(2), 3), 1)
                direct_rank2!(family, c, x, y, J; library=BLAS.libblastrampoline)
                @test all(==(T(4)), [c[i, j] for j in 1:3 for i in j:3])
                @test all(iszero, [c[i, j] for j in 1:3 for i in 1:(j - 1)])
            end
        end
    end
end
