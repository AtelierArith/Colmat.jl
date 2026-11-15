using Test, LinearAlgebra, Random, Libdl, Colmat
function packed_bk_diagnostic()
    return ccall(
        Libdl.dlsym(Libdl.dlopen(Colmat.backend_library()), :colmat_blas_last_error),
        Cint,
        (),
    )
end
for (T, p, kind) in (
        (Float32, "s", "sp"),
        (Float64, "d", "sp"),
        (ComplexF32, "c", "hp"),
        (ComplexF64, "z", "hp"),
    ),
    J in (Int32, Int64)

    suffix=J===Int64 ? "64_" : ""
    factor=Symbol(p*kind*"trf_"*suffix)
    solve=Symbol(p*kind*"trs_"*suffix)
    inverse=Symbol(p*kind*"tri_"*suffix)
    @eval function packed_bk_factor!(
        ap::AbstractVector{$T}, ::Type{$J}; uplo=('L'), n, library=Colmat.backend_library()
    )
        piv=fill($J(-777), max(0, n)+2)
        ipiv=view(piv, 2:(length(piv) - 1))
        info=fill($J(-777), 3)
        ip=view(info, 2:2)
        f=Libdl.dlsym(Libdl.dlopen(library), $(QuoteNode(factor)))
        ccall(
            f,
            Cvoid,
            (Ref{UInt8}, Ref{$J}, Ptr{$T}, Ptr{$J}, Ptr{$J}),
            uplo,
            n,
            ap,
            ipiv,
            ip,
        )
        (; piv, info=info[2], guard=info[[1, 3]], diagnostic=packed_bk_diagnostic())
    end
    @eval function packed_bk_solve!(
        ap::AbstractVector{$T},
        piv::AbstractVector{$J},
        b::Matrix{$T};
        uplo=('L'),
        n,
        nrhs=size(b, 2),
        ldb=size(b, 1),
        library=Colmat.backend_library(),
    )
        info=fill($J(-777), 3)
        ip=view(info, 2:2)
        f=Libdl.dlsym(Libdl.dlopen(library), $(QuoteNode(solve)))
        ccall(
            f,
            Cvoid,
            (Ref{UInt8}, Ref{$J}, Ref{$J}, Ptr{$T}, Ptr{$J}, Ptr{$T}, Ref{$J}, Ptr{$J}),
            uplo,
            n,
            nrhs,
            ap,
            piv,
            b,
            ldb,
            ip,
        )
        (; info=info[2], guard=info[[1, 3]], diagnostic=packed_bk_diagnostic())
    end
    @eval function packed_bk_inverse!(
        ap::AbstractVector{$T},
        piv::AbstractVector{$J};
        uplo=('L'),
        n,
        library=Colmat.backend_library(),
    )
        info=fill($J(-777), 3)
        ip=view(info, 2:2)
        work=fill($T(777), max(1, n))
        f=Libdl.dlsym(Libdl.dlopen(library), $(QuoteNode(inverse)))
        ccall(
            f,
            Cvoid,
            (Ref{UInt8}, Ref{$J}, Ptr{$T}, Ptr{$J}, Ptr{$T}, Ptr{$J}),
            uplo,
            n,
            ap,
            piv,
            work,
            ip,
        )
        (; info=info[2], guard=info[[1, 3]], diagnostic=packed_bk_diagnostic())
    end
end
function packed_bk_storage(a, uplo)
    n=size(a, 1)
    T=eltype(a)
    packed=T[T(777)]
    for j in 1:n, i in (uplo=='L' ? (j:n) : (1:j))
        push!(packed, i==j && T<:Complex ? complex(real(a[i, j]), real(T)(NaN)) : a[i, j])
    end
    push!(packed, T(777))
    return packed
end
function packed_bk_dense(ap, n, uplo)
    T=eltype(ap)
    a=zeros(T, n, n)
    k=1
    for j in 1:n, i in (uplo=='L' ? (j:n) : (1:j))
        a[i, j]=ap[k]
        k+=1
    end
    return Matrix(Hermitian(a, uplo=='L' ? :L : :U))
end
@testset "Packed Bunch-Kaufman external factor interoperability" begin
    reference=only(
        filter(
            l->l.interface==:ilp64&&occursin("openblas", lowercase(basename(l.libname))),
            BLAS.get_config().loaded_libs,
        ),
    ).libname
    rng=MersenneTwister(297)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64), uplo in ('L', 'U'), n in (0, 1, 2, 5, 8),
        mode in (:dense, :twopivot, :singular)

        tol=1000eps(real(T))
        a=Matrix(Hermitian(randn(rng, T, n, n)))
        if mode==:twopivot
            a=zeros(T, n, n)
            for j in 1:2:(n - 1)
                a[j + 1, j]=T(2)+(T<:Complex ? T(im/3) : zero(T))
                a[j, j + 1]=conj(a[j + 1, j])
            end
            isodd(n)&&(a[n, n]=T(3))
        elseif mode==:singular
            a=zeros(T, n, n)
        else
            for j in 1:n
                a[j, j]+=T(isodd(j) ? 3n : -3n)
            end
        end
        for (factor_library, K) in ((Colmat.backend_library(), J), (reference, Int64))
            storage=packed_bk_storage(a, uplo)
            ap=view(storage, 2:(length(storage) - 1))
            result=packed_bk_factor!(ap, K; uplo, n, library=factor_library)
            @test result.diagnostic==0
            @test result.guard==fill(K(-777), 2)
            @test result.piv[[1, end]]==fill(K(-777), 2)
            @test storage[[1, end]]==fill(T(777), 2)
            if mode==:singular && n>0
                @test 1<=result.info<=n
                continue
            end
            @test result.info==0
            if mode==:twopivot && n>=2
                @test any(<(0), result.piv[2:(end - 1)])
            end
            for (library, L) in ((Colmat.backend_library(), J), (reference, Int64))
                piv=L.(result.piv[2:(end - 1)])
                for nrhs in (0, 1, 3)
                    b=fill(T(777), n+2, nrhs)
                    rhs=randn(rng, T, n, nrhs)
                    b[1:n, :]=rhs
                    solved=packed_bk_solve!(ap, piv, b; uplo, n, library)
                    @test solved.info==solved.diagnostic==0
                    @test solved.guard==fill(L(-777), 2)
                    @test a*b[1:n, :]≈rhs rtol=tol atol=tol
                    @test all(==(T(777)), b[(n + 1):end, :])
                end
                inverted=copy(storage)
                invap=view(inverted, 2:(length(inverted) - 1))
                r=packed_bk_inverse!(invap, piv; uplo, n, library)
                @test r.info==r.diagnostic==0
                @test r.guard==fill(L(-777), 2)
                @test inverted[[1, end]]==fill(T(777), 2)
                @test a*packed_bk_dense(invap, n, uplo)≈Matrix{T}(I, n, n) rtol=tol atol=tol
            end
        end
    end
end
