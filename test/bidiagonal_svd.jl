using Test, LinearAlgebra, Random, Libdl, Colmat
function bidiagonal_svd_diagnostic()
    return ccall(
        Libdl.dlsym(Libdl.dlopen(Colmat.backend_library()), :colmat_blas_last_error),
        Cint,
        (),
    )
end
function bidiagonal_svd_reference()
    return only(
        filter(
            l->l.interface==:ilp64&&occursin("openblas", lowercase(basename(l.libname))),
            BLAS.get_config().loaded_libs,
        ),
    ).libname
end
for (T, R, prefix) in (
        (Float32, Float32, "s"),
        (Float64, Float64, "d"),
        (ComplexF32, Float32, "c"),
        (ComplexF64, Float64, "z"),
    ),
    J in (Int32, Int64)

    symbol=Symbol(prefix*"bdsqr_"*(J===Int64 ? "64_" : ""))
    @eval function direct_bdsqr!(
        d::AbstractVector{$R},
        e::AbstractVector{$R},
        vt::Matrix{$T},
        u::Matrix{$T},
        c::Matrix{$T},
        ::Type{$J};
        uplo=('U'),
        n=length(d),
        ncvt=size(vt, 2),
        nru=size(u, 1)-2,
        ncc=size(c, 2),
        ldvt=size(vt, 1),
        ldu=size(u, 1),
        ldc=size(c, 1),
        library=Colmat.backend_library(),
    )
        work=fill($R(777), max(1, 4*max(n, 0))+2)
        wp=view(work, 2:(length(work) - 1))
        info=fill($J(-777), 3)
        ip=view(info, 2:2)
        f=Libdl.dlsym(Libdl.dlopen(library), $(QuoteNode(symbol)))
        ccall(
            f,
            Cvoid,
            (
                Ref{UInt8},
                Ref{$J},
                Ref{$J},
                Ref{$J},
                Ref{$J},
                Ptr{$R},
                Ptr{$R},
                Ptr{$T},
                Ref{$J},
                Ptr{$T},
                Ref{$J},
                Ptr{$T},
                Ref{$J},
                Ptr{$R},
                Ptr{$J},
            ),
            uplo,
            n,
            ncvt,
            nru,
            ncc,
            d,
            e,
            vt,
            ldvt,
            u,
            ldu,
            c,
            ldc,
            wp,
            ip,
        )
        (; info=info[2], guard=info[[1, 3]], work, diagnostic=bidiagonal_svd_diagnostic())
    end
end
@testset "Bidiagonal SVD factor accumulation and reference residuals" begin
    rng=MersenneTwister(306)
    reference=bidiagonal_svd_reference()
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64), n in (0, 1, 2, 7, 40), uplo in ('U', 'L'),
        deficient in (false, true)

        R=real(T)
        tol=3000eps(R)
        input_d=randn(rng, R, n)
        input_e=randn(rng, R, max(0, n-1))/3
        if deficient
            input_d[1:2:end].=0
            input_e.=0
        end
        original=Matrix(Bidiagonal(input_d, input_e, uplo))
        initial_c=randn(rng, T, n, 3)
        for (library, K) in ((Colmat.backend_library(), J), (reference, Int64))
            ds=vcat(R(777), input_d, R(777))
            es=vcat(R(777), input_e, R(777))
            d=view(ds, 2:(length(ds) - 1))
            e=view(es, 2:(length(es) - 1))
            u=fill(T(777), n+2, n)
            vt=copy(u)
            c=fill(T(777), n+2, 3)
            u[1:n, :]=Matrix{T}(I, n, n)
            vt[1:n, :]=Matrix{T}(I, n, n)
            c[1:n, :]=initial_c
            r=direct_bdsqr!(d, e, vt, u, c, K; uplo, library)
            @test r.info==r.diagnostic==0
            @test r.guard==fill(K(-777), 2)
            @test ds[[1, end]]==es[[1, end]]==fill(R(777), 2)
            @test r.work[[1, end]]==fill(R(777), 2)
            @test all(>=(0), d)&&issorted(d; rev=true)
            left=u[1:n, :]
            rightt=vt[1:n, :]
            @test left*Diagonal(d)*rightt≈original atol=tol*max(norm(original), floatmin(R)) rtol=tol
            @test left'*left≈Matrix{T}(I, n, n) atol=tol rtol=tol
            @test rightt*rightt'≈Matrix{T}(I, n, n) atol=tol rtol=tol
            @test c[1:n, :]≈left'*initial_c atol=tol rtol=tol
            @test all(==(T(777)), u[(n + 1):end, :])
            @test all(==(T(777)), vt[(n + 1):end, :])
            @test all(==(T(777)), c[(n + 1):end, :])
        end
    end
end
