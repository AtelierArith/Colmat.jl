using Test, LinearAlgebra, Random, Colmat
@testset "Native column-pivoted QR Julia entry" begin
    Random.seed!(299)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        (m, n) in ((7, 4), (4, 7), (4, 4), (0, 3), (3, 0)),
        zeros_input in (false, true)

        a=zeros_input ? zeros(T, m, n) : randn(T, m, n)
        f=qr(a, ColumnNorm())
        q=Matrix(f.Q)
        if m>0 && n>0
            @test q*f.R ≈ a[:, f.p]
            @test sort(f.p)==collect(1:n)
        else
            @test all(iszero, f.p)
        end
        @test q'*q ≈ Matrix{T}(I, min(m, n), min(m, n))
    end
end

using Libdl
for (T, p) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    I in (Int32, Int64)

    R=real(T)
    sym=Symbol(p*"geqp3rk_"*(I===Int64 ? "64_" : ""))
    @eval function direct_rank_qr!(
        a::Matrix{$T}, m, n, nrhs, kmax, ::Type{$I}; abstol=-1, reltol=-1, query=false
    )
        absolute=Ref{$R}(abstol)
        relative=Ref{$R}(reltol)
        rank=Ref{$I}(-1)
        residual=Ref{$R}(-1)
        ratio=Ref{$R}(-1)
        jpiv=zeros($I, n)
        tau=zeros($T, min(m, n))
        minimum=min(m, n)==0 ? 1 : ($T <: Real ? 3n+nrhs-1 : n+nrhs-1)
        work=zeros($T, max(1, minimum))
        rwork=zeros($R, max(1, 2n))
        iwork=zeros($I, max(1, n))
        info=Ref{$I}(0)
        f=Libdl.dlsym(Libdl.dlopen(Colmat.backend_library()), $(QuoteNode(sym)))
        if $T <: Real
            ccall(
                f,
                Cvoid,
                (
                    Ref{$I},
                    Ref{$I},
                    Ref{$I},
                    Ref{$I},
                    Ref{$R},
                    Ref{$R},
                    Ptr{$T},
                    Ref{$I},
                    Ref{$I},
                    Ref{$R},
                    Ref{$R},
                    Ptr{$I},
                    Ptr{$T},
                    Ptr{$T},
                    Ref{$I},
                    Ptr{$I},
                    Ref{$I},
                ),
                m,
                n,
                nrhs,
                kmax,
                absolute,
                relative,
                a,
                size(a, 1),
                rank,
                residual,
                ratio,
                jpiv,
                tau,
                work,
                query ? -1 : minimum,
                iwork,
                info,
            )
        else
            ccall(
                f,
                Cvoid,
                (
                    Ref{$I},
                    Ref{$I},
                    Ref{$I},
                    Ref{$I},
                    Ref{$R},
                    Ref{$R},
                    Ptr{$T},
                    Ref{$I},
                    Ref{$I},
                    Ref{$R},
                    Ref{$R},
                    Ptr{$I},
                    Ptr{$T},
                    Ptr{$T},
                    Ref{$I},
                    Ptr{$R},
                    Ptr{$I},
                    Ref{$I},
                ),
                m,
                n,
                nrhs,
                kmax,
                absolute,
                relative,
                a,
                size(a, 1),
                rank,
                residual,
                ratio,
                jpiv,
                tau,
                work,
                query ? -1 : minimum,
                rwork,
                iwork,
                info,
            )
        end
        (;
            rank=Int(rank[]),
            residual=residual[],
            ratio=ratio[],
            jpiv=Int.(jpiv),
            tau,
            info=Int(info[]),
            work=work[1],
        )
    end
end
@testset "GEQP3RK rank and residual direct ABIs" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        I in (Int32, Int64), limit in (0, 1, 3, 8),
        tolerances in ((-1.0, -1.0), (Inf, -1.0), (-1.0, 1.0), (0.0, 0.0))

        m, n, nrhs=6, 4, 2
        original=randn(T, m, n)
        rhs=randn(T, m, nrhs)
        a=fill(T(NaN), m+2, n+nrhs)
        a[1:m, :]=hcat(original, rhs)
        saved=copy(a)
        query=direct_rank_qr!(a, m, n, nrhs, limit, I; query=true)
        @test query.info==0
        @test isequal(saved, a)
        result=direct_rank_qr!(
            a, m, n, nrhs, limit, I; abstol=tolerances[1], reltol=tolerances[2]
        )
        @test result.info==0
        k=result.rank
        @test 0<=k<=min(limit, m, n)
        @test sort(result.jpiv)==collect(1:n)
        qinput=zeros(T, m, m)
        qinput[:, 1:k]=a[1:m, 1:k]
        q=LAPACK.orgqr!(qinput, result.tau[1:k], k)
        residual=copy(a[1:m, 1:n])
        for j in 1:k
            residual[(j + 1):m, j].=0
        end
        @test q*residual ≈ original[:, result.jpiv]
        @test a[1:m, (n + 1):end] ≈ q'*rhs
        measured=if k==min(m, n)
            zero(real(T))
        else
            maximum(norm(view(residual, (k + 1):m, j)) for j in (k + 1):n)
        end
        @test result.residual ≈ measured
        initial=maximum(norm(view(original, :, j)) for j in 1:n)
        @test result.ratio ≈ measured/initial
        @test all(isnan, a[(m + 1):end, :])
        if isinf(tolerances[1]) || tolerances[2]==1
            @test k==0
            @test a[1:m, :]==saved[1:m, :]
        end
    end
end

for (T, p) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    I in (Int32, Int64)

    R=real(T)
    suffix=I===Int64 ? "64_" : ""
    p3=Symbol(p*"geqp3_"*suffix)
    pf=Symbol(p*"geqpf_"*suffix)
    @eval function direct_pivot_qr!(
        a::Matrix{$T},
        m,
        n,
        piv::Vector{$I};
        legacy=false,
        query=false,
        library=Colmat.backend_library(),
    )
        tau=zeros($T, min(m, n))
        minimum=min(m, n)==0 ? 1 : ($T <: Real ? 3n+1 : n+1)
        work=zeros($T, max(1, 3n+1))
        rwork=zeros($R, max(1, 2n))
        info=Ref{$I}(0)
        f=Libdl.dlsym(Libdl.dlopen(library), legacy ? $(QuoteNode(pf)) : $(QuoteNode(p3)))
        if $T <: Real
            if legacy
                ccall(
                    f,
                    Cvoid,
                    (
                        Ref{$I},
                        Ref{$I},
                        Ptr{$T},
                        Ref{$I},
                        Ptr{$I},
                        Ptr{$T},
                        Ptr{$T},
                        Ref{$I},
                    ),
                    m,
                    n,
                    a,
                    size(a, 1),
                    piv,
                    tau,
                    work,
                    info,
                )
            else
                ccall(
                    f,
                    Cvoid,
                    (
                        Ref{$I},
                        Ref{$I},
                        Ptr{$T},
                        Ref{$I},
                        Ptr{$I},
                        Ptr{$T},
                        Ptr{$T},
                        Ref{$I},
                        Ref{$I},
                    ),
                    m,
                    n,
                    a,
                    size(a, 1),
                    piv,
                    tau,
                    work,
                    query ? -1 : minimum,
                    info,
                )
            end
        else
            if legacy
                ccall(
                    f,
                    Cvoid,
                    (
                        Ref{$I},
                        Ref{$I},
                        Ptr{$T},
                        Ref{$I},
                        Ptr{$I},
                        Ptr{$T},
                        Ptr{$T},
                        Ptr{$R},
                        Ref{$I},
                    ),
                    m,
                    n,
                    a,
                    size(a, 1),
                    piv,
                    tau,
                    work,
                    rwork,
                    info,
                )
            else
                ccall(
                    f,
                    Cvoid,
                    (
                        Ref{$I},
                        Ref{$I},
                        Ptr{$T},
                        Ref{$I},
                        Ptr{$I},
                        Ptr{$T},
                        Ptr{$T},
                        Ref{$I},
                        Ptr{$R},
                        Ref{$I},
                    ),
                    m,
                    n,
                    a,
                    size(a, 1),
                    piv,
                    tau,
                    work,
                    query ? -1 : minimum,
                    rwork,
                    info,
                )
            end
        end
        (; tau, piv=Int.(piv), info=Int(info[]), work=work[1])
    end
end
@testset "GEQP3 GEQPF fixed columns direct ABIs" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        I in (Int32, Int64),
        legacy in (false, true),
        flags in ([0, 0, 0, 0], [0, 3, 0, -2], [1, 0, 2, 0]),
        zero_matrix in (false, true)

        m, n=7, 4
        original=zero_matrix ? zeros(T, m, n) : randn(T, m, n)
        a=fill(T(NaN), m+2, n)
        a[1:m, :]=original
        if !legacy
            query=direct_pivot_qr!(a, m, n, I.(flags); query=true)
            @test query.info==0
            @test a[1:m, :]==original
        end
        result=direct_pivot_qr!(a, m, n, I.(flags); legacy)
        @test result.info==0
        @test sort(result.piv)==collect(1:n)
        fixed=findall(!iszero, flags)
        @test result.piv[1:length(fixed)]==fixed
        q=LAPACK.orgqr!(copy(a[1:m, :]), result.tau)
        @test q*triu(a[1:n, :]) ≈ original[:, result.piv]
        @test all(isnan, a[(m + 1):end, :])
        external=BLAS.lbt_find_backing_library("dgelqf_", :ilp64).libname
        ea=copy(original)
        er=direct_pivot_qr!(ea, m, n, Int64.(flags); legacy, library=external)
        @test er.info==0
        eq=LAPACK.orgqr!(copy(ea), er.tau)
        @test eq*triu(ea[1:n, :]) ≈ original[:, er.piv]
    end
end
@testset "GEQP3RK exceptional diagnostics" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), I in (Int32, Int64)
        a=Matrix{T}(LinearAlgebra.I, 3, 3)
        a[2, 2]=T(NaN)
        saved=copy(a)
        result=direct_rank_qr!(a, 3, 3, 0, 3, I)
        @test result.info==2
        @test result.rank==0 && isnan(result.residual) && isnan(result.ratio)
        @test isequal(a, saved)
        a=Matrix{T}(LinearAlgebra.I, 3, 3)
        a[2, 2]=T(Inf)
        result=direct_rank_qr!(a, 3, 3, 0, 0, I)
        @test result.info==5
        @test result.rank==0 && isinf(result.residual) && result.ratio==1
        a=Matrix{T}(LinearAlgebra.I, 3, 3)
        @test direct_rank_qr!(a, 3, 3, 0, 3, I; abstol=NaN).info==-5
        @test direct_rank_qr!(a, 3, 3, 0, 3, I; reltol=NaN).info==-6
    end
end

@testset "Pivoted QR empty and scalar ABI boundaries" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), I in (Int32, Int64)
        for (m, n) in ((0, 3), (3, 0), (0, 0))
            a=zeros(T, max(1, m), n)
            result=direct_pivot_qr!(a, m, n, zeros(I, n))
            @test result.info==0
            @test sort(result.piv)==collect(1:n)
            rank=direct_rank_qr!(a, m, n, 0, 3, I)
            @test rank.info==0 && rank.rank==0 && rank.residual==0 && rank.ratio==0
        end
        a=reshape(T[2], 1, 1)
        rank=direct_rank_qr!(a, 1, 1, 0, 1, I)
        @test rank.info==0 && rank.rank==1 && rank.residual==0
        a=reshape(T[2], 1, 1)
        @test direct_rank_qr!(a, 1, 1, 0, -1, I).info==-4
        a=reshape(T[Inf], 1, 1)
        rank=direct_rank_qr!(a, 1, 1, 0, 1, I; reltol=0.1)
        @test rank.info==2 && rank.rank==1 && rank.residual==0
    end
    for p in ("s", "d", "c", "z"),
        family in ("geqp3", "geqpf", "geqp3rk"),
        abi in (:lp64, :ilp64)

        @test realpath(BLAS.lbt_find_backing_library(p*family*"_", abi).libname)==realpath(
            Colmat.backend_library()
        )
    end
end
