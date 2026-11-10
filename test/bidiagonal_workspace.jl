using Test, Colmat, LinearAlgebra, Random

@testset "Bidiagonal SVD workspace" begin
    team=FaerTeam(2)
    try
        for T in (Float32, Float64, ComplexF32, ComplexF64)
            R=typeof(real(zero(T)))
            tol=4096eps(R)
            for n in (0, 1, 4, 17, 65, 129),
                threshold in (2, 128),
                resource in (nothing, team),
                uplo in (:U, :L)

                d=T[T<:Complex ? T(i, 0.2*(-1)^i) : T(i) for i = 1:n]
                e=fill(T<:Complex ? T(0.25, -0.125) : T(0.25), max(0, n-1))
                A=Matrix(Bidiagonal(d, e, uplo))
                saved=(copy(d), copy(e))
                req=scratch_requirements(
                    bidiagonal_svd,
                    d,
                    e;
                    uplo,
                    team = resource,
                    recursion_threshold = threshold,
                )
                s=FaerScratch(
                    bidiagonal_svd,
                    d,
                    e;
                    uplo,
                    team = resource,
                    recursion_threshold = threshold,
                )
                try
                    @test scratch_capacity(s).bytes>=req.bytes
                    for rep = 1:2
                        F=bidiagonal_svd(
                            d,
                            e;
                            uplo,
                            scratch = s,
                            team = resource,
                            recursion_threshold = threshold,
                        )
                        try
                            U, Vt=Matrix(F.U), Matrix(F.Vt)
                            @test norm(A-U*Diagonal(F.S)*Vt)<=tol*max(norm(A), one(R))
                            @test U'U≈Matrix{T}(I, n, n) atol=tol
                            @test Vt*Vt'≈Matrix{T}(I, n, n) atol=tol
                            @test F.S≈svdvals(A) rtol=tol atol=tol
                            @test issorted(F.S; rev = true)
                            @test all(>=(0), F.S)
                        finally
                            close(F.U)
                            close(F.Vt)
                        end
                    end
                    @test (d, e)==saved
                    @test scratch_capacity(s).bytes>=req.bytes
                    if req.bytes>0
                        short=FaerScratch(req.bytes-1; alignment = req.alignment)
                        try
                            @test_throws ArgumentError bidiagonal_svd(
                                d,
                                e;
                                uplo,
                                scratch = short,
                                team = resource,
                                recursion_threshold = threshold,
                            )
                            reserve!(
                                short,
                                bidiagonal_svd,
                                d,
                                e;
                                uplo,
                                team = resource,
                                recursion_threshold = threshold,
                            )
                            F=bidiagonal_svd(
                                d,
                                e;
                                uplo,
                                scratch = short,
                                team = resource,
                                recursion_threshold = threshold,
                            )
                            @test F.S≈svdvals(A) rtol=tol atol=tol
                            close(F.U)
                            close(F.Vt)
                        finally
                            close(short)
                        end
                    end
                finally
                    close(s)
                end
                @test_throws ArgumentError bidiagonal_svd(
                    d,
                    e;
                    scratch = s,
                    team = resource,
                )
            end
            exponent=R===Float32 ? 100 : 1000
            for scale in (R(2)^(-exponent), R(2)^exponent), uplo in (:U, :L)
                d=T[1, 2, 3] .* scale
                e=fill(T<:Complex ? T(0.25, 0.125)*scale : T(0.25)*scale, 2)
                A=Matrix(Bidiagonal(d, e, uplo)) ./ scale
                F=bidiagonal_svd(d, e; uplo, team, recursion_threshold = 2)
                try
                    @test norm(A-Matrix(F.U)*Diagonal(F.S ./ scale)*Matrix(F.Vt))<=tol*norm(
                        A,
                    )
                    @test F.S ./ scale≈svdvals(A) rtol=tol atol=tol
                finally
                    close(F.U)
                    close(F.Vt)
                end
            end
            d=FaerVector(T[1, 2, 3])
            e=FaerVector(T[0.25, 0.125])
            b=Bidiagonal(d, e, :L)
            s=FaerScratch(svd, b; team)
            try
                @test scratch_requirements(svd, b; team)==scratch_requirements(
                    svdvals,
                    b;
                    team,
                )
                @test fetch(@async svdvals(b; scratch = s, team))≈svdvals(Matrix(b)) rtol=tol atol=tol
                reserve!(s, svdvals, b; team)
                old=d.len
                d.len+=1
                try
                    @test_throws Exception scratch_requirements(svd, b; team)
                    @test_throws Exception svdvals(b; scratch = s, team)
                finally
                    d.len=old
                end
            finally
                close(d)
                close(e)
                close(s)
            end
            @test_throws ArgumentError scratch_requirements(svd, b; team)
            @test_throws ArgumentError svdvals(b; team)
        end
    finally
        close(team)
    end
    @test_throws ArgumentError scratch_requirements(
        bidiagonal_svd,
        Float64[],
        Float64[];
        team,
    )
    @test_throws ArgumentError bidiagonal_svd(Float64[], Float64[]; team)
    @test_throws ArgumentError scratch_requirements(
        bidiagonal_svd,
        [1.0],
        Float64[];
        team = :bad,
    )
    @test_throws ArgumentError bidiagonal_svd([1.0], Float64[]; scratch = :bad)
    @test_throws DimensionMismatch scratch_requirements(bidiagonal_svd, [1.0, 2], Float64[])
    @test_throws ArgumentError scratch_requirements(
        bidiagonal_svd,
        [1.0],
        Float64[];
        uplo = :bad,
    )
    @test_throws ArgumentError scratch_requirements(
        bidiagonal_svd,
        [1.0],
        Float64[];
        recursion_threshold = 1,
    )
    @test_throws ArgumentError bidiagonal_svd([NaN], Float64[])
end

@testset "Saved bidiagonal SVD resources" begin
    rng=MersenneTwister(734)
    team=FaerTeam(2)
    try
        for T in (Float32, Float64, ComplexF32, ComplexF64),
            (m, n) in ((0, 0), (0, 3), (3, 0), (1, 1), (5, 3), (3, 5)),
            full in (false, true)

            R=typeof(real(zero(T)))
            tol=4096eps(R)
            A=randn(rng, T, m, n)
            source=FaerMatrix(A)
            F=bidiagonalize(source; full)
            close(source)
            s=FaerScratch(svd, F; full, team, recursion_threshold = 2)
            try
                @test scratch_requirements(svd, F; full, team)==scratch_requirements(
                    bidiagonal_svd,
                    F.B.dv,
                    F.B.ev;
                    team,
                )
                for resource in (nothing, team)
                    reserve!(s, svd, F; full, team = resource, recursion_threshold = 2)
                    E=svd(F; full, scratch = s, team = resource, recursion_threshold = 2)
                    try
                        k=min(m, n)
                        U, Vt=Matrix(E.U), Matrix(E.Vt)
                        @test size(U)==(m, full ? m : k)
                        @test size(Vt)==(full ? n : k, n)
                        @test norm(A-U[:, 1:k]*Diagonal(E.S)*Vt[1:k, :])<=tol*max(
                            norm(A),
                            one(R),
                        )
                        @test E.S≈svdvals(A) rtol=tol atol=tol
                    finally
                        close(E.U)
                        close(E.Vt)
                    end
                end
                reserve!(s, svdvals, F; team, recursion_threshold = 2)
                @test svdvals(F; scratch = s, team, recursion_threshold = 2)≈svdvals(A) rtol=tol atol=tol
                old=F.U.dims
                F.U.dims=(old[1]+1, old[2])
                try
                    @test_throws Exception scratch_requirements(svd, F; team)
                    @test_throws Exception svd(F; scratch = s, team)
                finally
                    F.U.dims=old
                end
                if !full && m!=n
                    @test_throws ArgumentError scratch_requirements(
                        svd,
                        F;
                        full = true,
                        team,
                    )
                    @test_throws ArgumentError svd(F; full = true, scratch = s, team)
                end
            finally
                close(s)
                close(F)
            end
            @test_throws ArgumentError scratch_requirements(svd, F; team)
            @test_throws ArgumentError svdvals(F; team)
        end
    finally
        close(team)
    end
end

struct _BidiagonalBadConversion <: AbstractVector{Float64}
    mode::Symbol
    calls::Base.RefValue{Int}
end
Base.size(::_BidiagonalBadConversion) = (2,)
Base.getindex(::_BidiagonalBadConversion, i::Int) = Float64(i)
function Base.Vector{T}(v::_BidiagonalBadConversion) where {T}
    v.calls[]+=1
    return v.mode===:dtype ? zeros(UInt8, 2) : zeros(T, 1)
end
@testset "Bidiagonal conversion and resource ordering" begin
    for mode in (:dtype, :shape)
        calls=Ref(0)
        d=_BidiagonalBadConversion(mode, calls)
        e=[0.1]
        req=scratch_requirements(bidiagonal_svd, d, e)
        @test calls[]==0
        short=FaerScratch(req.bytes-1; alignment = req.alignment)
        try
            @test_throws ArgumentError bidiagonal_svd(d, e; scratch = short)
            @test calls[]==0
        finally
            close(short)
        end
        @test_throws (mode===:dtype ? ArgumentError : DimensionMismatch) bidiagonal_svd(
            d,
            e,
        )
        @test calls[]==1
    end
    req=scratch_requirements(bidiagonal_svd, [1.0, 2], [0.1])
    if req.alignment>1
        s=FaerScratch(req.bytes; alignment = 1)
        try
            @test_throws ArgumentError bidiagonal_svd([1.0, 2], [0.1]; scratch = s)
        finally
            close(s)
        end
    end
end

@testset "Bidiagonal shared resources" begin
    team=FaerTeam(2)
    try
        for T in (Float32, Float64, ComplexF32, ComplexF64)
            R=typeof(real(zero(T)))
            d=T[1, 2, 3]
            e=fill(T<:Complex ? T(0.2, 0.1) : T(0.2), 2)
            A=Matrix(Bidiagonal(d, e, :U))
            work=FaerScratch(bidiagonal_svd, d, e; team, recursion_threshold = 2)
            try
                results=fetch.([
                    @async begin
                        GC.gc(false)
                        F=bidiagonal_svd(
                            d,
                            e;
                            scratch = work,
                            team,
                            recursion_threshold = 2,
                        )
                        try
                            F.S
                        finally
                            close(F.U)
                            close(F.Vt)
                        end
                    end for _ = 1:4
                ])
                @test all(
                    x->isapprox(x, svdvals(A); rtol = 4096eps(R), atol = 4096eps(R)),
                    results,
                )
            finally
                close(work)
            end
            f=FaerBidiagonalization(
                FaerMatrix(zeros(T, 3, 0)),
                Bidiagonal(ones(T, 2), ones(T, 1), :U),
                FaerMatrix(zeros(T, 2, 2)),
            )
            try
                @test_throws DimensionMismatch scratch_requirements(svd, f; team)
                @test_throws DimensionMismatch svd(f; team)
            finally
                close(f)
            end
        end
    finally
        close(team)
    end
end
