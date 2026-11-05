using Test, Colmat, LinearAlgebra, Random
@testset "Tridiagonal construction resources" begin
    rng=MersenneTwister(73)
    team=FaerTeam(2)
    try
        for T in (Float32, Float64, ComplexF32, ComplexF64), n in (0, 1, 2, 3, 17, 65)
            R=typeof(real(zero(T)))
            raw=randn(rng, T, n, n)
            A=Matrix(Hermitian(raw))
            for bs in unique([0, 1, min(2, max(1, n-1))]), resource in (nothing, team)
                req=scratch_requirements(tridiagonalize, A; blocksize = bs, team = resource)
                s=FaerScratch(tridiagonalize, A; blocksize = bs, team = resource)
                try
                    @test scratch_capacity(s).bytes>=req.bytes
                    for rep = 1:2
                        F=tridiagonalize(A; scratch = s, team = resource, blocksize = bs)
                        Q=Matrix(F.Q)
                        @test norm(A-Q*Matrix(F.H)*Q') <= 2048eps(R)*max(norm(A), one(R))
                        @test Q'Q ≈ Matrix{T}(I, n, n) atol=2048eps(R)
                        close(F)
                    end
                    if req.bytes>0
                        short=FaerScratch(req.bytes-1; alignment = req.alignment)
                        try
                            @test_throws ArgumentError tridiagonalize(
                                A;
                                scratch = short,
                                team = resource,
                                blocksize = bs,
                            )
                            reserve!(
                                short,
                                tridiagonalize!,
                                A;
                                blocksize = bs,
                                team = resource,
                            )
                            F=tridiagonalize(
                                A;
                                scratch = short,
                                team = resource,
                                blocksize = bs,
                            )
                            @test Matrix(F) ≈ A rtol=2048eps(R) atol=2048eps(R)
                            close(F)
                        finally
                            close(short)
                        end
                    end
                finally
                    close(s)
                end
                @test_throws ArgumentError tridiagonalize(
                    A;
                    scratch = s,
                    team = resource,
                    blocksize = bs,
                )
            end
            native=FaerMatrix(A)
            h=Hermitian(native)
            work=FaerScratch(hessenberg, h; team)
            try
                @test scratch_requirements(hessenberg, h; team)==scratch_requirements(
                    tridiagonalize,
                    h;
                    team,
                )
                F=hessenberg!(h; scratch = work, team)
                @test Matrix(F) ≈ A rtol=2048eps(R) atol=2048eps(R)
                @test Matrix(native)==Matrix(F.H)
                close(F)
            finally
                close(work)
                close(native)
            end
        end
    finally
        close(team)
    end
end
struct _TridiagonalBadConversion <: AbstractMatrix{Float64} end
Base.size(::_TridiagonalBadConversion) = (2, 2)
Base.getindex(::_TridiagonalBadConversion, i::Int, j::Int) = i==j ? 1.0 : 0.0
Base.Matrix{T}(::_TridiagonalBadConversion) where {T} = zeros(T, 1, 1)
@testset "Tridiagonal conversion shape" begin
    @test_throws DimensionMismatch tridiagonalize(_TridiagonalBadConversion())
end
@testset "Tridiagonal resource failure and lifetime contracts" begin
    team=FaerTeam(2)
    try
        for T in (Float32, Float64, ComplexF32, ComplexF64)
            R=typeof(real(zero(T)))
            A=Matrix{T}(I, 4, 4)
            A[2, 1]=T<:Complex ? T(0.2, 0.3) : T(0.2)
            A[1, 2]=conj(A[2, 1])
            owner=FaerMatrix(A)
            v=view(owner, 4:-1:1, 4:-1:1)
            req=scratch_requirements(tridiagonalize, v; team)
            work=FaerScratch(tridiagonalize, v; team)
            try
                short=FaerScratch(req.bytes-1; alignment = req.alignment)
                try
                    @test_throws ArgumentError tridiagonalize!(owner; scratch = short, team)
                    @test Matrix(owner)==A
                    @test Matrix(v)==A[4:-1:1, 4:-1:1]
                finally
                    close(short)
                end
                misaligned=FaerScratch(req.bytes; alignment = 1)
                try
                    @test_throws ArgumentError tridiagonalize(v; scratch = misaligned, team)
                    @test Matrix(owner)==A
                finally
                    close(misaligned)
                end
                old=owner.dims
                owner.dims=(5, 5)
                try
                    @test_throws Exception scratch_requirements(tridiagonalize, v; team)
                    @test_throws Exception tridiagonalize(v; scratch = work, team)
                finally
                    owner.dims=old
                end
                jobs=[@async tridiagonalize(v; scratch = work, team) for _ = 1:2]
                results=fetch.(jobs)
                GC.gc()
                for F in results
                    @test Matrix(F) ≈ A[4:-1:1, 4:-1:1] rtol=2048eps(R) atol=2048eps(R)
                    close(F)
                end
                F=tridiagonalize(v; scratch = work, team)
                close(owner)
                GC.gc()
                @test Matrix(F) ≈ A[4:-1:1, 4:-1:1] rtol=2048eps(R) atol=2048eps(R)
                close(F)
                @test_throws ArgumentError tridiagonalize(v; scratch = work, team)
            finally
                close(work)
                close(v)
                close(owner)
            end
            bad=FaerMatrix(A)
            try
                bad[1, 2]=T(7)
                snapshot=Matrix(bad)
                @test_throws ArgumentError tridiagonalize!(bad; team)
                @test Matrix(bad)==snapshot
                bad[1, 1]=T(NaN)
                snapshot=Matrix(bad)
                @test_throws ArgumentError tridiagonalize!(bad; team)
                @test isequal(Matrix(bad), snapshot)
            finally
                close(bad)
            end
            exponent=R===Float32 ? 100 : 1000
            for scale in (R(2)^exponent, R(2)^(-exponent))
                F=tridiagonalize(A .* scale; team)
                Q=Matrix(F.Q)
                @test norm(A-Q*(Matrix(F.H) ./ scale)*Q') <= 2048eps(R)*norm(A)
                close(F)
            end
            @test_throws ArgumentError tridiagonalize(A; scratch = "invalid", team)
            @test_throws ArgumentError scratch_requirements(
                tridiagonalize,
                A;
                blocksize = -1,
                team,
            )
        end
    finally
        close(team)
    end
    @test_throws ArgumentError scratch_requirements(
        tridiagonalize,
        Matrix{Float64}(I, 2, 2);
        team,
    )
end
struct _TridiagonalBadDtype <: AbstractMatrix{Float64} end
Base.size(::_TridiagonalBadDtype) = (2, 2)
Base.getindex(::_TridiagonalBadDtype, i::Int, j::Int) = i==j ? 1.0 : 0.0
Base.Matrix{T}(::_TridiagonalBadDtype) where {T} = ones(UInt8, 2, 2)
@testset "Tridiagonal conversion storage type" begin
    @test_throws ArgumentError tridiagonalize(_TridiagonalBadDtype())
end
