using Test, Colmat, LinearAlgebra, Random

@testset "Bidiagonal construction resources" begin
    rng=MersenneTwister(761)
    team=FaerTeam(2)
    try
        for T in (Float32, Float64, ComplexF32, ComplexF64),
            (m, n) in (
                (0, 0),
                (0, 3),
                (3, 0),
                (1, 1),
                (4, 1),
                (1, 4),
                (4, 4),
                (6, 4),
                (4, 6),
                (131, 129),
                (129, 131),
            )

            R=typeof(real(zero(T)))
            tol=4096eps(R)
            A=randn(rng, T, m, n)
            original=copy(A)
            k=min(m, n)
            for full in (false, true),
                bs in unique([0, 1, min(2, max(1, k)), max(1, k)]),
                resource in (nothing, team)

                req=scratch_requirements(
                    bidiagonalize,
                    A;
                    full,
                    blocksize = bs,
                    team = resource,
                )
                work=FaerScratch(bidiagonalize, A; full, blocksize = bs, team = resource)
                try
                    @test scratch_capacity(work).bytes>=req.bytes
                    @test scratch_requirements(
                        bidiagonalize!,
                        A;
                        full,
                        blocksize = bs,
                        team = resource,
                    )==req
                    for rep = 1:2
                        F=bidiagonalize(
                            A;
                            full,
                            blocksize = bs,
                            scratch = work,
                            team = resource,
                        )
                        try
                            U, V=Matrix(F.U), Matrix(F.V)
                            @test size(U)==(m, full ? m : k)
                            @test size(V)==(n, full ? n : k)
                            @test norm(A-U[:, 1:k]*Matrix(F.B)*V[:, 1:k]')<=tol*max(
                                norm(A),
                                one(R),
                            )
                            @test U'U≈Matrix{T}(I, size(U, 2), size(U, 2)) atol=tol
                            @test V'V≈Matrix{T}(I, size(V, 2), size(V, 2)) atol=tol
                            @test F.B.uplo==(m<n ? 'L' : 'U')
                        finally
                            close(F)
                        end
                    end
                    if req.bytes>0
                        short=FaerScratch(req.bytes-1; alignment = req.alignment)
                        try
                            @test_throws ArgumentError bidiagonalize(
                                A;
                                full,
                                blocksize = bs,
                                scratch = short,
                                team = resource,
                            )
                            reserve!(
                                short,
                                bidiagonalize!,
                                A;
                                full,
                                blocksize = bs,
                                team = resource,
                            )
                            F=bidiagonalize(
                                A;
                                full,
                                blocksize = bs,
                                scratch = short,
                                team = resource,
                            )
                            @test Matrix(F)≈A rtol=tol atol=tol
                            close(F)
                        finally
                            close(short)
                        end
                    end
                    @test A==original
                finally
                    close(work)
                end
                @test_throws ArgumentError bidiagonalize(
                    A;
                    full,
                    blocksize = bs,
                    scratch = work,
                    team = resource,
                )
            end
            source=FaerMatrix(A)
            saved=copy(A)
            work=FaerScratch(bidiagonalize!, source; full = true, team)
            try
                F=bidiagonalize!(source; full = true, scratch = work, team)
                expected=zeros(T, m, n)
                expected[1:k, 1:k]=Matrix(F.B)
                @test Matrix(source)==expected
                close(source)
                GC.gc(false)
                @test Matrix(F)≈saved rtol=tol atol=tol
                close(F)
            finally
                close(work)
                close(source)
            end
            @test A==saved
        end
    finally
        close(team)
    end
end

@testset "Bidiagonal construction resource and lifetime contracts" begin
    team=FaerTeam(2)
    try
        for T in (Float32, Float64, ComplexF32, ComplexF64)
            R=typeof(real(zero(T)))
            tol=4096eps(R)
            A=reshape(T.(1:30), 6, 5)
            T<:Complex && (A .+= T(0.1, -0.2))
            owner=FaerMatrix(A)
            v=view(owner, 6:-2:2, 5:-1:1)
            expected=copy(A[6:-2:2, 5:-1:1])
            req=scratch_requirements(bidiagonalize, v; full = true, team)
            work=FaerScratch(bidiagonalize, v; full = true, team)
            try
                native_req=scratch_requirements(bidiagonalize!, owner; team)
                short=FaerScratch(native_req.bytes-1; alignment = native_req.alignment)
                try
                    @test_throws ArgumentError bidiagonalize!(owner; scratch = short, team)
                    @test Matrix(owner)==A
                finally
                    close(short)
                end
                misaligned=FaerScratch(req.bytes; alignment = 1)
                try
                    @test_throws ArgumentError bidiagonalize(
                        v;
                        full = true,
                        scratch = misaligned,
                        team,
                    )
                    @test Matrix(owner)==A
                finally
                    close(misaligned)
                end
                old=owner.dims
                owner.dims=(7, 5)
                try
                    @test_throws Exception scratch_requirements(bidiagonalize, v; team)
                    @test_throws Exception bidiagonalize(v; scratch = work, team)
                finally
                    owner.dims=old
                end
                results=fetch.([
                    @async bidiagonalize(v; full = true, scratch = work, team) for _ = 1:3
                ])
                GC.gc(false)
                for F in results
                    @test Matrix(F)≈expected rtol=tol atol=tol
                    close(F)
                end
                F=bidiagonalize(v; full = true, scratch = work, team)
                close(owner)
                GC.gc(false)
                @test Matrix(F)≈expected rtol=tol atol=tol
                close(F)
                @test_throws ArgumentError bidiagonalize(v; scratch = work, team)
                @test_throws ArgumentError scratch_requirements(bidiagonalize, v; team)
            finally
                close(work)
                close(v)
                close(owner)
            end
            bad=FaerMatrix(A)
            try
                bad[1, 1]=T(NaN)
                snapshot=Matrix(bad)
                @test_throws ArgumentError bidiagonalize!(bad; team)
                @test isequal(Matrix(bad), snapshot)
            finally
                close(bad)
            end
            exponent=R===Float32 ? 120 : 1000
            for scale in (R(2)^exponent, R(2)^(-exponent)),
                shape in ((5, 3), (3, 5)),
                bs in (0, 1, 2)

                m, n=shape
                a=Matrix{T}(I, m, n)
                a[2, 1]=T<:Complex ? T(0.2, 0.3) : T(0.2)
                for full in (false, true)
                    F=bidiagonalize(a .* scale; full, blocksize = bs, team)
                    try
                        k=min(shape...)
                        U, V=Matrix(F.U), Matrix(F.V)
                        @test norm(a-U[:, 1:k]*(Matrix(F.B) ./ scale)*V[:, 1:k]')<=tol*norm(
                            a,
                        )
                    finally
                        close(F)
                    end
                end
            end
            @test_throws ArgumentError bidiagonalize(A; scratch = "bad", team)
            @test_throws ArgumentError scratch_requirements(
                bidiagonalize,
                A;
                blocksize = -1,
                team,
            )
            row=FaerRow(T[1, 2, 3, 4])
            rv=view(row, 1:1, 4:-1:1)
            try
                wanted=Matrix(rv)
                F=bidiagonalize(rv; team, full = true)
                close(row)
                @test Matrix(F)≈wanted rtol=tol atol=tol
                close(F)
            finally
                close(rv)
                close(row)
            end
            @test_throws ArgumentError scratch_requirements(
                bidiagonalize,
                A;
                blocksize = 6,
                team,
            )
        end
    finally
        close(team)
    end
    @test_throws ArgumentError scratch_requirements(bidiagonalize, zeros(0, 3); team)
    @test_throws ArgumentError bidiagonalize(zeros(0, 3); team)
end

struct _BidiagonalReductionConversion <: AbstractMatrix{Float64}
    mode::Symbol
    calls::Base.RefValue{Int}
end
Base.size(::_BidiagonalReductionConversion) = (2, 3)
Base.getindex(::_BidiagonalReductionConversion, i::Int, j::Int) = Float64(i+j)
function Base.Matrix{T}(a::_BidiagonalReductionConversion) where {T}
    a.calls[]+=1
    return a.mode===:dtype ? zeros(UInt8, 2, 3) : zeros(T, 1, 1)
end
@testset "Bidiagonal reduction conversion ordering" begin
    for mode in (:dtype, :shape)
        calls=Ref(0)
        a=_BidiagonalReductionConversion(mode, calls)
        req=scratch_requirements(bidiagonalize, a)
        @test calls[]==0
        short=FaerScratch(req.bytes-1; alignment = req.alignment)
        try
            @test_throws ArgumentError bidiagonalize(a; scratch = short)
            @test calls[]==0
        finally
            close(short)
        end
        @test_throws (mode===:dtype ? ArgumentError : DimensionMismatch) bidiagonalize(a)
        @test calls[]==1
    end
end
