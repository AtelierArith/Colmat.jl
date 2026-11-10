using Test, Colmat, LinearAlgebra, Random

@testset "Bidiagonal parallel policies" begin
    @test FaerParThreshold()==FaerParThreshold(:derived)
    @test FaerParThreshold(0)==FaerParThreshold(1)
    @test FaerParThreshold(0)!=FaerParThreshold()
    @test_throws ArgumentError FaerParThreshold(-1)
    @test_throws ArgumentError FaerParThreshold(typemax(UInt))
    @test_throws ArgumentError FaerParThreshold(big(typemax(UInt))+1)
    @test_throws ArgumentError FaerParThreshold(:unknown)
    rng=MersenneTwister(771)
    team=FaerTeam(2)
    try
        policies=(
            FaerParThreshold(),
            FaerParThreshold(0),
            FaerParThreshold(4096),
            FaerParThreshold(:never),
        )
        for T in (Float32, Float64, ComplexF32, ComplexF64),
            (m, n) in ((0, 3), (3, 0), (1, 1), (5, 3), (3, 5), (131, 129), (129, 131))

            R=typeof(real(zero(T)))
            tol=4096eps(R)
            k=min(m, n)
            A=randn(rng, T, m, n)
            snapshot=copy(A)
            for p in policies, full in (false, true), resource in (nothing, team)
                req=scratch_requirements(
                    bidiagonalize,
                    A;
                    full,
                    team = resource,
                    par_threshold = p,
                )
                s=FaerScratch(bidiagonalize, A; full, team = resource, par_threshold = p)
                try
                    F=bidiagonalize(
                        A;
                        full,
                        scratch = s,
                        team = resource,
                        par_threshold = p,
                    )
                    try
                        U, V=Matrix(F.U), Matrix(F.V)
                        @test norm(A-U[:, 1:k]*Matrix(F.B)*V[:, 1:k]')<=tol*max(
                            norm(A),
                            one(R),
                        )
                        @test U'U≈Matrix{T}(I, size(U, 2), size(U, 2)) atol=tol
                        @test V'V≈Matrix{T}(I, size(V, 2), size(V, 2)) atol=tol
                    finally
                        close(F)
                    end
                    @test scratch_requirements(
                        bidiagonalize!,
                        A;
                        full,
                        team = resource,
                        par_threshold = p,
                    )==req
                    if req.bytes>0
                        short=FaerScratch(req.bytes-1; alignment = req.alignment)
                        try
                            @test_throws ArgumentError bidiagonalize(
                                A;
                                full,
                                scratch = short,
                                team = resource,
                                par_threshold = p,
                            )
                            reserve!(
                                short,
                                bidiagonalize,
                                A;
                                full,
                                team = resource,
                                par_threshold = p,
                            )
                            F=bidiagonalize(
                                A;
                                full,
                                scratch = short,
                                team = resource,
                                par_threshold = p,
                            )
                            @test Matrix(F)≈A rtol=tol atol=tol
                            close(F)
                        finally
                            close(short)
                        end
                    end
                    @test A==snapshot
                finally
                    close(s)
                end
                @test_throws ArgumentError bidiagonalize(
                    A;
                    scratch = s,
                    team = resource,
                    par_threshold = p,
                )
            end
            @test scratch_requirements(
                bidiagonalize,
                A;
                full = true,
                team,
                par_threshold = FaerParThreshold(:never),
            ) == scratch_requirements(
                bidiagonalize,
                A;
                full = true,
                par_threshold = FaerParThreshold(:never),
            )
        end
        for p in policies
            a=FaerMatrix(ComplexF64[1 2 3; 4+im 5 6])
            original=Matrix(a)
            s=FaerScratch(bidiagonalize!, a; team, par_threshold = p)
            try
                F=fetch(@async bidiagonalize!(a; scratch = s, team, par_threshold = p))
                try
                    @test Matrix(F)≈original
                    @test Matrix(a)==hcat(Matrix(F.B), zeros(ComplexF64, 2, 1))
                finally
                    close(F)
                end
            finally
                close(s)
                close(a)
            end
        end
    finally
        close(team)
    end
    @test_throws ArgumentError scratch_requirements(
        bidiagonalize,
        zeros(0, 3);
        team,
        par_threshold = FaerParThreshold(:never),
    )
    @test_throws ArgumentError bidiagonalize(
        zeros(0, 3);
        team,
        par_threshold = FaerParThreshold(:never),
    )
end

struct _ThresholdConversion <: AbstractMatrix{Float64}
    calls::Base.RefValue{Int}
end
Base.size(::_ThresholdConversion) = (2, 3)
Base.getindex(::_ThresholdConversion, i::Int, j::Int) = Float64(i+j)
function Base.Matrix{T}(a::_ThresholdConversion) where {T}
    a.calls[]+=1
    return ones(T, 2, 3)
end
@testset "Bidiagonal policy validation precedes conversion" begin
    # Reuse an observable conversion rather than relying on a native crash.
    calls=Ref(0)
    a=_ThresholdConversion(calls)
    @test_throws ArgumentError scratch_requirements(
        bidiagonalize,
        a;
        par_threshold = :never,
    )
    @test_throws ArgumentError bidiagonalize(a; par_threshold = :never)
    @test calls[]==0
end
