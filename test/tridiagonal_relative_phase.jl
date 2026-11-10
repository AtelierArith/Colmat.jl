using Test, Colmat, LinearAlgebra

# Componentwise real division keeps this diagnostic finite for subnormal x.
_relative_phase_scaled(z, x) = complex(real(z)/x, imag(z)/x)

@testset "Tiny relative tridiagonal complex phases" begin
    team=FaerTeam(2)
    try
        for T in (ComplexF32, ComplexF64)
            R=typeof(real(zero(T)))
            scales=R===Float32 ? (one(R), R(1e-30), R(2)^(-130), R(2)^(-140)) :
                   (one(R), R(1e-300), R(2)^(-1030), R(2)^(-1070))
            for (index, x) in enumerate(scales),
                direction in (T(0, 1), T(-1, 0), T(0.6, 0.8))
                # Pure-axis phases remain exact even at the deepest subnormals;
                # mixed phases also exercise accumulation with a small rounding bound.
                index==4 && direction==T(0.6, 0.8) && continue
                tol=512eps(R)
                bandtol=tol+8nextfloat(zero(R))/x
                for n in (2, 4), bs in (0, 1, 2)
                    bs>n-1 && continue
                    source=Matrix{T}(I, n, n)
                    for i = 1:n
                        source[i, i]=T(i)
                        if i<n
                            source[i+1, i]=direction*x
                            source[i, i+1]=conj(source[i+1, i])
                        end
                    end
                    snapshot=copy(source)
                    for policy in (
                            FaerParThreshold(),
                            FaerParThreshold(0),
                            FaerParThreshold(4096),
                            FaerParThreshold(:never),
                        ),
                        resource in (nothing, team)

                        work=FaerScratch(
                            tridiagonalize,
                            source;
                            blocksize = bs,
                            team = resource,
                            par_threshold = policy,
                        )
                        try
                            F=tridiagonalize(
                                source;
                                scratch = work,
                                blocksize = bs,
                                team = resource,
                                par_threshold = policy,
                            )
                            try
                                Q=Matrix(F.Q)
                                reconstructed=Q*Matrix(F.H)*adjoint(Q)
                                @test all(isfinite, Q)
                                @test all(isfinite, F.H.dv)
                                @test all(isfinite, F.H.ev)
                                @test adjoint(Q)*Q≈Matrix{T}(I, n, n) atol=tol
                                @test norm(source-reconstructed)<=tol*norm(source)
                                # Native Householder explicitly flushes subnormal heads.
                                # Retain its normwise contract there, rather than claiming
                                # componentwise preservation beyond the kernel.
                                for i in (x>=floatmin(R) ? (1:(n-1)) : (1:0))
                                    # A global norm alone would ignore these tiny entries.
                                    @test abs(
                                        _relative_phase_scaled(reconstructed[i+1, i], x)-_relative_phase_scaled(
                                            source[i+1, i],
                                            x,
                                        ),
                                    )<=bandtol
                                end
                                @test source==snapshot
                            finally
                                close(F)
                            end
                        finally
                            close(work)
                        end
                    end
                end
            end
        end
    finally
        close(team)
    end
end

@testset "Tiny phase staged tridiagonal and self-adjoint bang" begin
    FaerTeam(2) do team
        for T in (ComplexF32, ComplexF64)
            R=typeof(real(zero(T)))
            tiny=R===Float32 ? R(1e-30) : R(1e-300)
            for x in (tiny, R(2)^(R===Float32 ? -140 : -1070)), wrapper in (false, true)
                source=Matrix{T}(I, 4, 4)
                for i = 1:3
                    source[i+1, i]=T(0, x)
                    source[i, i+1]=conj(source[i+1, i])
                end
                A=FaerMatrix(source)
                policy=FaerParThreshold(:never)
                work=FaerScratch(tridiagonalize!, A; team, par_threshold = policy)
                try
                    F=wrapper ?
                      hessenberg!(
                        Hermitian(A);
                        scratch = work,
                        team,
                        par_threshold = policy,
                    ) : tridiagonalize!(A; scratch = work, team, par_threshold = policy)
                    try
                        Q=Matrix(F.Q)
                        reconstructed=Q*Matrix(F.H)*adjoint(Q)
                        @test all(isfinite, Q)
                        @test Matrix(A)==Matrix(F.H)
                        @test adjoint(Q)*Q≈Matrix{T}(I, 4, 4) atol=512eps(R)
                        for i in (x>=floatmin(R) ? (1:3) : (1:0))
                            @test abs(
                                _relative_phase_scaled(reconstructed[i+1, i], x)-_relative_phase_scaled(
                                    source[i+1, i],
                                    x,
                                ),
                            )<=512eps(R)
                        end
                    finally
                        close(F)
                    end
                finally
                    close(work)
                    close(A)
                end
            end
        end
    end
end

@testset "Globally subnormal tridiagonal inputs retain their band" begin
    FaerTeam(2) do team
        for T in (ComplexF32, ComplexF64), n in (2, 4)
            R=typeof(real(zero(T)))
            x=R(2)^(R===Float32 ? -140 : -1070)
            base=Matrix{T}(I, n, n)
            for i = 1:n
                base[i, i]=T(i)
                if i<n
                    base[i+1, i]=T(0, 1)
                    base[i, i+1]=T(0, -1)
                end
            end
            source=base*x
            for resource in (nothing, team),
                p in (FaerParThreshold(), FaerParThreshold(0), FaerParThreshold(:never))

                F=tridiagonalize(source; team = resource, par_threshold = p)
                try
                    Q=Matrix(F.Q)
                    @test all(isfinite, Q)
                    @test all(isfinite, F.H.dv)
                    @test all(isfinite, F.H.ev)
                    @test norm(base-Q*(Matrix(F.H) ./ x)*adjoint(Q))<=512eps(R)*norm(base)
                    @test adjoint(Q)*Q≈Matrix{T}(I, n, n) atol=512eps(R)
                    @test source==base*x
                finally
                    close(F)
                end
            end
        end
    end
end
