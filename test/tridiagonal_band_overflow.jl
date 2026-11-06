using Colmat, LinearAlgebra, Test

@testset "Unrepresentable real tridiagonal bands reject before writeback" begin
    team=FaerTeam(2)
    try
        for T in (ComplexF32, ComplexF64),
            resource in (nothing, team),
            compact in (false, true)

            z=T===ComplexF32 ? T(3.0f38, 3.0f38) : T(1.3e308, 1.3e308)
            data=T[0 conj(z); z 0]
            @test all(isfinite, data)
            @test ishermitian(data)
            @test !isfinite(abs(z))
            error_type=Colmat.RustCall.RustError
            @test_throws error_type tridiagonalize(data; compact, team = resource)
            source=FaerMatrix(data)
            before=Matrix(source)
            try
                @test_throws error_type tridiagonalize!(source; compact, team = resource)
                @test Matrix(source)==before
            finally
                close(source)
            end
            for uplo in (:L, :U), borrowed in (false, true)
                parent=FaerMatrix(zeros(T, 4, 4))
                source=borrowed ? view(parent, 2:3, 2:3) : FaerMatrix(data)
                borrowed&&copyto!(source, data)
                owner=borrowed ? parent : source
                before=Matrix(owner)
                try
                    @test_throws error_type hessenberg!(
                        Hermitian(source, uplo);
                        compact,
                        team = resource,
                    )
                    @test Matrix(owner)==before
                finally
                    close(source)
                    close(parent)
                end
            end
        end
    finally
        close(team)
    end
end

@testset "Large representable and tiny-phase tridiagonal bands remain valid" begin
    team=FaerTeam(2)
    try
        for T in (ComplexF32, ComplexF64),
            resource in (nothing, team),
            compact in (false, true)

            R=real(T)
            z=T===ComplexF32 ? T(2.3f38, 2.3f38) : T(1.2e308, 1.2e308)
            data=T[0 conj(z); z 0]
            scale=max(abs(real(z)), abs(imag(z)))
            @test all(isfinite, data)&&isfinite(abs(z))
            f=tridiagonalize(data; compact, team = resource)
            try
                @test all(isfinite, f.H.dv)&&all(isfinite, f.H.ev)
                @test f.H.ev[1]≈abs(z) rtol=100eps(R)
                @test Matrix(f)/scale≈data/scale rtol=100eps(R) atol=100eps(R)
                q=Matrix(f.Q)
                @test q'*q≈Matrix{T}(I, 2, 2) rtol=100eps(R) atol=100eps(R)
            finally
                close(f)
            end
            source=FaerMatrix(data)
            try
                f=tridiagonalize!(source; compact, team = resource)
                try
                    @test Matrix(source)==Matrix(f.H)
                    @test Matrix(f)/scale≈data/scale rtol=100eps(R) atol=100eps(R)
                finally
                    close(f)
                end
            finally
                close(source)
            end
            small=T===ComplexF32 ? R(1e-30) : R(1e-300)
            tiny=T[1 -small*im; small*im 2]
            f=tridiagonalize(tiny; compact, team = resource)
            try
                rebuilt=Matrix(f)
                @test abs(imag(rebuilt[2, 1])/small-one(R))<=100eps(R)
                @test abs(real(rebuilt[2, 1])/small)<=100eps(R)
                @test all(isfinite, f.H.dv)&&all(isfinite, f.H.ev)
            finally
                close(f)
            end
        end
    finally
        close(team)
    end
end
