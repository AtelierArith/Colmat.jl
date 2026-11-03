using Test, LinearAlgebra, Random, Colmat

@testset "Native pseudoinverse and spectral interfaces" begin
    rng=MersenneTwister(190)
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        R=typeof(real(zero(T)))
        tol=R===Float32 ? 3e-4 : 3e-11
        for (m, n) in ((5, 3), (3, 5), (4, 4), (1, 1), (0, 3), (3, 0), (0, 0))
            data=randn(rng, T, m, n)
            a=FaerMatrix(data)
            b=pinv(a)
            @test b isa FaerMatrix{T}
            @test size(b)==(n, m)
            @test Matrix(b)≈pinv(data) rtol=tol atol=tol
            @test Matrix(a*b*a)≈data rtol=tol atol=tol
            @test Matrix(b*a*b)≈Matrix(b) rtol=tol atol=tol
            @test Matrix(a)==data
            @test rank(a)==rank(data)
            @test rank(a; atol = 1)==rank(data; atol = 1)
            @test rank(a; rtol = 0.25)==rank(data; rtol = 0.25)
            for p in (1, 2, Inf)
                @test opnorm(a, p)≈opnorm(data, p) rtol=tol atol=tol
            end
            if !isempty(data)||m==n
                @test cond(a)≈cond(data) rtol=tol atol=tol
            else
                @test_throws DimensionMismatch cond(a)
            end
            close(a)
            @test_throws ArgumentError pinv(a)
            @test_throws ArgumentError rank(a)
            @test_throws ArgumentError cond(a)
        end
        data=Matrix(Diagonal(T[4, 0.1, 0]))
        a=FaerMatrix(data)
        @test Matrix(pinv(a; atol = 0.2))≈Matrix(Diagonal(T[0.25, 0, 0]))
        @test Matrix(pinv(a; rtol = 0.1))≈pinv(data; rtol = 0.1)
        @test Matrix(pinv(a; atol = 0.2, rtol = 0.01))≈pinv(data; atol = 0.2, rtol = 0.01)
        @test rank(a; atol = R(0.1))==1
        @test rank(a; atol = 0.099)==2
        @test Matrix(pinv(a; atol = 0.1))≈pinv(data; atol = 0.1)
        @test Matrix(pinv(a; atol = R(0.1)))≈pinv(data; atol = R(0.1))
        @test Matrix(pinv(a; rtol = 0.025))≈pinv(data; rtol = 0.025)
        @test Matrix(pinv(a; rtol = R(0.025)))≈pinv(data; rtol = R(0.025))
        @test rank(a; atol = 0, rtol = 0)==2
        @test isinf(cond(a))
        @test isinf(cond(a, 1))
        @test isinf(cond(a, Inf))
        @test isinf(cond(FaerMatrix(zeros(T, 2, 2))))
        @test Matrix(pinv(a; atol = Inf))==zeros(T, 3, 3)
        for p in (1, 2, Inf)
            well=Matrix{T}(I, 3, 3)+randn(rng, T, 3, 3)/10
            @test cond(FaerMatrix(well), p)≈cond(well, p) rtol=tol
        end
        for kwargs in ((; atol = -1), (; rtol = -1), (; atol = NaN), (; rtol = NaN))
            @test_throws ArgumentError pinv(a; kwargs...)
            @test_throws ArgumentError rank(a; kwargs...)
        end
        @test_throws ArgumentError cond(a, 3)
        @test_throws ArgumentError opnorm(a, 3)
        for side in (:L, :U)
            hdata=T[3 1 0; 1 -2 0; 0 0 0]
            T<:Complex && (hdata[1, 2] = 1+im; hdata[2, 1] = 1-im)
            h=Hermitian(FaerMatrix(hdata), side)
            expected=pinv(Hermitian(hdata, side))
            b=pinv(h)
            @test Matrix(b)≈expected rtol=tol atol=tol
            @test Matrix(b)≈Matrix(b)' rtol=tol atol=tol
            @test Matrix(pinv(h; atol = 2))≈pinv(Hermitian(hdata, side); atol = 2) rtol=tol atol=tol
            @test rank(h)==2
            # Ignore unused triangle and imaginary Hermitian diagonal parts.
            garbage=copy(hdata)
            if side===:L
                garbage[1, 2]=T(NaN)
            else
                garbage[2, 1]=T(NaN)
            end
            if T<:Complex
                garbage[1, 1]=complex(R(3), R(NaN))
            end
            hg=Hermitian(FaerMatrix{T}(undef, 3, 3), side)
            for j = 1:3, i = 1:3
                parent(hg)[i, j]=garbage[i, j]
            end
            @test Matrix(pinv(hg))≈expected rtol=tol atol=tol
            if T<:Real
                @test Matrix(pinv(Symmetric(FaerMatrix(hdata), side)))≈expected rtol=tol atol=tol
            end
        end
        v=view(a, 3:-1:1, 3:-1:1)
        @test Matrix(pinv(v))≈pinv(Matrix(v)) rtol=tol atol=tol
        @test rank(v)==rank(Matrix(v))
        @test cond(v)==cond(Matrix(v))
        bad=FaerMatrix{T}(undef, 2, 2)
        bad[1, 1]=T(NaN)
        @test_throws Exception pinv(bad)
    end
    @test isempty(detect_ambiguities(Colmat, Base, LinearAlgebra; recursive = true))
end

@testset "Empty operator norm respects owner lifecycle" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        a=FaerMatrix(zeros(T, 0, 0))
        close(a)
        for p in (1, 2, Inf)
            @test_throws ArgumentError opnorm(a, p)
        end
    end
end
