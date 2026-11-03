using Test, LinearAlgebra, Random, Colmat

function qz_spectrum_matches(actual, expected, tol)
    @test length(actual)==length(expected)
    left=collect(expected)
    for value in actual
        index=argmin(abs.(left .- value))
        @test value≈left[index] rtol=tol atol=tol
        deleteat!(left, index)
    end
end

@testset "Native generalized Hessenberg and QZ" begin
    rng=MersenneTwister(530)
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        tol=typeof(real(zero(T)))===Float32 ? 2e-3 : 5e-10
        for n in (0, 1, 2, 5, 12, 35)
            a=randn(rng, T, n, n)
            b=randn(rng, T, n, n)+T(3)*I
            aa, bb=FaerMatrix(a), FaerMatrix(b)
            h=hessenberg(aa, bb)
            @test h isa FaerGeneralizedHessenberg
            @test size(h)==(n, n)
            @test eltype(h.H)===T
            @test Matrix(h.H)==triu(Matrix(h.H), -1)
            @test Matrix(h.T)==triu(Matrix(h.T))
            @test Matrix(h.Q*h.H*h.Z')≈a rtol=tol atol=tol
            @test Matrix(h.Q*h.T*h.Z')≈b rtol=tol atol=tol
            for m in (h.Q, h.Z)
                @test Matrix(m)'*Matrix(m)≈Matrix{T}(I, n, n) rtol=tol atol=tol
            end
            hh, tt, qq, zz=h
            @test hh===h.H && tt===h.T && qq===h.Q && zz===h.Z
            for g in (copy(h), deepcopy(h))
                @test Matrix(g.Q*g.H*g.Z')≈a rtol=tol atol=tol
                @test Matrix(g.Q*g.T*g.Z')≈b rtol=tol atol=tol
            end
            saved=map(Matrix, (h.H, h.T, h.Q, h.Z))
            for f in (schur(aa, bb), schur(h))
                @test f isa GeneralizedSchur
                @test size(f)==(n, n)
                @test f.S isa FaerMatrix{T}
                @test f.T isa FaerMatrix{T}
                @test eltype(f.α)===Complex{typeof(real(zero(T)))}
                @test eltype(f.β)===T
                @test Matrix(f.Q*f.S*f.Z')≈a rtol=tol atol=tol
                @test Matrix(f.Q*f.T*f.Z')≈b rtol=tol atol=tol
                @test Matrix(f.S)==triu(Matrix(f.S), T<:Real ? -1 : 0)
                @test Matrix(f.T)==triu(Matrix(f.T))
                for m in (f.Q, f.Z)
                    @test Matrix(m)'*Matrix(m)≈Matrix{T}(I, n, n) rtol=tol atol=tol
                end
                qz_spectrum_matches(f.values, eigvals(a, b), tol)
                @test eigvals(f)==f.values
                @test f.left===f.Q && f.right===f.Z
                @test map(Matrix, (h.H, h.T, h.Q, h.Z))==saved
                for g in (copy(f), deepcopy(f))
                    @test Matrix(g.Q*g.S*g.Z')≈a rtol=tol atol=tol
                    @test Matrix(g.Q*g.T*g.Z')≈b rtol=tol atol=tol
                end
                @test Matrix(aa)==a && Matrix(bb)==b
                close(f)
                @test !isopen(f)
                @test_throws ArgumentError eigvals(f)
                @test_throws ArgumentError copy(f)
            end
            close(h)
            @test !isopen(h)
            @test_throws ArgumentError schur(h)
            close(aa)
            @test_throws ArgumentError schur(aa, bb)
        end
    end
end

@testset "Homogeneous generalized eigenvalues" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        a=Matrix(Diagonal(T[1, 2, 0, 0]))
        b=Matrix(Diagonal(T[1, 0, 1, 0]))
        f=schur(FaerMatrix(a), FaerMatrix(b))
        @test Matrix(f.Q*f.S*f.Z')≈a
        @test Matrix(f.Q*f.T*f.Z')≈b
        @test all(isfinite, f.α) && all(isfinite, f.β)
        @test count(i->iszero(f.β[i])&&!iszero(f.α[i]), 1:4)==1
        @test count(i->iszero(f.α[i])&&iszero(f.β[i]), 1:4)==1
        @test count(i->!iszero(f.β[i])&&iszero(f.α[i]), 1:4)==1
        @test count(i->!iszero(f.β[i])&&isapprox(f.α[i]/f.β[i], 1), 1:4)==1
        @test isequal(eigvals(f), f.values)
        for n in (1, 2, 4)
            z=FaerMatrix(zeros(T, n, n))
            f=schur(z, z)
            @test all(iszero, f.α) && all(iszero, f.β)
            @test all(isnan, f.values)
        end
    end
end

@testset "Mutating and mixed QZ interfaces" begin
    rng=MersenneTwister(749)
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        tol=typeof(real(zero(T)))===Float32 ? 2e-3 : 5e-10
        a, b=randn(rng, T, 6, 6), randn(rng, T, 6, 6)
        aa, bb=FaerMatrix(a), FaerMatrix(b)
        av, bv=view(aa, :, :), view(bb, :, :)
        f=schur!(aa, bb)
        @test f.S===aa && f.T===bb
        @test Matrix(av)==Matrix(f.S) && Matrix(bv)==Matrix(f.T)
        @test Matrix(f.Q*f.S*f.Z')≈a rtol=tol atol=tol
        @test Matrix(f.Q*f.T*f.Z')≈b rtol=tol atol=tol
        aa, bb=FaerMatrix(a), FaerMatrix(b)
        av, bv=view(aa, :, :), view(bb, :, :)
        h=hessenberg!(aa, bb)
        @test h.H===aa && h.T===bb
        @test Matrix(av)==Matrix(h.H) && Matrix(bv)==Matrix(h.T)
        qv, zv=view(h.Q, :, :), view(h.Z, :, :)
        f=schur!(h)
        @test f.S===h.H && f.T===h.T && f.Q===h.Q && f.Z===h.Z
        @test Matrix(av)==Matrix(f.S) && Matrix(bv)==Matrix(f.T)
        @test Matrix(qv)==Matrix(f.Q) && Matrix(zv)==Matrix(f.Z)
        @test Matrix(f.Q*f.S*f.Z')≈a rtol=tol atol=tol
        @test Matrix(f.Q*f.T*f.Z')≈b rtol=tol atol=tol
        f=schur(view(FaerMatrix(a), 6:-1:1, :), view(FaerMatrix(b), 6:-1:1, :))
        @test Matrix(f.Q*f.S*f.Z')≈a[6:-1:1, :] rtol=tol atol=tol
        @test Matrix(f.Q*f.T*f.Z')≈b[6:-1:1, :] rtol=tol atol=tol
        @test_throws ArgumentError schur!(aa, aa)
        @test_throws ArgumentError hessenberg!(aa, aa)
        bad=hessenberg(FaerMatrix(a), FaerMatrix(b))
        bad.T[4, 1]=T(1)
        saved=map(Matrix, (bad.H, bad.T, bad.Q, bad.Z))
        @test_throws Exception schur!(bad)
        @test map(Matrix, (bad.H, bad.T, bad.Q, bad.Z))==saved
    end
    for (T, S) in ((Float32, Float64), (Float64, ComplexF32), (ComplexF32, ComplexF64))
        a, b=randn(rng, T, 4, 4), randn(rng, S, 4, 4)
        f=schur(FaerMatrix(a), FaerMatrix(b))
        @test eltype(f)===promote_type(T, S)
        @test Matrix(f.Q*f.S*f.Z')≈a rtol=2e-3
        @test Matrix(f.Q*f.T*f.Z')≈b rtol=2e-3
        @test_throws ArgumentError schur!(FaerMatrix(a), FaerMatrix(b))
        for f in (schur(FaerMatrix(a), b), schur(a, FaerMatrix(b)))
            @test Matrix(f.Q*f.S*f.Z')≈a rtol=2e-3
        end
    end
    @test_throws DimensionMismatch schur(FaerMatrix(ones(2, 3)), FaerMatrix(ones(2, 3)))
    @test_throws DimensionMismatch schur(FaerMatrix(ones(2, 2)), FaerMatrix(ones(3, 3)))
    @test isempty(detect_ambiguities(Colmat, Base, LinearAlgebra; recursive = true))
end

@testset "QZ mixed view intersections and scaled pencils" begin
    rng=MersenneTwister(792)
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        R=typeof(real(zero(T)))
        tol=R===Float32 ? 3e-3 : 2e-9
        a=T[0.1 0.2 -0.3; 0.4 -0.2 0.1; 0.1 0.3 0.2]
        b=T[1.1 0.1 -0.1; 0 0.9 0.1; 0.1 0 1.2]
        T<:Complex && (a .+= im .* T[0.1 0.2 0; 0.3 0.1 0.2; 0.1 0 0.2])
        aa, bb=FaerMatrix(a), FaerMatrix(b)
        for f in (schur(view(aa, :, :), bb), schur(aa, view(bb, :, :)))
            @test Matrix(f.Q*f.S*f.Z')≈a rtol=tol atol=tol
            @test Matrix(f.Q*f.T*f.Z')≈b rtol=tol atol=tol
        end
        scales=R===Float32 ? (R(1e-25), R(1e25)) : (R(1e-250), R(1e250))
        for ascale in scales, bscale in scales
            f=schur(FaerMatrix(a .* ascale), FaerMatrix(b .* bscale))
            @test Matrix(f.Q*f.S*f.Z') ./ ascale≈a rtol=tol atol=tol
            @test Matrix(f.Q*f.T*f.Z') ./ bscale≈b rtol=tol atol=tol
            @test all(isfinite, f.α)&&all(isfinite, f.β)
            qz_spectrum_matches((f.α ./ ascale) ./ (f.β ./ bscale), eigvals(a, b), tol)
        end
        bad=FaerMatrix{T}(undef, 2, 2)
        bad[1, 1]=T(NaN)
        other=FaerMatrix{T}(I, 2, 2)
        snapshot=Matrix(bad)
        @test_throws Exception schur!(bad, other)
        @test isequal(Matrix(bad), snapshot)
        @test Matrix(other)==Matrix{T}(I, 2, 2)
    end
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        S in (Float32, Float64, ComplexF32, ComplexF64)

        a, b=randn(rng, T, 3, 3), randn(rng, S, 3, 3)
        f=schur(FaerMatrix(a), FaerMatrix(b))
        @test eltype(f)===promote_type(T, S)
        @test Matrix(f.Q*f.S*f.Z')≈a rtol=3e-3 atol=3e-3
        @test Matrix(f.Q*f.T*f.Z')≈b rtol=3e-3 atol=3e-3
    end
    @test isempty(detect_ambiguities(Colmat, Base, LinearAlgebra; recursive = true))
end

@testset "Blocked generalized Hessenberg and QZ" begin
    rng=MersenneTwister(941)
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        tol=typeof(real(zero(T)))===Float32 ? 5e-3 : 5e-9
        n=96
        a, b=randn(rng, T, n, n), randn(rng, T, n, n)
        f=schur(FaerMatrix(a), FaerMatrix(b))
        @test Matrix(f.Q*f.S*f.Z')≈a rtol=tol atol=tol
        @test Matrix(f.Q*f.T*f.Z')≈b rtol=tol atol=tol
        @test Matrix(f.Q)'*Matrix(f.Q)≈Matrix{T}(I, n, n) rtol=tol atol=tol
        @test Matrix(f.Z)'*Matrix(f.Z)≈Matrix{T}(I, n, n) rtol=tol atol=tol
        @test Matrix(f.S)==triu(Matrix(f.S), T<:Real ? -1 : 0)
        @test Matrix(f.T)==triu(Matrix(f.T))
    end
    for T in (Float64, ComplexF64)
        n=257
        a, b=randn(rng, T, n, n), randn(rng, T, n, n)
        h=hessenberg(FaerMatrix(a), FaerMatrix(b))
        @test Matrix(h.Q*h.H*h.Z')≈a rtol=5e-9 atol=5e-9
        @test Matrix(h.Q*h.T*h.Z')≈b rtol=5e-9 atol=5e-9
        @test Matrix(h.H)==triu(Matrix(h.H), -1)
        @test Matrix(h.T)==triu(Matrix(h.T))
    end
end

@testset "Real QZ rounded-window scratch regression" begin
    rng=MersenneTwister(922)
    for T in (Float32, Float64), n in (78, 258)
        tol=T===Float32 ? 5e-3 : 5e-9
        a, b=randn(rng, T, n, n), randn(rng, T, n, n)
        f=schur(FaerMatrix(a), FaerMatrix(b))
        @test Matrix(f.Q*f.S*f.Z')≈a rtol=tol atol=tol
        @test Matrix(f.Q*f.T*f.Z')≈b rtol=tol atol=tol
    end
end
