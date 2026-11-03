using Test, Colmat, LinearAlgebra

@testset "Retained spectral solves" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        R in (Float32, Float64, ComplexF32, ComplexF64),
        dims in ((3, 2), (2, 3), (2, 2)),
        full in (false, true)

        m, n = dims
        A = reshape(T.(1:(m*n)), m, n)
        T <: Complex && (A .+= T(0.3im))
        a = FaerMatrix(A)
        F = svd(a; native = true, full)
        close(a)
        for (G, M) in ((F, A), (transpose(F), transpose(A)), (adjoint(F), adjoint(A)))
            p, q = size(M)
            B = reshape(R.(1:(p*2)), p, 2)
            C = reshape(R.(1:(2*q)), 2, q)
            R <: Complex && (B .+= R(0.4im); C .+= R(0.2im))
            expected = pinv(M) * B
            X = G \ B
            @test X isa FaerMatrix{promote_type(T, R)}
            @test Matrix(X) ≈ expected atol=3e-3 rtol=3e-3
            @test G \ B[:, 1] ≈ expected[:, 1] atol=3e-3 rtol=3e-3
            Y = C / G
            @test Matrix(Y) ≈ C * pinv(M) atol=3e-3 rtol=3e-3
            dest = zeros(promote_type(T, R), q, 2)
            @test ldiv!(dest, G, B) === dest
            @test dest ≈ expected atol=3e-3 rtol=3e-3
            out = zeros(promote_type(T, R), 2, p)
            @test rdiv!(out, C, G) === out
            @test out ≈ C * pinv(M) atol=3e-3 rtol=3e-3
            close(X)
            close(Y)
        end
        close(F)
    end
end

@testset "Spectral clipping, inverse and staged writes" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), kind in (:svd, :eigen)
        A = Matrix{T}(Diagonal(T[2, -0.01, 0]))
        a = FaerMatrix(A)
        F = kind === :svd ? svd(a; native = true) : eigen(Hermitian(a); native = true)
        FaerTeam(2) do team
            X = (\)(F, Matrix{T}(I, 3, 3); atol = 0.1, team)
            @test Matrix(X) ≈ Diagonal(T[0.5, 0, 0])
            close(X)
        end
        @test_throws SingularException inv(F)
        b = T[2, 3, 4]
        @test ldiv!(F, b) === b
        @test b ≈ T[1, -300, 0] rtol=1e-4
        @test_throws ArgumentError (\)(F, ones(T, 3); atol = -1)
        @test_throws DimensionMismatch F \ ones(T, 2)
        close(F)
        @test_throws ArgumentError F \ ones(T, 3)
        close(a)
        A = T[4 1; 1 3]
        a = FaerMatrix(A)
        F = kind === :svd ? svd(a; native = true) : eigen(Hermitian(a); native = true)
        for G in (F, transpose(F), adjoint(F))
            X = inv(G)
            @test Matrix(X) ≈ inv(A) rtol=1e-4
            close(X)
        end
        B = Matrix{T}(I, 2, 2)
        @test rdiv!(B, F) === B
        @test B ≈ inv(A) rtol=1e-4
        close(F)
        close(a)
    end
    a = FaerMatrix([2.0 0; 0 1])
    F = svd(a; native = true)
    dest = fill(7.0, 2, 1)
    @test_throws InexactError ldiv!(dest, F, fill(1.0im, 2, 1))
    @test dest == fill(7.0, 2, 1)
    B = FaerMatrix([2.0 4; 3 5])
    expected = [1.0 2; 3 5]
    @test ldiv!(B, F, B) === B
    @test Matrix(B) ≈ expected
    close(B)
    close(F)
    close(a)
end

@testset "Empty and rectangular spectral solves" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), (m, n) in ((0, 0), (0, 3), (3, 0))
        a=FaerMatrix(zeros(T, m, n))
        F=svd(a; native = true)
        X=F\zeros(T, m, 2)
        Y=zeros(T, 2, n)/F
        @test size(X)==(n, 2)
        @test Matrix(X)==zeros(T, n, 2)
        @test size(Y)==(2, m)
        @test Matrix(Y)==zeros(T, 2, m)
        if m!=n
            @test_throws DimensionMismatch ldiv!(F, zeros(T, m, 2))
            @test_throws DimensionMismatch rdiv!(zeros(T, 2, n), F)
            @test_throws DimensionMismatch inv(F)
        end
        close(X)
        close(Y)
        close(F)
        close(a)
    end
end
