using Test, Colmat, LinearAlgebra, SparseArrays, Random

@testset "Native bidiagonal reduction and SVD" begin
    rng=MersenneTwister(123)
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        R=typeof(real(zero(T)))
        tol=4096eps(R)
        for (m, n) in (
            (0, 0),
            (0, 3),
            (3, 0),
            (1, 1),
            (4, 1),
            (1, 4),
            (2, 2),
            (5, 3),
            (3, 5),
            (9, 9),
            (131, 129),
        )
            A=randn(rng, T, m, n)
            k=min(m, n)
            for full in (false, true)
                f=bidiagonalize(A; full)
                @test f isa FaerBidiagonalization
                @test size(f)==(m, n)
                @test size(f.U)==(m, full ? m : k)
                @test size(f.V)==(n, full ? n : k)
                @test f.B.uplo==(m<n ? 'L' : 'U')
                @test Matrix(f) ≈ A rtol=tol atol=tol
                U=Matrix(f.U)
                V=Matrix(f.V)
                @test U'*U ≈ Matrix{T}(I, size(U, 2), size(U, 2)) atol=tol
                @test V'*V ≈ Matrix{T}(I, size(V, 2), size(V, 2)) atol=tol
                E=svd(f)
                @test E.S ≈ svdvals(A) rtol=tol atol=tol
                @test Matrix(E.U)*Diagonal(E.S)*Matrix(E.Vt) ≈ A rtol=tol atol=tol
                @test svdvals(f) ≈ E.S rtol=tol atol=tol
                if full
                    EF=svd(f; full = true)
                    @test size(EF.U)==(m, m)
                    @test size(EF.Vt)==(n, n)
                    @test Matrix(EF.U)[:, 1:k]*Diagonal(EF.S)*Matrix(EF.Vt)[1:k, :] ≈ A rtol=tol atol=tol
                elseif m!=n
                    @test_throws ArgumentError svd(f; full = true)
                end
                copied=copy(f)
                @test copied.U !== f.U
                @test copied.V !== f.V
                @test copied.B.dv !== f.B.dv
                close(f)
                @test !isopen(f)
                @test isopen(copied)
                @test_throws ArgumentError svd(f)
            end
            native=FaerMatrix(A)
            f=bidiagonalize!(native)
            expected=zeros(T, m, n)
            expected[1:k, 1:k]=Matrix(f.B)
            @test Matrix(native)==expected
            @test Matrix(f) ≈ A rtol=tol atol=tol
        end
        for n in (0, 1, 2, 7, 33, 130), uplo in (:U, :L)
            d=randn(rng, T, n)
            e=randn(rng, T, max(0, n-1))
            saved=copy(d)
            A=Matrix(Bidiagonal(d, e, uplo))
            for threshold in (16, 256)
                F=bidiagonal_svd(d, e; uplo, recursion_threshold = threshold)
                @test F.S ≈ svdvals(A) rtol=tol atol=tol
                @test issorted(F.S; rev = true)
                @test all(>=(0), F.S)
                @test eltype(F.S)===R
                @test Matrix(F.U)*Diagonal(F.S)*Matrix(F.Vt) ≈ A rtol=tol atol=tol
                @test Matrix(F.U)'*Matrix(F.U) ≈ Matrix{T}(I, n, n) atol=tol
                @test Matrix(F.Vt)*Matrix(F.Vt)' ≈ Matrix{T}(I, n, n) atol=tol
                @test d==saved
            end
            native=Bidiagonal(FaerVector(d), FaerVector(e), uplo)
            @test svd(native).S ≈ svdvals(A) rtol=tol atol=tol
            @test svdvals(native) ≈ svdvals(A) rtol=tol atol=tol
        end
        for bs in (1, 2)
            A=randn(rng, T, 6, 4)
            @test Matrix(bidiagonalize(A; blocksize = bs)) ≈ A rtol=tol atol=tol
        end
        for scale in (sqrt(floatmin(R)), sqrt(floatmax(R))/R(100))
            base=T[1 2; 3 4; 5 6]
            @test Matrix(bidiagonalize(scale .* base)) ./ scale ≈ base rtol=tol atol=tol
            small=bidiagonal_svd(T[1, 2] .* scale, T[0.2] .* scale)
            @test small.S ./ scale ≈ svdvals(Matrix(Bidiagonal(R[1, 2], R[0.2], :U))) rtol=tol atol=tol
        end
        @test_throws ArgumentError bidiagonalize(ones(T, 2, 2); blocksize = -1)
        @test_throws DimensionMismatch bidiagonal_svd(T[1, 2], T[])
        @test_throws ArgumentError bidiagonal_svd(T[1], T[]; uplo = :bad)
        @test_throws ArgumentError bidiagonal_svd(T[1], T[]; recursion_threshold = 1)
        @test_throws ArgumentError bidiagonal_svd(T[NaN], T[])
        native=FaerMatrix(zeros(T, 2, 3))
        native[1, 1]=T(NaN)
        saved=Matrix(native)
        @test_throws ArgumentError bidiagonalize!(native)
        @test isequal(Matrix(native), saved)
    end
    A=reshape(collect(1.0:40_000), 20_000, 2)
    f=bidiagonalize(A)
    @test size(f.U)==(20_000, 2)
    @test Matrix(f) ≈ A
    parent=FaerMatrix(reshape(collect(1.0:30), 6, 5))
    region=view(parent, 2:5, 2:4)
    @test Matrix(bidiagonalize(region)) ≈ Matrix(region)
    close(parent)
    @test_throws ArgumentError bidiagonalize(region)
    @test isempty(
        Test.detect_ambiguities(
            Colmat,
            Base,
            LinearAlgebra,
            SparseArrays;
            recursive = true,
        ),
    )
end
