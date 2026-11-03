using Test, Colmat, LinearAlgebra, SparseArrays, Random

@testset "Native matrix balancing" begin
    rng=MersenneTwister(29)
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        R=typeof(real(zero(T)))
        tol=1024eps(R)
        for n in (0, 1, 2, 5, 17)
            raw=randn(rng, T, n, n)
            powers=R.(exp2.(range(-8, 8; length = max(n, 2)))[1:n])
            A=raw .* reshape(powers, 1, n) ./ reshape(powers, n, 1)
            f=balance(A)
            @test f isa FaerBalance
            @test eltype(f.B)===T
            @test sort(f.perm)==collect(1:n)
            @test all(x->isfinite(x)&&x>0, f.scale)
            @test all(x->isinteger(log2(x)), f.scale)
            @test Matrix(f.B) ≈
                  A[f.perm, f.perm] .* reshape(f.scale, 1, n) ./ reshape(f.scale, n, 1) rtol=tol
            @test 1<=first(f.active)<=n+1
            @test 0<=last(f.active)<=n
            @test first(f.active)<=last(f.active)+1
            @test isopen(f)
            copied=copy(f)
            @test copied.B !== f.B
            @test copied.perm !== f.perm
            @test copied.scale !== f.scale
            @test Matrix(copied.B)==Matrix(f.B)
            vectors=randn(rng, T, n, 3)
            expected=similar(vectors)
            expected[f.perm, :]=f.scale .* vectors
            @test Matrix(balance_eigenvectors(f, vectors)) ≈ expected rtol=tol
            target=FaerMatrix(vectors)
            @test balance_eigenvectors!(f, target)===target
            @test Matrix(target) ≈ expected rtol=tol
            host=copy(vectors)
            @test balance_eigenvectors!(f, host)===host
            @test host ≈ expected rtol=tol
            v=randn(rng, T, n)
            expectedv=similar(v)
            expectedv[f.perm]=f.scale .* v
            @test balance_eigenvectors(f, v) ≈ expectedv rtol=tol
            nativev=FaerVector(v)
            @test balance_eigenvectors(f, nativev) isa FaerVector{T}
            @test Vector(balance_eigenvectors(f, nativev)) ≈ expectedv rtol=tol
            source=FaerMatrix(A)
            result=balance!(source)
            @test result.B===source
            @test Matrix(source)==Matrix(f.B)
            @test result.perm==f.perm
            if n>0
                E=eigen(f.B)
                X=Matrix(balance_eigenvectors(f, E.vectors))
                @test norm(A*X-X*Diagonal(E.values)) <= tol*max(norm(A)*norm(X), one(R))
            end
            close(f)
            @test !isopen(f)
            @test_throws ArgumentError balance_eigenvectors(f, vectors)
            @test isopen(copied)
            close(copied)
        end
        A=T[1 0 1024 0; 0 7 0 0; 1/1024 0 2 1; 0 0 0 8]
        f=balance(A)
        @test Matrix(f.B) ≈
              A[f.perm, f.perm] .* reshape(f.scale, 1, 4) ./ reshape(f.scale, 4, 1)
        @test length(f.active)<4
        @test_throws DimensionMismatch balance(zeros(T, 2, 3))
        dest=FaerMatrix(zeros(T, 2, 2))
        dest[1, 1]=T(NaN)
        before=Matrix(dest)
        @test_throws ArgumentError balance!(dest)
        @test isequal(Matrix(dest), before)
        @test_throws DimensionMismatch balance_eigenvectors(f, zeros(T, 3, 2))
        invalid=FaerBalance(copy(f.B), fill(1, 4), copy(f.scale), f.active)
        output=ones(T, 4, 2)
        saved=copy(output)
        @test_throws ArgumentError balance_eigenvectors!(invalid, output)
        @test output==saved
        badscale=FaerBalance(copy(f.B), copy(f.perm), zeros(R, 4), f.active)
        @test_throws ArgumentError balance_eigenvectors(badscale, output)
        parent=FaerMatrix(randn(rng, T, 6, 3))
        region=view(parent, 2:5, 1:2)
        expected=Matrix(balance_eigenvectors(f, Matrix(region)))
        @test balance_eigenvectors!(f, region)===region
        @test Matrix(region) ≈ expected
        close(parent)
        @test_throws ArgumentError balance_eigenvectors(f, region)
    end
    f=balance([1.0 1024; 1/1024 2])
    @test maximum(f.scale)>minimum(f.scale)
    X=ComplexF64[1+im 2-im; 3-im 4+im]
    Y=Matrix(balance_eigenvectors(f, X))
    @test Y[f.perm, :] ≈ f.scale .* X
    @test eltype(f)===Float64
    @test eltype(typeof(f))===Float64
    extreme=FaerBalance(
        FaerMatrix(Matrix{Float64}(I, 2, 2)),
        [1, 2],
        fill(exp2(1000.0), 2),
        1:2,
    )
    output=fill(exp2(1000.0), 2, 1)
    saved=copy(output)
    @test_throws ArgumentError balance_eigenvectors!(extreme, output)
    @test output==saved
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
