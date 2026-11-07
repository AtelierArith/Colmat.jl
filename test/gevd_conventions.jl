using Colmat, Test, LinearAlgebra, Random

const GC_TYPES=(Float32, Float64, ComplexF32, ComplexF64)

@testset "Generalized eigenvalue quotient conventions" begin
    for T in GC_TYPES
        A=T[1 2; 0 3]
        B=T[1 0; 0 0]  # one finite eigenvalue (1) and one infinite one
        v=eigvals(FaerMatrix(A), FaerMatrix(B))
        @test v[1]≈1
        if T<:Real
            # Real pencils: the real infinite eigenvalue is Inf+0im (LAPACK: real Inf).
            @test v[2]===complex(T(Inf))
            @test real.(v)==eigvals(A, B)
        else
            # Complex pencils keep Julia's complex quotient, as LinearAlgebra does.
            @test isnan(real(v[2])) && isnan(imag(v[2]))
            @test isequal(v, eigvals(A, B))
        end
        h=eigvals(FaerMatrix(A), FaerMatrix(B); homogeneous = true)
        @test iszero(h.β[2]) && !iszero(h.α[2])
        E=eigen(FaerMatrix(A), FaerMatrix(B))
        @test isequal(E.values, v)
        # Eigenvectors have unit 2-norm (LAPACK ggev scales max |re|+|im| to 1).
        @test all(c->norm(c)≈1, eachcol(E.vectors))
    end
    # A real pencil with a complex pair keeps both nonzero imaginary parts.
    v=eigvals(FaerMatrix([0.0 -1; 1 0]), FaerMatrix([2.0 0; 0 1]))
    @test v≈[-im/sqrt(2), im/sqrt(2)]
end

@testset "Graded and larger generalized pencils" begin
    rng=MersenneTwister(11)
    for T in (Float64, ComplexF64), n in (40, 120)
        A0=randn(rng, T, n, n)
        B0=randn(rng, T, n, n)+n*I
        ref=eigvals(A0, B0)
        # Row and column grading leave the eigenvalues unchanged.
        for g in (1e-3, 1e3)
            D1=Diagonal(T.(g .^ range(-1, 1; length = n)))
            D2=Diagonal(T.(g .^ range(1, -1; length = n)))
            A=D1*A0*D2
            B=D1*B0*D2
            v=eigvals(FaerMatrix(A), FaerMatrix(B))
            @test sort(abs.(v))≈sort(abs.(ref)) rtol = 1e-6
        end
        E=eigen(FaerMatrix(A0), FaerMatrix(B0))
        V=E.vectors
        resid=norm(A0*V-B0*V*Diagonal(E.values))/(norm(A0)+norm(B0)*maximum(abs, E.values))
        @test resid<1e3*n*eps(real(T))
    end
end
