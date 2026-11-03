using Test, Colmat, LinearAlgebra, SparseArrays
@testset "Homogeneous generalized eigensystems" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), n in (0, 1, 4), width in (1, 2)
        A=Matrix{T}(Diagonal(T.(1:n)))
        B=Matrix{T}(I, n, n)
        if n>1
            A[1, 2]=-2
            A[2, 1]=3
            B[1, 1]=2
        end
        T<:Complex && (A .*= T(1+0.2im))
        a, b=FaerMatrix(A), FaerMatrix(B)
        FaerTeam(width) do team
            s=FaerScratch(eigen, a, b; team)
            F=eigen(a, b; homogeneous = true, scratch = s, team)
            @test keys(F)==(:α, :β, :vectors)
            @test eltype(F.α)==Complex{typeof(real(zero(T)))}
            @test eltype(F.β)==T
            @test size(F.vectors)==(n, n)
            @test A*F.vectors*Diagonal(F.β)≈B*F.vectors*Diagonal(F.α) atol=1e-3
            V=eigvals(a, b; homogeneous = true, scratch = s, team)
            @test keys(V)==(:α, :β)
            @test V.α≈F.α atol=1e-3
            @test V.β≈F.β atol=1e-3
            G=eigen(a, b; homogeneous = true, scratch = s, team, sortby = x->-real(x))
            @test issorted(real.(G.α ./ G.β); rev = true)
            @test A*G.vectors*Diagonal(G.β)≈B*G.vectors*Diagonal(G.α) atol=1e-3
            @test Matrix(a)==A
            @test Matrix(b)==B
            close(s)
        end
    end
end
@testset "Infinite, repeated and indeterminate homogeneous values" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        A=Matrix{T}(Diagonal(T[2, 3, 4]))
        for B in (Matrix{T}(Diagonal(T[1, 0, 0])), zeros(T, 3, 3))
            a, b=FaerMatrix(A), FaerMatrix(B)
            FaerTeam(2) do team
                F=eigen(a, b; homogeneous = true, team)
                @test all(isfinite, F.α)
                @test all(isfinite, F.β)
                @test all(isfinite, F.vectors)
                @test count(iszero, F.β)==count(iszero, diag(B))
                @test all(x->!iszero(x), F.α)
                @test A*F.vectors*Diagonal(F.β)≈B*F.vectors*Diagonal(F.α) atol=1e-3
                @test all(v->isapprox(norm(v), 1; atol = 1e-4), eachcol(F.vectors))
            end
        end
        a, b=FaerMatrix(zeros(T, 2, 2)), FaerMatrix(zeros(T, 2, 2))
        F=eigvals(a, b; homogeneous = true)
        @test all(iszero, F.α)
        @test all(iszero, F.β)
        @test all(isnan, eigvals(a, b))
        a, b=FaerMatrix(Matrix{T}(I, 3, 3)), FaerMatrix(Matrix{T}(I, 3, 3))
        F=eigen(a, b; homogeneous = true)
        @test all(isfinite, F.vectors)
        @test F.vectors'*F.vectors≈I atol=1e-4
        @test F.α≈F.β atol=1e-4
    end
end
@testset "Homogeneous promotion and owner errors" begin
    a, b=FaerMatrix(Float32[2 0; 0 3]), FaerMatrix([1.0 0; 0 2])
    F=eigen(a, b; homogeneous = true)
    @test eltype(F.α)==ComplexF64
    @test eltype(F.β)==Float64
    @test sort(real.(F.α ./ F.β))≈[1.5, 2]
    close(a)
    @test_throws ArgumentError eigvals(a, b; homogeneous = true)
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
