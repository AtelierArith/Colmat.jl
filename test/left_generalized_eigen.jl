using Test, Colmat, LinearAlgebra, SparseArrays
@testset "Left generalized eigenvectors" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), n in (0, 1, 4), width in (1, 2)
        A=Matrix{T}(Diagonal(T.(1:n)))
        B=Matrix{T}(I, n, n)
        if n>1
            A[1, 2]=-3
            A[2, 1]=2
            B[1, 1]=2
            B[1, 2]=T(0.2)
        end
        T<:Complex && (A .*= T(1+0.3im))
        a, b=FaerMatrix(A), FaerMatrix(B)
        FaerTeam(width) do team
            s=FaerScratch(eigen, a, b; team)
            F=eigen(a, b; side = :left, scratch = s, team)
            @test F.vectors'*A≈Diagonal(F.values)*F.vectors'*B atol=1e-3
            @test all(v->isapprox(norm(v), 1; atol = 1e-4), eachcol(F.vectors))
            H=eigen(a, b; side = :left, homogeneous = true, scratch = s, team)
            @test Diagonal(H.β)*H.vectors'*A≈Diagonal(H.α)*H.vectors'*B atol=1e-3
            G=eigen(
                a,
                b;
                side = :left,
                homogeneous = true,
                scratch = s,
                team,
                sortby = x->-real(x),
            )
            @test issorted(real.(G.α ./ G.β); rev = true)
            @test Diagonal(G.β)*G.vectors'*A≈Diagonal(G.α)*G.vectors'*B atol=1e-3
            @test Matrix(a)==A
            @test Matrix(b)==B
            close(s)
        end
    end
end
@testset "Left repeated and infinite eigenspaces" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        for (A, B) in (
            (Matrix{T}(I, 3, 3), Matrix{T}(I, 3, 3)),
            (Matrix{T}(Diagonal(T[2, 3, 4])), Matrix{T}(Diagonal(T[1, 0, 0]))),
        )
            a, b=FaerMatrix(A), FaerMatrix(B)
            FaerTeam(2) do team
                F=eigen(a, b; side = :left, homogeneous = true, team)
                @test all(isfinite, F.vectors)
                @test Diagonal(F.β)*F.vectors'*A≈Diagonal(F.α)*F.vectors'*B atol=1e-3
                @test F.vectors'*F.vectors≈I atol=1e-4
            end
        end
    end
end
@testset "Left promotion, validation and larger spectrum" begin
    a, b=FaerMatrix(Float32[1 -2; 3 1]), FaerMatrix([2.0 0; 0 1])
    F=eigen(a, b; side = :left)
    @test eltype(F.values)==ComplexF64
    @test F.vectors'*Matrix(a)≈Diagonal(F.values)*F.vectors'*Matrix(b)
    @test_throws ArgumentError eigen(a, b; side = :both)
    close(a)
    @test_throws ArgumentError eigen(a, b; side = :left)
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        n=64
        C=Matrix{T}(SymTridiagonal(fill(2.0, n), fill(-1.0, n-1)))
        X=reshape(T.(1:(n*n)), n, n)/(n*n)
        T<:Complex && (X .+= T(0.1im))
        B=X*X'+Matrix{T}(I, n, n)
        A=B*C
        a, b=FaerMatrix(A), FaerMatrix(B)
        FaerTeam(2) do team
            s=FaerScratch(eigen, a, b; team)
            F=eigen(a, b; side = :left, scratch = s, team)
            @test real.(F.values)≈[2-2cos(k*pi/(n+1)) for k = 1:n] atol=3e-3
            @test F.vectors'*A≈Diagonal(F.values)*F.vectors'*B rtol=1e-3 atol=1e-3
            close(s)
        end
    end
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
