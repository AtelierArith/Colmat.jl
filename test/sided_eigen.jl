using Test, Colmat, LinearAlgebra, SparseArrays
@testset "Single-matrix left and simultaneous eigenvectors" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        n in (0, 1, 4),
        side in (:left, :both),
        width in (1, 2)

        A=Matrix{T}(Diagonal(T.(1:n)))
        n>1 && (A[1, 2] = -3; A[2, 1] = 2)
        n>2 && (A[1, 3]=T(0.3))
        T<:Complex && (A .*= T(1+0.3im))
        a=FaerMatrix(A)
        FaerTeam(width) do team
            s=FaerScratch(eigen, a; side, team)
            @test scratch_capacity(s)==scratch_requirements(eigen, a; side, team)
            F=eigen(a; side, scratch = s, team)
            side===:left && @test keys(F)==(:values, :vectors)
            left=side===:left ? F.vectors : F.left
            @test left'*A≈Diagonal(F.values)*left' atol=1e-3
            @test all(v->isapprox(norm(v), 1; atol = 1e-4), eachcol(left))
            if side===:both
                @test keys(F)==(:values, :left, :right)
                @test A*F.right≈F.right*Diagonal(F.values) atol=1e-3
                @test all(v->isapprox(norm(v), 1; atol = 1e-4), eachcol(F.right))
            end
            G=eigen(a; side, scratch = s, team, sortby = x->-real(x))
            @test issorted(real.(G.values); rev = true)
            left=side===:left ? G.vectors : G.left
            @test left'*A≈Diagonal(G.values)*left' atol=1e-3
            side===:both && @test A*G.right≈G.right*Diagonal(G.values) atol=1e-3
            @test Matrix(a)==A
            close(s)
        end
    end
end
@testset "Larger sided EVD and guards" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), side in (:left, :both)
        n=64
        # Diagonal similarity preserves an independently known tridiagonal spectrum.
        C=Matrix{T}(SymTridiagonal(fill(2.0, n), fill(-1.0, n-1)))
        d=range(1, 2; length = n)
        A=T.(Diagonal(d)*C*Diagonal(1 ./ d))
        a=FaerMatrix(A)
        FaerTeam(2) do team
            s=FaerScratch(eigen, a; side, team)
            F=eigen(a; side, scratch = s, team)
            @test real.(F.values)≈[2-2cos(k*pi/(n+1)) for k = 1:n] atol=3e-3
            left=side===:left ? F.vectors : F.left
            @test left'*A≈Diagonal(F.values)*left' rtol=1e-3 atol=1e-3
            side===:both && @test A*F.right≈F.right*Diagonal(F.values) rtol=1e-3 atol=1e-3
            close(s)
        end
    end
    a=FaerMatrix([1.0 -2; 3 4])
    @test_throws ArgumentError eigen(a; side = :invalid)
    @test_throws ArgumentError scratch_requirements(eigen, a; side = :invalid)
    s=FaerScratch(eigen, a)
    @test_throws ArgumentError eigen(a; side = :both, scratch = s)
    reserve!(s, eigen, a; side = :both)
    F=eigen(a; side = :both, scratch = s)
    @test Matrix(a)*F.right≈F.right*Diagonal(F.values)
    @test F.left'*Matrix(a)≈Diagonal(F.values)*F.left'
    close(s)
    close(a)
    @test_throws ArgumentError eigen(a; side = :left)
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
