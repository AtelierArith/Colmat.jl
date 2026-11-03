using Test, Colmat, LinearAlgebra, SparseArrays
@testset "Generalized EVD scratch and team" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), n in (0, 1, 4), width in (1, 2)
        A=Matrix{T}(I, n, n)
        n>1 && (A[1, 2] = -2; A[2, 1] = 3)
        n>2 && (A[1, 3]=T(0.2))
        T<:Complex && (A .*= T(1+0.2im))
        B=Matrix{T}(I, n, n)
        n>1 && (B[1, 1] = 2; B[1, 2] = T(0.1))
        a, b=FaerMatrix(A), FaerMatrix(B)
        FaerTeam(width) do team
            s=FaerScratch(eigen, a, b; team)
            @test scratch_capacity(s)==scratch_requirements(eigen!, a, b; team)
            F=eigen(a, b; scratch = s, team)
            @test A*F.vectors≈B*F.vectors*Diagonal(F.values) atol=1e-3
            @test all(v->isapprox(norm(v), 1; atol = 1e-4), eachcol(F.vectors))
            @test Matrix(a)==A
            @test Matrix(b)==B
            @test eltype(F.values)==Complex{typeof(real(zero(T)))}
            reserve!(s, eigvals!, a, b; team)
            @test sort(eigvals(a, b; scratch = s, team); by = x->(real(x), imag(x)))≈sort(
                F.values;
                by = x->(real(x), imag(x)),
            ) atol=1e-3
            @test eigen(a, b; team).values≈F.values atol=1e-3
            x, y=FaerMatrix(A), FaerMatrix(B)
            G=eigen!(x, y; scratch = s, team)
            @test A*G.vectors≈B*G.vectors*Diagonal(G.values) atol=1e-3
            @test istriu(Matrix(y))
            @test all(iszero, tril(Matrix(x), -2))
            x, y=FaerMatrix(A), FaerMatrix(B)
            vals=eigvals!(x, y; scratch = s, team)
            @test vals≈F.values atol=1e-3
            @test istriu(Matrix(y))
            close(s)
        end
    end
end
@testset "Generalized EVD larger analytic spectrum" begin
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
            F=eigen(a, b; scratch = s, team)
            @test real.(F.values)≈[2-2cos(k*pi/(n+1)) for k = 1:n] atol=3e-3
            @test maximum(abs, imag.(F.values))<3e-3
            @test A*F.vectors≈B*F.vectors*Diagonal(F.values) rtol=1e-3 atol=1e-3
            close(s)
        end
    end
end
@testset "Generalized EVD boundary and transactions" begin
    a, b=FaerMatrix([0.0 -1; 1 0]), FaerMatrix([2.0 0; 0 1])
    A, B=Matrix(a), Matrix(b)
    s=FaerScratch(eigen, a, b)
    @test_throws ArgumentError eigen(a, b; scratch = FaerScratch())
    @test_throws ArgumentError eigen(a, b; team = 1)
    @test_throws ErrorException eigen!(a, b; scratch = s, sortby = x->error("sort failed"))
    @test Matrix(a)==A
    @test Matrix(b)==B
    @test_throws ArgumentError eigen!(a, a; scratch = s)
    b[1, 1]=NaN
    @test_throws Colmat.RustCall.RustError eigen!(a, b; scratch = s)
    @test Matrix(a)==A
    close(s)
    a, b=FaerMatrix(Float32[0 -1; 1 0]), FaerMatrix(B)
    s=FaerScratch(eigen, a, b)
    @test eltype(eigen(a, b; scratch = s).values)==ComplexF64
    close(s)
    a.dims=(3, 3)
    @test_throws DimensionMismatch scratch_requirements(eigen, a, b)
    b.dims=(3, 3)
    s=FaerScratch(eigen, a, b)
    @test_throws Colmat.RustCall.RustError eigen(a, b; scratch = s)
    close(s)
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
@testset "Repeated finite generalized eigenvalues" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        a, b=FaerMatrix(Matrix{T}(I, 3, 3)), FaerMatrix(Matrix{T}(I, 3, 3))
        F=eigen(a, b)
        @test all(isfinite, F.vectors)
        @test F.values≈ones(3)
        @test F.vectors'*F.vectors≈I atol=1e-4
    end
end
