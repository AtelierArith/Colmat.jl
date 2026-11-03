using Test, Colmat, LinearAlgebra, SparseArrays
@testset "Native general eigen writeback" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), n in (0, 1, 4)
        A=reshape(T.(1:(n*n)), n, n)
        T<:Complex && (A .+= T(0.25im))
        a=FaerMatrix(A)
        v=view(a, :, :)
        F=eigen!(a)
        @test A*F.vectors≈F.vectors*Diagonal(F.values) atol=2e-3
        @test Matrix(v)==Matrix(a)
        @test istriu(Matrix(a), T<:Real ? -1 : 0)
        @test eigvals(Matrix(a))≈F.values atol=2e-3
        @test eltype(F.values)==Complex{typeof(real(zero(T)))}
        for j = 1:n
            @test norm(F.vectors[:, j])≈1 atol=1e-5
        end
        saved=copy(F.vectors)
        fill!(a, zero(T))
        close(a)
        @test F.vectors==saved
        b=FaerMatrix(A)
        values=eigvals!(b)
        @test values≈F.values atol=2e-3
        @test eigvals(Matrix(b))≈values atol=2e-3
    end
    for T in (Float32, Float64)
        A=T[0 -1; 1 0]
        a=FaerMatrix(A)
        F=eigen!(a)
        @test A*F.vectors≈F.vectors*Diagonal(F.values)
        @test eltype(a)==T
        @test imag.(F.values)≈[-1, 1]
        @test Matrix(a)[2, 1]!=0
    end
end
@testset "Selected self-adjoint eigen mutation" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), uplo in (:L, :U), n in (0, 1, 4)
        X=reshape(T.(1:(n*n)), n, n)
        T<:Complex && (X .+= T(0.25im))
        A=X*X'+Matrix{T}(I, n, n)
        a=FaerMatrix(A)
        n>1 && (a[uplo===:L ? 1 : 2, uplo===:L ? 2 : 1]=T(NaN))
        F=eigen!(Hermitian(a, uplo))
        @test Matrix(a)≈F.vectors
        @test A*F.vectors≈F.vectors*Diagonal(F.values) atol=2e-3
        @test F.vectors'*F.vectors≈Matrix{T}(I, n, n) atol=2e-4
        b=FaerMatrix(A)
        values=eigvals!(Hermitian(b, uplo); sortby = nothing)
        @test Matrix(b)==Matrix{T}(Diagonal(values))
        @test sort(values)≈F.values atol=2e-3
    end
end
@testset "Generalized eigen mutation" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), n in (0, 1, 3)
        X=reshape(T.(1:(n*n)), n, n)
        T<:Complex && (X .+= T(0.25im))
        A=X+3Matrix{T}(I, n, n)
        B=Matrix{T}(I, n, n)+fill(T(0.1), n, n)
        a, b=FaerMatrix(A), FaerMatrix(B)
        F=eigen!(a, b)
        @test A*F.vectors≈B*F.vectors*Diagonal(F.values) atol=2e-3
        @test istriu(Matrix(b))
        @test sort(eigvals(Matrix(a), Matrix(b)); by = x->(real(x), imag(x)))≈F.values atol=2e-3
        a, b=FaerMatrix(A), FaerMatrix(B)
        values=eigvals!(a, b)
        @test values≈F.values atol=2e-3
        A=X*X'+Matrix{T}(I, n, n)
        a, b=FaerMatrix(A), FaerMatrix(B)
        G=eigen!(Hermitian(a), Hermitian(b))
        @test A*G.vectors≈B*G.vectors*Diagonal(G.values) atol=2e-3
        @test Matrix(a)≈G.vectors
        @test Matrix(b)==Matrix{T}(Diagonal(G.values))
    end
end
@testset "Eigen mutation failure transactions" begin
    a=FaerMatrix([1.0 2; 3 4])
    saved=Matrix(a)
    @test_throws ErrorException eigen!(a; sortby = x->error("sort callback failed"))
    @test Matrix(a)==saved
    @test_throws ArgumentError eigen!(a, a)
    @test Matrix(a)==saved
    b=FaerMatrix(ones(Float32, 2, 2))
    @test_throws ArgumentError eigen!(a, b)
    @test Matrix(a)==saved
    bad=FaerMatrix([1.0 2; 3 4])
    bad[1, 1]=NaN
    @test_throws Colmat.RustCall.RustError eigen!(bad)
    @test isnan(bad[1, 1])
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

@testset "Eigen mutation pair policies and sorting" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), uplo in (:L, :U)
        A=T[4 1; 1 3]
        B=T[2 0; 0 1]
        a, b=FaerMatrix(A), FaerMatrix(B)
        values=eigvals!(Hermitian(a, uplo), Hermitian(b, uplo))
        @test Matrix(a)==Matrix{T}(Diagonal(values))
        @test Matrix(b)==Matrix(a)
        @test values≈eigvals(Hermitian(A), Hermitian(B)) atol=2e-4
    end
    A=[1.0 2; 3 4]
    B=[2.0 0; 0 1]
    a, b=FaerMatrix(A), FaerMatrix(B)
    @test_throws ErrorException eigen!(a, b; sortby = x->error("sort callback failed"))
    @test Matrix(a)==A
    @test Matrix(b)==B
    @test_throws ArgumentError eigen!(Hermitian(a), Hermitian(a))
    a=FaerMatrix(A)
    F=eigen!(a; sortby = nothing)
    @test A*F.vectors≈F.vectors*Diagonal(F.values)
end
