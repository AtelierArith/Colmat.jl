using Test, Colmat, LinearAlgebra, SparseArrays
@testset "Dense factor reconstruction and typed conversion" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), n in (0, 3)
        A=n==0 ? zeros(T, 0, 0) : T[4 1 0; 1 3 1; 0 1 2]
        T<:Complex && n>0 && (A[1, 2]+=T(0.2im); A[2, 1]-=T(0.2im))
        a=FaerMatrix(A)
        factories=(
            x->lu(x),
            x->qr(x),
            x->cholesky(Hermitian(x)),
            x->ldlt(Hermitian(x)),
            x->bunchkaufman(Hermitian(x)),
            x->hessenberg(x),
            x->schur(x),
            x->tridiagonalize(Hermitian(x)),
            x->bidiagonalize(x; full = true),
        )
        for factory in factories
            F=factory(a)
            R=reconstruct(F)
            @test R isa FaerMatrix{T}
            @test Matrix(R)≈A atol=1e-3
            @test Matrix(F)≈A atol=1e-3
            @test Array(F)≈A atol=1e-3
            C=FaerMatrix{ComplexF64}(F)
            @test C isa FaerMatrix{ComplexF64}
            @test Matrix(C)≈A atol=1e-3
            @test Matrix{ComplexF64}(F)≈A atol=1e-3
            close(C)
            n>0 && (R[1, 1]+=one(T))
            @test Matrix(F)≈A atol=1e-3
            close(F)
            expected=copy(A)
            n>0 && (expected[1, 1]+=one(T))
            @test Matrix(R)≈expected atol=1e-3
            @test_throws ArgumentError reconstruct(F)
            @test_throws ArgumentError Matrix{ComplexF64}(F)
            close(R)
        end
        close(a)
    end
end
@testset "Simplicial sparse LU reconstruction uses actual factors" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        format in (:csc, :csr),
        n in (0, 3),
        custom in (false, true)

        A=n==0 ? zeros(T, 0, 0) : T[0 2 1; 3 1 0; 0 1 4]
        T<:Complex && n>0 && (A[1, 2]+=T(0.5im); A[3, 3]+=T(0.25im))
        native=FaerSparseMatrix(sparse(A); format)
        ordering=custom ? (n==0 ? Int[] : [3, 1, 2]) : :identity
        F=lu(native; algorithm = :simplicial, ordering)
        R=reconstruct(F)
        @test R isa FaerSparseMatrix{T}
        @test Matrix(R)≈A atol=1e-3
        @test Matrix(F)≈A atol=1e-3
        @test Matrix(FaerMatrix(F))≈A atol=1e-3
        @test Array(F)≈A atol=1e-3
        @test SparseMatrixCSC(F)≈sparse(A) atol=1e-3
        @test sparse(F)≈sparse(A) atol=1e-3
        S=SparseMatrixCSC{ComplexF64,Int32}(F)
        @test S isa SparseMatrixCSC{ComplexF64,Int32}
        @test S≈sparse(A) atol=1e-3
        @test Matrix{ComplexF64}(F)≈A atol=1e-3
        C=FaerSparseMatrix{ComplexF64}(F)
        @test Matrix(C)≈A atol=1e-3
        close(C)
        close(native)
        # The retained input snapshot is deliberately corrupted: reconstruction
        # must use independent numeric L/U components instead.
        fill!(nonzeros(F.source), zero(T))
        @test Matrix(F)≈A atol=1e-3
        updated=FaerSparseMatrix(sparse(2A); format)
        lu!(F, updated)
        @test Matrix(F)≈2A atol=1e-3
        @test Matrix(R)≈A atol=1e-3
        close(F)
        @test Matrix(R)≈A atol=1e-3
        @test_throws ArgumentError reconstruct(F)
        close(updated)
        close(R)
    end
end
@testset "Generalized pair reconstruction and errors" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), n in (0, 3)
        A=n==0 ? zeros(T, 0, 0) : T[1 -2 1; 3 1 0; 0 1 4]
        B=n==0 ? zeros(T, 0, 0) : T[2 0 0; 0 1 0; 0 0 3]
        T<:Complex && n>0 && (A .*= T(1+0.2im))
        a, b=FaerMatrix(A), FaerMatrix(B)
        for factory in (hessenberg, schur)
            F=factory(a, b)
            R=reconstruct(F)
            @test keys(R)==(:A, :B)
            @test R.A isa FaerMatrix{T}
            @test R.B isa FaerMatrix{T}
            @test Matrix(R.A)≈A atol=1e-3
            @test Matrix(R.B)≈B atol=1e-3
            close(F)
            @test Matrix(R.A)≈A atol=1e-3
            @test Matrix(R.B)≈B atol=1e-3
            @test_throws ArgumentError reconstruct(F)
            close(R.A)
            close(R.B)
        end
    end
    F=lu(FaerSparseMatrix(sparse([1.0 2; 2 4])); algorithm = :simplicial, check = false)
    @test_throws SingularException reconstruct(F)
    close(F)
    F=lu(FaerSparseMatrix(sparse([2.0 1; 1 3])))
    @test_throws Colmat.RustCall.RustError reconstruct(F)
    close(F)
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
@testset "Reconstruction conversion and shape errors preserve factors" begin
    A=ComplexF64[2+1im 0; 0 3]
    F=schur(FaerMatrix(A))
    @test_throws InexactError FaerMatrix{Float64}(F)
    @test_throws InexactError Matrix{Float64}(F)
    @test isopen(F)
    @test Matrix(F)≈A
    close(F)
    F=schur(FaerMatrix([2.0 1; 0 3]), FaerMatrix([1.0 0; 0 1]))
    F.T.dims=(3, 3)
    @test_throws DimensionMismatch reconstruct(F)
    close(F)
end
