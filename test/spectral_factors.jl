using Test, Colmat, LinearAlgebra, SparseArrays
@testset "Owned faer SVD reuse" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        dims in ((0, 0), (0, 3), (3, 0), (3, 2), (2, 3)),
        full in (false, true)

        m, n=dims
        k=min(m, n)
        A=reshape(T.(1:(m*n)), m, n)
        T<:Complex && (A .+= T(0.2im))
        a=FaerMatrix(A)
        FaerTeam(2) do team
            F=svd(a; native = true, full, team)
            @test F isa FaerSpectralFactor{T,:svd}
            @test size(F)==dims
            @test size(F.U)==(m, full ? m : k)
            @test size(F.V)==(n, full ? n : k)
            @test eltype(F.S)==typeof(real(zero(T)))
            @test Matrix(F)≈A atol=1e-3
            @test svdvals(F)≈svdvals(A) atol=1e-3
            P=pinv(F; team)
            @test Matrix(P)≈pinv(A) atol=2e-3
            @test A*Matrix(P)*A≈A atol=2e-3
            @test Matrix(P)*A*Matrix(P)≈Matrix(P) atol=2e-3
            @test rank(F)==rank(A)
            U, S, V=F.U, F.S, F.V
            @test Matrix(U)[:, 1:k]*Diagonal(S)*Matrix(V)[:, 1:k]'≈A atol=1e-3
            G=copy(F)
            H=deepcopy(F)
            close(F)
            @test !isopen(F)
            @test_throws ArgumentError F.S
            @test_throws ArgumentError pinv(F)
            @test_throws ArgumentError copy(F)
            @test Matrix(G)≈A atol=1e-3
            @test Matrix(H)≈A atol=1e-3
            @test Matrix(U)[:, 1:k]*Diagonal(S)*Matrix(V)[:, 1:k]'≈A atol=1e-3
            close(a)
            @test Matrix(pinv(G))≈pinv(A) atol=2e-3
            close(G)
            close(H)
            close(U)
            close(V)
            close(P)
        end
    end
end
@testset "Owned self-adjoint EVD reuse" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), n in (0, 1, 4), uplo in (:L, :U)
        A=Matrix{T}(Diagonal(T.(1:n)))
        n>1 && (A[1, 2] = T(0.3); A[2, 1] = T(0.3))
        T<:Complex && n>1 && (A[1, 2]+=T(0.2im); A[2, 1]-=T(0.2im))
        a=FaerMatrix(A)
        n>1 && (a[uplo===:L ? 1 : 2, uplo===:L ? 2 : 1]=T(NaN))
        FaerTeam(2) do team
            F=eigen(Hermitian(a, uplo); native = true, team, sortby = x->-x)
            @test F isa FaerSpectralFactor{T,:eigen}
            @test issorted(F.values; rev = true)
            @test Matrix(F)≈A atol=1e-3
            U=F.vectors
            @test A*Matrix(U)≈Matrix(U)*Diagonal(F.values) atol=1e-3
            @test eigvals(F)==F.values
            @test Matrix(eigvecs(F))≈Matrix(U)
            P=pinv(F; team)
            @test Matrix(P)≈pinv(A) atol=1e-3
            @test rank(F)==rank(A)
            G=copy(F)
            close(F)
            close(a)
            @test Matrix(G)≈A atol=1e-3
            @test Matrix(pinv(G))≈pinv(A) atol=1e-3
            @test Matrix(U)*Diagonal(G.values)*Matrix(U)'≈A atol=1e-3
            @test_throws ArgumentError F.values
            close(G)
            close(U)
            close(P)
        end
    end
end
@testset "Spectral tolerances, independence and owner checks" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        A=Matrix{T}(Diagonal(T[-4, 1e-4, 0, 2]))
        for F in (
            svd(FaerMatrix(A); native = true),
            eigen(Hermitian(FaerMatrix(A)); native = true),
        )
            @test Matrix(pinv(F; atol = 1e-3))≈Matrix{T}(Diagonal(T[-0.25, 0, 0, 0.5])) atol=1e-5
            @test rank(F; atol = 1e-3)==2
            @test_throws ArgumentError pinv(F; rtol = -1)
            @test_throws ArgumentError pinv(F; atol = NaN)
            @test Matrix{ComplexF64}(F)≈A atol=1e-3
            @test Matrix(FaerMatrix{ComplexF64}(F))≈A atol=1e-3
            close(F)
        end
    end
    a=FaerMatrix([3.0 1; 1 2])
    @test svd(a) isa SVD
    @test eigen(Hermitian(a)) isa Eigen
    @test_throws ArgumentError svd(a; native = true, scratch = FaerScratch())
    @test_throws ArgumentError svd(a; native = true, team = 1)
    @test_throws ErrorException eigen(
        Hermitian(a);
        native = true,
        sortby = x->error("sort failed"),
    )
    F=svd(a; native = true)
    U=F.U
    U[1, 1]=99
    @test Matrix(F)≈Matrix(a)
    F.dims=(3, 3)
    @test_throws Colmat.RustCall.RustError F.S
    @test_throws Colmat.RustCall.RustError reconstruct(F)
    close(F)
    close(U)
    close(a)
    @test_throws ArgumentError svd(a; native = true)
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
@testset "Spectral iteration, team lifetime and Float32 cutoff" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        A=T[4 1; 1 3]
        a=FaerMatrix(A)
        F=FaerTeam(2) do team
            svd(a; native = true, team)
        end
        U, S, V=F
        @test length(F)==3
        @test issuccess(F)
        @test Matrix(U)*Diagonal(S)*Matrix(V)'≈A atol=1e-3
        close(a)
        @test Matrix(pinv(F))≈inv(A) atol=1e-3
        close(F)
        close(F)
        @test !isopen(F)
        @test_throws ArgumentError issuccess(F)
        close(U)
        close(V)
        E=eigen(Hermitian(FaerMatrix(A)); native = true)
        values, vectors=E
        @test length(E)==2
        @test Matrix(vectors)*Diagonal(values)*Matrix(vectors)'≈A atol=1e-3
        close(E)
        close(vectors)
    end
    for T in (Float32, ComplexF32)
        value=nextfloat(0.5f0)
        A=Matrix{T}(Diagonal(T[1, value]))
        for F in (
            svd(FaerMatrix(A); native = true),
            eigen(Hermitian(FaerMatrix(A)); native = true),
        )
            # Locate the boundary around the stored factor value: self-adjoint
            # EVD may round the input diagonal by one Float32 ulp.
            stored=hasproperty(F, :S) ? F.S : F.values
            edge=minimum(abs, stored)
            lower=prevfloat(edge)
            threshold=Float64(lower)+0.75*(Float64(edge)-Float64(lower))
            @test Matrix(pinv(F; atol = threshold))≈Matrix{T}(Diagonal(T[1, inv(value)])) atol=2e-6
            close(F)
        end
    end
    for T in (Float32, Float64)
        A=T[4 1; 1 3]
        F=eigen(Symmetric(FaerMatrix(A), :U); native = true)
        @test Matrix(F)≈A atol=1e-3
        close(F)
    end
end
