using Test, Colmat, LinearAlgebra, SparseArrays
@testset "Self-adjoint EVD scratch and team execution" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        n in (0, 1, 4), uplo in (:L, :U),
        width in (1, 2)

        FaerTeam(width) do team
            X = reshape(T.(1:(n * n)), n, n)
            if T <: Complex
                for j in 1:n, i in 1:n
                    X[i, j] += T((i + 2j) * 0.05im)
                end
            end
            A = X + X' + Matrix{T}(I, n, n)
            source = FaerMatrix(A)
            n > 1 && (source[uplo === :L ? 1 : 2, uplo === :L ? 2 : 1] = T(NaN))
            wrapped = Hermitian(source, uplo)
            req = scratch_requirements(eigen, wrapped; team)
            s = FaerScratch(eigen, wrapped; team)
            @test scratch_capacity(s) == req
            F = eigen(wrapped; scratch=s, team)
            @test A * F.vectors ≈ F.vectors * Diagonal(F.values) atol = 1e-3
            @test F.vectors' * F.vectors ≈ Matrix{T}(I, n, n) atol = 2e-4
            @test F.values ≈ eigvals(Hermitian(A)) atol = 1e-3
            @test eltype(F.values) == typeof(real(zero(T)))
            @test scratch_requirements(eigen!, wrapped; team) == req
            @test scratch_requirements(eigvals!, wrapped; team) ==
                scratch_requirements(eigvals, wrapped; team)
            reserve!(s, eigvals, wrapped; team)
            @test eigvals(wrapped; scratch=s, team) ≈ F.values atol = 1e-3
            @test eigen(wrapped; team).values ≈ F.values atol = 1e-3
            native = FaerMatrix(A)
            G = eigen!(Hermitian(native, uplo); scratch=s, team)
            @test Matrix(native) == G.vectors
            @test A * G.vectors ≈ G.vectors * Diagonal(G.values) atol = 1e-3
            native = FaerMatrix(A)
            values = eigvals!(Hermitian(native, uplo); scratch=s, team)
            @test Matrix(native) == Matrix{T}(Diagonal(values))
            close(s)
        end
    end
end
@testset "EVD selected diagonal, sorting and failure transactions" begin
    for T in (ComplexF32, ComplexF64)
        clean = T[4 1im; -1im 3]
        a = FaerMatrix(clean)
        a[1, 1] = T(4, NaN)
        wrapped = Hermitian(a, :L)
        s = FaerScratch(eigen, wrapped)
        F = eigen(wrapped; scratch=s, sortby=x -> -x)
        @test F.values ≈ reverse(eigvals(Hermitian(clean))) atol = 1e-4
        @test clean * F.vectors ≈ F.vectors * Diagonal(F.values) atol = 1e-4
        @test isnan(imag(a[1, 1]))
        close(s)
    end
    a = FaerMatrix([4.0 1; 1 3])
    A = Matrix(a)
    wrapped = Hermitian(a)
    s = FaerScratch(eigen, wrapped)
    @test_throws ErrorException eigen!(wrapped; scratch=s, sortby=x -> error("sort failed"))
    @test Matrix(a) == A
    tiny = FaerScratch()
    @test_throws ArgumentError eigen!(wrapped; scratch=tiny)
    @test Matrix(a) == A
    @test_throws ArgumentError eigen(wrapped; scratch=1)
    @test_throws ArgumentError eigen(wrapped; team=1)
    a[1, 1] = NaN
    @test_throws Colmat.RustCall.RustError eigen!(wrapped; scratch=s)
    @test isnan(a[1, 1])
    @test a[2, 2] == 3
    close(s)
    close(tiny)
    @test isempty(
        Test.detect_ambiguities(Colmat, Base, LinearAlgebra, SparseArrays; recursive=true)
    )
end
@testset "Blocked EVD, real Symmetric wrappers and shape guards" begin
    n = 64
    A = Matrix(SymTridiagonal(fill(2.0, n), fill(-1.0, n - 1)))
    H = Hermitian(FaerMatrix(A))
    FaerTeam(2) do team
        s = FaerScratch(eigen, H; team)
        F = eigen(H; scratch=s, team)
        expected = [2 - 2cos(k * pi / (n + 1)) for k in 1:n]
        @test F.values ≈ expected atol = 1e-10
        @test A * F.vectors ≈ F.vectors * Diagonal(F.values) atol = 1e-9
        @test F.vectors' * F.vectors ≈ I atol = 1e-10
        reserve!(s, eigvals, H; team)
        @test eigvals(H; scratch=s, team) ≈ F.values atol = 1e-10
        close(s)
    end
    for T in (Float32, Float64)
        A = T[3 1; 1 -2]
        a = FaerMatrix(A)
        wrapped = Symmetric(a, :U)
        s = FaerScratch(eigen, wrapped)
        F = eigen(wrapped; scratch=s)
        @test A * F.vectors ≈ F.vectors * Diagonal(F.values) atol = 1e-4
        @test eltype(F.values) == T
        @test Matrix(a) == A
        G = eigen!(wrapped; scratch=s)
        @test Matrix(a) == G.vectors
        close(s)
    end
    a = FaerMatrix([2.0 1; 1 3])
    wrapped = Hermitian(a)
    a.dims = (3, 3)
    s = FaerScratch(eigen, wrapped)
    @test_throws Colmat.RustCall.RustError eigen(wrapped; scratch=s)
    close(s)
    close(a)
end
