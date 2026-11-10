using Test, Colmat, LinearAlgebra, SparseArrays
@testset "Low-level native SVD workspace and writeback" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        dims in ((0, 0), (0, 3), (3, 0), (1, 1), (5, 3), (3, 5), (4, 4)),
        full in (false, true)

        m, n = dims
        k = min(dims...)
        A = reshape(T.(1:(m * n)), dims)
        T <: Complex && (A .+= T(0.25im))
        a = FaerMatrix(A)
        req = scratch_requirements(svd, a; full)
        s = FaerScratch(svd, a; full)
        @test scratch_capacity(s) == req
        F = svd(a; full, scratch=s)
        @test F.U[:, 1:k] * Diagonal(F.S) * F.Vt[1:k, :] ≈ A atol = 1e-4
        @test F.S ≈ svdvals(A) atol = 1e-4
        @test eltype(F.S) == typeof(real(zero(T)))
        @test size(F.U) == (m, full ? m : k)
        @test size(F.Vt) == (full ? n : k, n)
        @test F.U' * F.U ≈ Matrix{T}(I, size(F.U, 2), size(F.U, 2)) atol = 2e-4
        @test F.Vt * F.Vt' ≈ Matrix{T}(I, size(F.Vt, 1), size(F.Vt, 1)) atol = 2e-4
        @test Matrix(a) == A
        v = view(a, :, :)
        G = svd!(a; full, scratch=s)
        sigma = zeros(T, m, n)
        for i in 1:k
            sigma[i, i] = G.S[i]
        end
        @test Matrix(a) == sigma
        @test Matrix(v) == sigma
        @test G.U[:, 1:k] * Diagonal(G.S) * G.Vt[1:k, :] ≈ A atol = 1e-4
        close(a)
        @test G.U[:, 1:k] * Diagonal(G.S) * G.Vt[1:k, :] ≈ A atol = 1e-4
        b = FaerMatrix(A)
        reserve!(s, svdvals, b)
        vals = svdvals!(b; scratch=s)
        @test vals ≈ F.S atol = 1e-4
        @test diag(Matrix(b)) ≈ vals atol = 1e-4
        close(s)
    end
    @test isempty(
        Test.detect_ambiguities(Colmat, Base, LinearAlgebra, SparseArrays; recursive=true)
    )
end
@testset "SVD teams, scalar modes and transaction failures" begin
    for width in (1, 2)
        FaerTeam(width) do team
            for T in (Float32, Float64, ComplexF32, ComplexF64)
                A = T[1 2 3; 4 5 7; 2 4 1; 3 1 2]
                T <: Complex && (A .+= T(0.5im))
                a = FaerMatrix(A)
                s = FaerScratch(svd, a; team)
                F = svd(a; scratch=s, team)
                @test F.U * Diagonal(F.S) * F.Vt ≈ A atol = 2e-4
                @test svdvals(a; team) ≈ F.S atol = 2e-4
                @test svd(a; team).S ≈ F.S atol = 2e-4
                @test scratch_capacity(s) == scratch_requirements(svd, a; team)
                close(s)
            end
        end
    end
    a = FaerMatrix([1.0 2; 3 4])
    saved = Matrix(a)
    tiny = FaerScratch()
    @test_throws ArgumentError svd!(a; scratch=tiny)
    @test Matrix(a) == saved
    a[1, 1] = NaN
    @test_throws Colmat.RustCall.RustError svd!(a)
    @test isnan(a[1, 1])
    @test a[2, 2] == 4
    @test_throws ArgumentError svd(a; team=1)
    @test_throws ArgumentError svd(a; scratch=1)
end

@testset "SVD query aliases and metadata guards" begin
    a = FaerMatrix([1.0 2 3; 4 5 6])
    @test scratch_requirements(svd!, a) == scratch_requirements(svd, a)
    @test scratch_requirements(svdvals!, a) == scratch_requirements(svdvals, a)
    s = FaerScratch(svdvals!, a)
    @test svdvals!(copy(a); scratch=s) ≈ svdvals(a)
    close(s)
    broken = copy(a)
    broken.dims = (3, 2)
    s = FaerScratch(svd, broken)
    @test_throws Colmat.RustCall.RustError svd(broken; scratch=s)
    close(s)
    close(broken)
end
