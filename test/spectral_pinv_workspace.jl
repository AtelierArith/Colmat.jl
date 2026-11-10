using Test, Colmat, LinearAlgebra, SparseArrays
@testset "Saved SVD pseudoinverse workspace" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        dims in ((0, 0), (0, 3), (3, 0), (3, 2), (2, 3), (3, 3)),
        full in (false, true)

        m, n = dims
        A = reshape(T.(1:(m * n)), m, n)
        T <: Complex && (A .+= T(0.2im))
        a = FaerMatrix(A)
        FaerTeam(2) do team
            F = svd(a; native=true, full, team)
            close(a)
            req = scratch_requirements(pinv, F; team)
            s = FaerScratch(pinv, F; team)
            @test scratch_capacity(s) == req
            for _ in 1:2
                P = pinv(F; scratch=s, team)
                @test P isa FaerMatrix{T}
                @test size(P) == reverse(dims)
                @test Matrix(P) ≈ pinv(A) atol = 2e-3
                @test A * Matrix(P) * A ≈ A atol = 2e-3
                @test Matrix(P) * A * Matrix(P) ≈ Matrix(P) atol = 2e-3
                @test scratch_capacity(s) == req
                close(P)
            end
            C = copy(s)
            D = deepcopy(s)
            close(s)
            @test Matrix(pinv(F; scratch=C, team)) ≈ pinv(A) atol = 2e-3
            @test Matrix(pinv(F; scratch=D)) ≈ pinv(A) atol = 2e-3
            @test_throws ArgumentError pinv(F; scratch=s)
            close(C)
            close(D)
            close(F)
        end
    end
end
@testset "Saved self-adjoint pseudoinverse workspace" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), n in (0, 1, 4), uplo in (:L, :U)
        A = Matrix{T}(Diagonal(T.(0:(n - 1))))
        n > 1 && (A[1, 1]=-2; A[1, 2]=T(0.2); A[2, 1]=T(0.2))
        T <: Complex && n > 1 && (A[1, 2] += T(0.1im); A[2, 1] -= T(0.1im))
        a = FaerMatrix(A)
        n > 1 && (a[uplo === :L ? 1 : 2, uplo === :L ? 2 : 1] = T(NaN))
        FaerTeam(2) do team
            F = eigen(Hermitian(a, uplo); native=true, team, sortby=x -> -x)
            close(a)
            s = FaerScratch(pinv, F; team)
            @test scratch_capacity(s) == scratch_requirements(pinv, F; team)
            for _ in 1:2
                P = pinv(F; scratch=s, team)
                @test Matrix(P) ≈ pinv(A) atol = 2e-3
                @test A * Matrix(P) * A ≈ A atol = 2e-3
                @test Matrix(P) * A * Matrix(P) ≈ Matrix(P) atol = 2e-3
                close(P)
            end
            close(s)
            close(F)
        end
    end
end
@testset "Spectral workspace tolerances, reserve and metadata guards" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        A = Matrix{T}(Diagonal(T[-4, 1e-4, 0, 2]))
        for F in
            (svd(FaerMatrix(A); native=true), eigen(Hermitian(FaerMatrix(A)); native=true))
            s = FaerScratch(pinv, F)
            @test Matrix(pinv(F; scratch=s, atol=1e-3)) ≈
                Matrix{T}(Diagonal(T[-0.25, 0, 0, 0.5])) atol = 1e-4
            @test_throws ArgumentError pinv(F; scratch=s, rtol=-1)
            @test_throws ArgumentError pinv(F; scratch=1)
            @test_throws ArgumentError scratch_requirements(pinv, F; team=1)
            close(s)
            close(F)
        end
    end
    F = svd(FaerMatrix([1.0 2; 3 4; 5 6]); native=true)
    E = eigen(Hermitian(FaerMatrix(Matrix{Float64}(I, 8, 8))); native=true)
    s = FaerScratch()
    @test_throws ArgumentError pinv(F; scratch=s)
    reserve!(s, pinv, F)
    @test Matrix(pinv(F; scratch=s)) ≈ pinv([1.0 2; 3 4; 5 6])
    @test_throws ArgumentError pinv(E; scratch=s)
    reserve!(s, pinv, E)
    @test Matrix(pinv(E; scratch=s)) ≈ Matrix{Float64}(I, 8, 8)
    req = scratch_requirements(pinv, F)
    if req.alignment > 1
        bad = FaerScratch(req.bytes; alignment=1)
        @test_throws ArgumentError pinv(F; scratch=bad)
        close(bad)
    end
    F.dims = (4, 4)
    @test_throws Colmat.RustCall.RustError scratch_requirements(pinv, F)
    @test_throws Colmat.RustCall.RustError pinv(F; scratch=s)
    close(F)
    @test_throws ArgumentError scratch_requirements(pinv, F)
    close(E)
    close(s)
    @test isempty(
        Test.detect_ambiguities(Colmat, Base, LinearAlgebra, SparseArrays; recursive=true)
    )
end
@testset "Larger retained-factor pinv workspace" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        n = 64
        A = Matrix{T}(SymTridiagonal(fill(2.0, n), fill(-1.0, n - 1)))
        for F in
            (svd(FaerMatrix(A); native=true), eigen(Hermitian(FaerMatrix(A)); native=true))
            FaerTeam(2) do team
                s = FaerScratch(pinv, F; team)
                P = pinv(F; scratch=s, team)
                @test A * Matrix(P) * A ≈ A rtol = 3e-3 atol = 3e-3
                @test Matrix(P) ≈ inv(A) rtol = 3e-3 atol = 3e-3
                close(s)
                close(P)
            end
            close(F)
        end
    end
end
