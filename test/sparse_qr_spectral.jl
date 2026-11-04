using Colmat, LinearAlgebra, SparseArrays, Test
@testset "Saved sparse QR singular values rank and pseudoinverse" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        algorithm in (:auto, :simplicial, :supernodal),
        format in (:csc, :csr),
        fixture in (
            zeros(0, 0),
            zeros(5, 0),
            zeros(5, 3),
            [1.0 2 0; 0 0 0; 0 0 0; 0 0 0; 0 0 0],
            [3.0 0; 0 1e-6; 0 0],
            [2.0 0; 1 3; 0 1],
        ),
        reversed in (false, true)

        A=T.(fixture)
        T<:Complex && (A .*= T(1+0.3im))
        n=size(A, 2)
        order=reversed ? reverse(collect(1:n)) : :default
        F=qr(FaerSparseMatrix(sparse(A); format); algorithm, ordering = order)
        close(F.source)
        @test svdvals(F)≈svdvals(A) atol=1e-3 rtol=1e-3
        for atol in (0.0, 1e-4), rtol in (0.0, 1e-4)
            @test rank(F; atol, rtol)==rank(A; atol, rtol)
            P=pinv(F; atol, rtol)
            expected=pinv(A; atol, rtol)
            @test Matrix(P)≈expected atol=1e-3 rtol=1e-3
            close(P)
        end
        @test_throws ArgumentError rank(F; atol = -1)
        @test_throws ArgumentError pinv(F; rtol = NaN)
        P=pinv(F; rtol = 1e-4)
        close(F)
        @test Matrix(P)≈pinv(A; rtol = 1e-4) atol=1e-3 rtol=1e-3
        @test_throws ArgumentError rank(F)
        close(P)
    end
end
@testset "Explicit tolerance minimum-norm sparse QR solves" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        algorithm in (:auto, :simplicial, :supernodal)

        A=T[1 2 0; 0 0 0; 0 0 0; 0 0 0; 0 0 0]
        T<:Complex && (A .*= T(1+0.2im))
        F=qr(FaerSparseMatrix(sparse(A)); algorithm)
        close(F.source)
        for (f, ref) in ((F, A), (transpose(F), transpose(A)), (adjoint(F), adjoint(A)))
            B=ones(ComplexF64, size(ref, 1), 2)
            X=fill(9.0+im, size(ref, 2), 2)
            @test ldiv!(X, f, B; rtol = 1e-4)===X
            @test X≈pinv(ComplexF64.(ref); rtol = 1e-4)*B atol=1e-3 rtol=1e-3
            y=fill(7.0+im, size(ref, 2))
            @test ldiv!(y, f, B[:, 1]; atol = 1e-4)===y
            @test y≈pinv(ComplexF64.(ref); atol = 1e-4)*B[:, 1] atol=1e-3 rtol=1e-3
            fill!(X, 9.0+im)
            @test_throws ArgumentError ldiv!(X, f, B; rtol = -1)
            @test all(==(9.0+im), X)
        end
        close(F)
    end
end
