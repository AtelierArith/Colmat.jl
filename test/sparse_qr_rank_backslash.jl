using Colmat, LinearAlgebra, SparseArrays, Test
@testset "Default rank-aware sparse QR solves and inverse actions" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        R in (T, ComplexF64),
        algorithm in (:auto, :simplicial, :supernodal),
        format in (:csc, :csr),
        raw in (
            zeros(5, 3),
            [1.0 2 0; 0 0 0; 0 0 0; 0 0 0; 0 0 0],
            [1.0 2 3; 2 4 6; 3 6 9; 1 2 3; 0 0 0],
            [1.0 2 0; 0 1 2; 0 0 1],
        )

        A=T.(raw)
        T<:Complex && (A .*= T(1+0.2im))
        F=qr(FaerSparseMatrix(sparse(A); format); algorithm)
        close(F.source)
        M=FaerLinearOperator(F)
        P=promote_type(T, R)
        rtol=min(size(A)...)*eps(typeof(real(zero(T))))
        for (f, ref, op) in (
            (F, A, M),
            (transpose(F), transpose(A), transpose(M)),
            (adjoint(F), adjoint(A), adjoint(M)),
        )
            B=ones(R, size(ref, 1), 2)
            expected=pinv(P.(ref); rtol)*B
            @test Matrix(f\B)≈expected atol=1e-3 rtol=1e-3
            @test f\B[:, 1]≈expected[:, 1] atol=1e-3 rtol=1e-3
            @test Matrix(op*B)≈expected atol=1e-3 rtol=1e-3
        end
        close(F)
        @test Matrix(M)≈pinv(A; rtol) atol=1e-3 rtol=1e-3
        close(M)
    end
end

@testset "Rank-deficient workspace solve rejects before output mutation" begin
    F=qr(FaerSparseMatrix(sparse([1.0 0; 0 0; 0 0])))
    B=ones(3, 1)
    @test_throws ArgumentError scratch_requirements(ldiv!, F, B)
    s=FaerScratch(0)
    X=fill(9.0, 2, 1)
    @test_throws ArgumentError ldiv!(X, F, B; scratch = s)
    @test X==fill(9.0, 2, 1)
    close(s)
    close(F)
end
