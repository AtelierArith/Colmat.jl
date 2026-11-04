using Colmat, LinearAlgebra, SparseArrays, Test
@testset "Retained sparse QR solve workspace" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        algorithm in (:auto, :supernodal),
        format in (:csc, :csr),
        (m, n) in ((0, 0), (3, 0), (4, 4), (7, 4))

        A=zeros(T, m, n)
        for j = 1:n
            A[j, j]=T(3+j)
            j<m && (A[j+1, j]=T(0.2))
            T<:Complex && (A[j, j]+=T(0.3im))
        end
        a=FaerSparseMatrix(sparse(A); format)
        F=qr(a; algorithm)
        close(a)
        B=reshape(T.(1:2m), m, 2)
        req=scratch_requirements(ldiv!, F, B)
        s=FaerScratch(ldiv!, F, B)
        X=zeros(T, n, 2)
        for _ = 1:2
            @test ldiv!(X, F, B; scratch = s)===X
            @test X≈A\B atol=1e-3 rtol=1e-3
            @test scratch_capacity(s).bytes>=req.bytes
        end
        fill!(nonzeros(F.source), 0)
        @test ldiv!(X, F, B; scratch = s)≈A\B atol=1e-3 rtol=1e-3
        close(s)
        @test_throws ArgumentError ldiv!(X, F, B; scratch = s)
        close(F)
        @test_throws ArgumentError scratch_requirements(ldiv!, F, B)
    end
end

@testset "Sparse QR workspace failure and reserve contract" begin
    A=FaerSparseMatrix(sparse([2.0 0; 0 3; 1 0]))
    F=qr(A; algorithm = :supernodal)
    B=ones(3, 2)
    X=fill(99.0, 2, 2)
    req=scratch_requirements(ldiv!, F, B)
    s=FaerScratch(0; alignment = 1)
    @test req.bytes>0
    @test_throws ArgumentError ldiv!(X, F, B; scratch = s)
    @test X==fill(99.0, 2, 2)
    @test reserve!(s, ldiv!, F, B)===s
    @test ldiv!(X, F, B; scratch = s)≈Matrix(A)\B
    @test_throws DimensionMismatch scratch_requirements(ldiv!, F, ones(2, 2))
    @test scratch_requirements(ldiv!, F, ones(Float32, 3, 2))==req
    @test_throws DimensionMismatch ldiv!(zeros(3, 2), F, B; scratch = s)
    @test_throws ArgumentError ldiv!(X, F, B; scratch = :invalid)
    reserve!(s, ldiv!, adjoint(F), ones(2, 2))
    @test ldiv!(zeros(3, 2), adjoint(F), ones(2, 2); scratch = s)≈pinv(Matrix(A))' *
                                                                  ones(2, 2)
    G=qr(A; algorithm = :simplicial)
    @test scratch_requirements(ldiv!, G, B).bytes>0
    close(G)
    close(F)
    close(A)
    close(s)
end
