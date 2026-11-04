using Colmat, LinearAlgebra, SparseArrays, Random, Test

@testset "Owned auto/supernodal sparse QR components and transpose solves" begin
    rng=MersenneTwister(4729)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        format in (:csc, :csr),
        algorithm in (:auto, :supernodal),
        (m, n) in ((0, 0), (3, 0), (1, 1), (7, 4), (12, 12), (65, 40), (130, 80))
        # Two disconnected column groups force multiple supernodes; random
        # sparse couplings exercise row gathers and nontrivial COLAMD ordering.
        A=zeros(T, m, n)
        for j = 1:n
            A[j, j]=T(5)
            for i = 1:m
                if m==130 || (rand(rng)<0.12 && (i<=cld(m, 2))==(j<=cld(n, 2)))
                    A[i, j]+=randn(rng, T)
                end
            end
        end
        a=FaerSparseMatrix(sparse(A); format)
        F=qr(a; algorithm)
        a .= zero(T)
        close(a)
        close(F.source)
        Q, R, p=F.Q, F.R, F.pcol
        @test size(Q)==(m, n)
        @test size(R)==(n, n)
        @test Matrix(Q)'*Matrix(Q)≈Matrix{T}(I, n, n)
        @test Matrix(Q)*Matrix(R)≈A[:, Vector(p)]
        @test Matrix(F)≈A
        B=randn(rng, T, n, 2)
        for (f, ref) in ((transpose(F), transpose(A)), (adjoint(F), adjoint(A)))
            X=f\B
            tol=sqrt(eps(real(T)))
            @test Matrix(X)≈pinv(Matrix(ref))*B atol=tol rtol=tol
            @test f\B[:, 1]≈pinv(Matrix(ref))*B[:, 1] atol=tol rtol=tol
            out=zeros(T, m, 2)
            @test ldiv!(out, f, B)===out
            @test out≈Matrix(X) atol=tol rtol=tol
            close(X)
        end
        # Extracted components are independent of numeric updates/close.
        close(F)
        @test Matrix(Q)*Matrix(R)≈A[:, Vector(p)]
        close(Q)
        close(R)
        close(p)
    end
end

@testset "Owned sparse QR updates, rank and closed-owner errors" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), algorithm in (:auto, :supernodal)
        A=T[4 0; 1 3; 0 1]
        T<:Complex && (A[2, 1]+=T(0.4im))
        a=FaerSparseMatrix(sparse(A))
        F=qr(a; algorithm)
        Q, R, p=F.Q, F.R, F.p
        update=FaerSparseMatrix(sparse(2A))
        @test qr!(F, update)===F
        @test Matrix(F)≈2A
        @test Matrix(Q)*Matrix(R)≈A[:, Vector(p)]
        @test Matrix(F'\ones(T, 2, 1))≈pinv(Matrix((2A)'))*ones(T, 2, 1)
        @test_throws DimensionMismatch F'\zeros(T, 3, 1)
        close(F)
        @test_throws ArgumentError F.Q
        @test_throws ArgumentError F.R
        @test_throws ArgumentError reconstruct(F)
        close(Q)
        close(R)
        close(p)
        close(a)
        close(update)
        bad=FaerSparseMatrix(sparse(T[1 0; 0 0; 0 0]))
        G=qr(bad; algorithm)
        @test Matrix(G'\ones(T, 2, 1))≈pinv(Matrix(bad)')*ones(T, 2, 1) atol=1e-3 rtol=1e-3
        close(G)
        close(bad)
    end
end
