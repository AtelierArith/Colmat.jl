using Colmat, LinearAlgebra, SparseArrays, Test
@testset "Mixed sparse solves reuse factors with closed source snapshot" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        R in (Float32, Float64, ComplexF32, ComplexF64),
        format in (:csc, :csr),
        (kind, algorithm) in (
            (:lu, :auto),
            (:lu, :supernodal),
            (:lu, :simplicial),
            (:qr, :auto),
            (:qr, :supernodal),
            (:qr, :simplicial),
            (:cholesky, :auto),
            (:cholesky, :simplicial),
            (:cholesky, :supernodal),
        ),
        n in (0, 3)

        A=n==0 ? zeros(T, 0, 0) :
          kind===:lu ? T[0 2 1; 3 1 0; 0 1 4] :
          kind===:qr ? T[2 0 0; 1 3 0; 0 1 4] : T[5 1 0; 1 4 1; 0 1 3]
        if T<:Complex && n>0
            if kind===:cholesky
                A[1, 2]+=T(0.2im)
                A[2, 1]-=T(0.2im)
            else
                A[n, n]+=T(0.3im)
            end
        end
        a=FaerSparseMatrix(sparse(A); format)
        order=reverse(collect(1:n))
        F=kind===:lu ?
          lu(a; algorithm, ordering = algorithm===:simplicial ? order : :default) :
          kind===:qr ?
          qr(a; algorithm, ordering = algorithm===:simplicial ? order : :default) :
          cholesky(Hermitian(a); algorithm, ordering = order)
        close(a)
        close(F.source)
        P=promote_type(T, R)
        for (f, ref) in ((F, A), (transpose(F), transpose(A)), (adjoint(F), adjoint(A)))
            B=ones(R, size(ref, 1), 2)
            expected=P.(ref)\P.(B)
            X=f\B
            @test eltype(X)===P
            @test Matrix(X)≈expected atol=1e-3 rtol=1e-3
            @test f\B[:, 1]≈expected[:, 1] atol=1e-3 rtol=1e-3
            dest=zeros(P, size(ref, 2), 2)
            @test ldiv!(dest, f, B)===dest
            @test dest≈expected atol=1e-3 rtol=1e-3
            close(X)
        end
        close(F)
    end
end
@testset "Mixed tall sparse QR and transposed minimum-norm reuse" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        R in (Float32, Float64, ComplexF32, ComplexF64)

        A=T[2 0; 1 3; 0 1]
        T<:Complex && (A[2, 1]+=T(0.3im))
        F=qr(FaerSparseMatrix(sparse(A)); algorithm = :simplicial, ordering = [2, 1])
        close(F.source)
        P=promote_type(T, R)
        for (f, ref) in ((F, A), (transpose(F), transpose(A)), (adjoint(F), adjoint(A)))
            B=ones(R, size(ref, 1), 2)
            @test Matrix(f\B)≈pinv(P.(ref))*P.(B) atol=1e-3 rtol=1e-3
        end
        close(F)
    end
end
