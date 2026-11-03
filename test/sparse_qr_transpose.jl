using Test, Colmat, LinearAlgebra, SparseArrays
@testset "Simplicial QR transpose minimum-norm solve" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        dims in ((0, 0), (3, 0), (3, 3), (5, 3)),
        format in (:csc, :csr),
        custom in (false, true)

        m, n=dims
        A=zeros(T, m, n)
        for j = 1:n
            A[j, j]=T(3+j)
            for i = (j+1):m
                A[i, j]=T(i+j)/T(5)
            end
        end
        T<:Complex && n>0 && (A[m, 1]+=T(0.7im))
        a=FaerSparseMatrix(sparse(A); format)
        F=qr(
            a;
            algorithm = :simplicial,
            ordering = custom ? reverse(collect(1:n)) : :identity,
        )
        close(a)
        for (G, M) in ((transpose(F), transpose(A)), (adjoint(F), adjoint(A)))
            B=reshape(T.(1:(n*2)), n, 2)
            T<:Complex && (B .+= T(0.3im))
            X=G\B
            reference=pinv(Matrix(M))*B
            @test X isa FaerMatrix{T}
            @test size(X)==(m, 2)
            @test Matrix(X)≈reference atol=1e-3 rtol=1e-3
            @test M*Matrix(X)≈B atol=1e-3 rtol=1e-3
            @test G\B[:, 1]≈reference[:, 1] atol=1e-3 rtol=1e-3
            dest=zeros(T, m, 2)
            @test ldiv!(dest, G, B)===dest
            @test dest≈reference atol=1e-3 rtol=1e-3
            @test_throws DimensionMismatch G\zeros(T, n+1, 1)
            close(X)
        end
        close(F)
        @test_throws ArgumentError adjoint(F)\zeros(T, n, 1)
    end
end
@testset "QR transpose factors survive source mutation" begin
    A=ComplexF64[1 0; 2im 3; 0 4]
    a=FaerSparseMatrix(sparse(A))
    F=qr(a; algorithm = :simplicial, ordering = [2, 1])
    fill!(nonzeros(F.source), 0)
    @test Matrix(adjoint(F)\ones(ComplexF64, 2, 1))≈pinv(A')*ones(2, 1)
    @test Matrix(transpose(F)\ones(ComplexF64, 2, 1))≈pinv(transpose(A))*ones(2, 1)
    close(F)
    close(a)
end

@testset "QR transpose mutation and promotion" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        A=T[4 1; 2 3]
        T<:Complex && (A[1, 2]+=T(0.4im))
        a=FaerSparseMatrix(sparse(A))
        F=qr(a; algorithm = :simplicial, ordering = [2, 1])
        for (G, M) in ((transpose(F), transpose(A)), (adjoint(F), adjoint(A)))
            B=ones(T, 2, 2)
            @test ldiv!(G, B)===B
            @test B≈M\ones(T, 2, 2) atol=1e-3
            E=G\zeros(T, 2, 0)
            @test size(E)==(2, 0)
            close(E)
            for R in (Float32, Float64, ComplexF32, ComplexF64)
                b=ones(R, 2, 1)
                x=G\b
                @test eltype(x)==promote_type(T, R)
                @test Matrix(x)≈M\b atol=1e-3
                close(x)
            end
        end
        close(F)
        close(a)
    end
    a=FaerSparseMatrix(sparse([1.0 0; 0 0; 0 1]))
    F=qr(a; algorithm = :simplicial)
    # A structurally empty second column tests a genuinely rank-deficient R.
    b=FaerSparseMatrix(sparse([1.0 0; 0 0; 0 0]))
    G=qr(b; algorithm = :simplicial)
    @test_throws Colmat.RustCall.RustError adjoint(G)\ones(2, 1)
    close(G)
    close(b)
    close(F)
    close(a)
end
