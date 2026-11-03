using Colmat, LinearAlgebra, SparseArrays, Test
@testset "Native operator tree scalar promotion" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        R in (Float32, Float64, ComplexF32, ComplexF64),
        format in (:dense, :csc, :csr, :diagonal, :identity)

        A=T[3 0; 0 2]
        T<:Complex && (A[1, 1]+=T(0.3im))
        a =
            format===:identity ? FaerLinearOperator(I, 2; T) :
            format===:diagonal ? FaerLinearOperator(Diagonal(diag(A))) :
            format===:dense ? FaerLinearOperator(A) :
            FaerLinearOperator(FaerSparseMatrix(sparse(A); format))
        format===:identity && (A=Matrix{T}(I, 2, 2))
        c=adjoint(a)*a
        B=R[1 2; 3 4]
        P=promote_type(T, R)
        x=c*B
        @test eltype(x)===P
        @test Matrix(x)≈P.(A'*A)*P.(B) atol=1e-3 rtol=1e-3
        other=FaerLinearOperator(B)
        nested=c*other
        @test eltype(nested)===P
        @test Matrix(nested)≈P.(A'*A)*P.(B) atol=1e-3 rtol=1e-3
        close(a)
        close(c)
        close(other)
        @test Matrix(nested)≈P.(A'*A)*P.(B) atol=1e-3 rtol=1e-3
        close(nested)
        close(x)
    end
end
