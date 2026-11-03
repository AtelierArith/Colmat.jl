using Colmat, LinearAlgebra, SparseArrays, Test
@testset "Native operator composition ownership and transforms" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        dims in ((0, 0, 0), (3, 0, 2), (3, 2, 4)),
        sparse_left in (false, true)

        m, k, n=dims
        A=reshape(T.(1:(m*k)), m, k)
        B=reshape(T.(1:(k*n)), k, n)
        T<:Complex && (A .*= T(1+0.2im); B .*= T(1-0.3im))
        a=FaerLinearOperator(sparse_left ? sparse(A) : A)
        b=FaerLinearOperator(B)
        c=a*b
        @test c isa FaerLinearOperator{T}
        @test size(c)==(m, n)
        @test Matrix(c)≈A*B atol=1e-3
        @test Matrix(transpose(c))≈transpose(A*B) atol=1e-3
        @test Matrix(adjoint(c))≈adjoint(A*B) atol=1e-3
        @test Matrix(conj(c))≈conj(A*B) atol=1e-3
        @test_throws ArgumentError c.coefficients
        d=copy(c)
        close(a)
        close(b)
        close(c)
        @test Matrix(d)≈A*B atol=1e-3
        e=d*FaerLinearOperator(I, n; T)
        @test Matrix(e)≈A*B atol=1e-3
        close(d)
        close(e)
    end
end
@testset "Composition iterative solve and validation" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        A=T[3 0; 1 2; 0 1]
        T<:Complex && (A[2, 1]+=T(0.2im))
        a=FaerLinearOperator(A)
        c=adjoint(a)*a
        b=ones(T, 2)
        @test cg(c, b).x≈(A'*A)\b atol=1e-3 rtol=1e-3
        X=ones(T, 2, 2)
        @test Matrix(c*X)≈A'*A*X atol=1e-3
        close(a)
        close(c)
    end
    a=FaerLinearOperator(ones(2, 3))
    b=FaerLinearOperator(ones(4, 2))
    @test_throws DimensionMismatch a*b
    close(a)
    @test_throws ArgumentError a*b
    close(b)
end
