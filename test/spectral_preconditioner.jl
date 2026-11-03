using Colmat, LinearAlgebra, Test
@testset "Retained spectral inverse and pseudoinverse actions" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        R in (Float32, Float64, ComplexF32, ComplexF64),
        dims in ((0, 0), (3, 0), (0, 3), (3, 2), (2, 3), (3, 3))

        m, n=dims
        A=zeros(T, m, n)
        for i = 1:min(m, n)
            A[i, i]=T(i==2 ? 1e-5 : i+2)
        end
        T<:Complex && m>0 && n>0 && (A[1, 1]+=T(0.3im))
        F=svd(FaerMatrix(A); native = true)
        M=FaerLinearOperator(F; rtol = 1e-3)
        close(F)
        expected=pinv(A; rtol = 1e-3)
        @test size(M)==(n, m)
        for (op, ref) in (
            (M, expected),
            (transpose(M), transpose(expected)),
            (adjoint(M), adjoint(expected)),
        )
            B=ones(R, size(ref, 2), 2)
            P=promote_type(T, R)
            X=op*B
            @test eltype(X)===P
            @test Matrix(X)≈P.(ref)*P.(B) atol=1e-3 rtol=1e-3
            close(X)
        end
        G=copy(M)
        close(M)
        @test Matrix(G)≈expected atol=1e-3 rtol=1e-3
        close(G)
    end
end
@testset "Self-adjoint spectral actions and iteration" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        R in (Float32, Float64, ComplexF32, ComplexF64),
        n in (0, 1, 3)

        A=zeros(T, n, n)
        for i = 1:n
            A[i, i]=T(i==2 ? 1e-5 : isodd(i) ? -3 : 4)
        end
        T<:Complex && n>1 && (A[1, 3] = T(0.2im); A[3, 1] = T(-0.2im))
        F=eigen(Hermitian(FaerMatrix(A)); native = true)
        M=FaerLinearOperator(F; rtol = 1e-3)
        close(F)
        expected=pinv(A; rtol = 1e-3)
        B=ones(R, n, 2)
        P=promote_type(T, R)
        for (op, ref) in (
            (M, expected),
            (transpose(M), transpose(expected)),
            (adjoint(M), adjoint(expected)),
        )
            X=op*B
            @test eltype(X)===P
            @test Matrix(X)≈P.(ref)*P.(B) atol=1e-3 rtol=1e-3
            close(X)
        end
        close(M)
    end
    A=FaerMatrix([4.0 1; 1 3])
    for kind in (:svd, :eigen)
        F=kind===:svd ? svd(A; native = true) : eigen(Hermitian(A); native = true)
        M=FaerLinearOperator(F)
        @test cg(A, ones(2); M).x≈Matrix(A)\ones(2)
        @test_throws ArgumentError FaerLinearOperator(F; rtol = -1)
        close(F)
        @test_throws ArgumentError FaerLinearOperator(F)
        close(M)
    end
    close(A)
end
