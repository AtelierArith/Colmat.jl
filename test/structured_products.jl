using Test, Colmat, LinearAlgebra, SparseArrays
@testset "Native structured multiplication" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), n in (0, 1, 4)
        A=reshape(T.(1:(n*n)), n, n)
        T<:Complex && (A .+= T(0.5im))
        B=reverse(A; dims = 1)
        for W in
            (LowerTriangular, UpperTriangular, UnitLowerTriangular, UnitUpperTriangular)
            # Unused half and implicit diagonal are poisonous but must not enter the product.
            poisoned=copy(A)
            for j = 1:n, i = 1:n
                if (W in (LowerTriangular, UnitLowerTriangular) ? i<j : i>j) ||
                   (W in (UnitLowerTriangular, UnitUpperTriangular) && i==j)
                    poisoned[i, j]=T(NaN)
                end
            end
            native=W(FaerMatrix(A))
            # Native input loader rejects NaN; install unused poison via setters.
            for j = 1:n, i = 1:n
                if isnan(poisoned[i, j])
                    parent(native)[i, j]=poisoned[i, j]
                end
            end
            expected=Matrix(W(A))
            b=FaerMatrix(B)
            @test Matrix(native*b)≈expected*B
            @test Matrix(b*native)≈B*expected
            @test Matrix(W(A)*b)≈expected*B
            C=FaerMatrix(ones(T, n, n))
            @test mul!(C, native, b, T(2), T(3))===C
            @test Matrix(C)≈2expected*B .+3
            @test mul!(b, native, b)===b
            @test Matrix(b)≈expected*B
            for V in
                (LowerTriangular, UpperTriangular, UnitLowerTriangular, UnitUpperTriangular)
                right=V(FaerMatrix(B))
                @test Matrix(native*right)≈expected*Matrix(V(B))
            end
        end
    end
    @test isempty(
        Test.detect_ambiguities(
            Colmat,
            Base,
            LinearAlgebra,
            SparseArrays;
            recursive = true,
        ),
    )
end
@testset "Diagonal products and structured mul controls" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), n in (0, 1, 4)
        A=reshape(T.(1:(n*n)), n, n)
        d=T.(1:n)
        T<:Complex && (d .+= T(0.5im))
        D=FaerDiagonal(d)
        N=FaerMatrix(A)
        @test Matrix(D*N)≈Diagonal(d)*A
        @test Matrix(N*D)≈A*Diagonal(d)
        @test Matrix(Diagonal(d)*N)≈Diagonal(d)*A
        @test Matrix(N*Diagonal(d))≈A*Diagonal(d)
        for W in
            (LowerTriangular, UpperTriangular, UnitLowerTriangular, UnitUpperTriangular)
            Q=W(FaerMatrix(A))
            @test Matrix(D*Q)≈Diagonal(d)*Matrix(W(A))
            @test Matrix(Q*D)≈Matrix(W(A))*Diagonal(d)
            @test Matrix(D*W(A))≈Diagonal(d)*Matrix(W(A))
            @test Matrix(W(A)*D)≈Matrix(W(A))*Diagonal(d)
        end
        C=FaerMatrix(ones(T, n, n))
        @test mul!(C, D, N, 2, 3)===C
        @test Matrix(C)≈2Diagonal(d)*A .+3
        @test mul!(N, D, N)===N
        @test Matrix(N)≈Diagonal(d)*A
    end
    A=FaerMatrix([1.0 2; 3 4])
    B=FaerMatrix(ones(2, 2))
    A[1, 1]=NaN
    @test Matrix(mul!(B, LowerTriangular(A), B, 0, 2))==fill(2.0, 2, 2)
    C=FaerMatrix(zeros(2, 2))
    C[1, 1]=NaN
    @test Matrix(mul!(C, UpperTriangular(FaerMatrix([1.0 2; 3 4])), B, 1, 0))≈[6.0 6; 8 8]
    @test_throws DimensionMismatch LowerTriangular(FaerMatrix(ones(2, 2)))*FaerMatrix(
        ones(3, 2),
    )
    @test isempty(
        Test.detect_ambiguities(
            Colmat,
            Base,
            LinearAlgebra,
            SparseArrays;
            recursive = true,
        ),
    )
end
@testset "Structured rectangular, promotion and views" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        a=T[2 7 8; 3 4 9; 5 6 10]
        A=LowerTriangular(FaerMatrix(a))
        b=reshape(T.(1:6), 3, 2)
        B=FaerMatrix(b)
        @test Matrix(A*B)≈Matrix(LowerTriangular(a))*b
        c=permutedims(b)
        @test Matrix(FaerMatrix(c)*A)≈c*Matrix(LowerTriangular(a))
        big=FaerMatrix(zeros(T, 6, 4))
        dest=view(big, 6:-2:2, 1:2)
        @test mul!(dest, A, B)===dest
        @test Matrix(dest)≈Matrix(LowerTriangular(a))*b
        @test Matrix(A*view(big, 6:-2:2, 1:2))≈Matrix(LowerTriangular(a))*Matrix(dest)
        D=FaerDiagonal(T[2, 3, 4])
        @test Matrix(D*B)≈Diagonal(T[2, 3, 4])*b
        closed=FaerMatrix(ones(T, 3, 2))
        close(closed)
        @test_throws ArgumentError mul!(B, A, closed, 0, 0)
        @test_throws DimensionMismatch mul!(B, A, FaerMatrix(ones(T, 2, 2)), 0, 0)
    end
    A=LowerTriangular(FaerMatrix(Float32[1 2; 3 4]))
    B=FaerMatrix(ComplexF64[1 2im; 3 4])
    @test eltype(A*B)==ComplexF64
    @test Matrix(A*B)≈Matrix(A)*Matrix(B)
end
@testset "Structured coefficient promotion" begin
    A=LowerTriangular(FaerMatrix([1.0 2; 3 4]))
    B=FaerMatrix(ones(2, 2))
    C=FaerMatrix(ones(ComplexF64, 2, 2))
    @test mul!(C, A, B, 1im, 2im)===C
    @test Matrix(C)≈1im*Matrix(A)*Matrix(B) .+2im
    closed=FaerDiagonal([1.0, 2])
    close(closed)
    @test_throws ArgumentError mul!(C, closed, B, 0, 0)
end
