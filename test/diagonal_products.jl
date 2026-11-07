using Test, Colmat, LinearAlgebra
@testset "Native diagonal scaling products" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), (m, n) in ((0, 0), (1, 3), (4, 3))
        a=reshape(T.(1:(m*n)), m, n)
        T<:Complex && (a .+= T(0.5im))
        dl, dr=T.(1:m) .+ 1, T.(n:-1:1) .- T(0.25)
        N=FaerMatrix(a)
        for (L, R) in ((FaerDiagonal(dl), FaerDiagonal(dr)), (Diagonal(dl), Diagonal(dr)))
            left=L*N
            @test left isa FaerMatrix{T}
            @test Matrix(left)≈Diagonal(dl)*a
            right=N*R
            @test right isa FaerMatrix{T}
            @test Matrix(right)≈a*Diagonal(dr)
            C=FaerMatrix(zeros(T, m, n))
            # The native loader rejects NaN; poison the destination via setters.
            for j = 1:n, i = 1:m
                C[i, j]=T(NaN)
            end
            # beta == 0 never reads the destination.
            @test mul!(C, L, N)===C
            @test Matrix(C)≈Diagonal(dl)*a
            C=FaerMatrix(ones(T, m, n))
            @test mul!(C, N, R, T(2), T(3))===C
            @test Matrix(C)≈2a*Diagonal(dr) .+3
            # Destination aliasing the dense operand is computed before writing.
            M=FaerMatrix(a)
            @test mul!(M, L, M, 2, 1)===M
            @test Matrix(M)≈2Diagonal(dl)*a+a
            M=FaerMatrix(a)
            @test mul!(M, M, R)===M
            @test Matrix(M)≈a*Diagonal(dr)
        end
    end
end
@testset "Diagonal products with views, promotion and boundaries" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        big=reshape(T.(1:48), 6, 8)
        T<:Complex && (big .+= T(1im))
        B=FaerMatrix(big)
        V=view(B, 6:-2:2, 2:3:8)
        v=big[6:-2:2, 2:3:8]
        D3, D3j=FaerDiagonal(T[2, 3, 4]), Diagonal(T[2, 3, 4])
        @test Matrix(D3*V)≈D3j*v
        @test Matrix(V*D3)≈v*D3j
        @test Matrix(D3j*V)≈D3j*v
        # Strided view destination aliasing a disjoint part of the same owner.
        dest=view(B, 5:-2:1, 2:3:8)
        @test mul!(dest, D3, V, 1, 1)===dest
        @test Matrix(dest)≈D3j*v+big[5:-2:1, 2:3:8]
        @test Matrix(B)[6:-2:2, 2:3:8]==v
        # Overlapping alias.
        B=FaerMatrix(big)
        V=view(B, 6:-2:2, 2:3:8)
        @test mul!(V, V, D3)===V
        @test Matrix(V)≈v*D3j
        @test Matrix(B)[1:2:5, :]==big[1:2:5, :]
        # A one-row view of a native row is a valid destination.
        row=FaerRow(T[1, 2, 3])
        rv=view(row, 1:1, 1:3)
        @test mul!(rv, FaerDiagonal(T[5]), FaerMatrix(T[1 1 1]), 1, 2)===rv
        @test Vector(row)≈T[7, 9, 11]
        # alpha == 0 never reads either operand.
        D=FaerDiagonal(T[1, 2])
        D[1, 1]=T(NaN)
        N=FaerMatrix(T[1 2; 3 4])
        N[1, 2]=T(NaN)
        C=FaerMatrix(ones(T, 2, 2))
        @test Matrix(mul!(C, D, N, 0, 3))==fill(T(3), 2, 2)
        @test Matrix(mul!(C, N, Diagonal(T[NaN, 1]), 0, 0))==zeros(T, 2, 2)
    end
    # Mixed precision promotes like Julia; non-finite Julia diagonals stay valid.
    D=FaerDiagonal(Float32[1, 2])
    N=FaerMatrix(ComplexF64[1 2im; 3 4])
    @test eltype(D*N)==ComplexF64
    @test Matrix(D*N)≈Diagonal(Float32[1, 2])*Matrix(N)
    @test eltype(N*Diagonal([1, 2]))==ComplexF64
    @test Matrix(Diagonal([Inf, 1])*FaerMatrix([1.0 2; 3 4]))==[Inf Inf; 3 4]
    C=FaerMatrix(zeros(ComplexF64, 2, 2))
    @test Matrix(mul!(C, D, N, 1im, 0))≈1im*Diagonal(Float32[1, 2])*Matrix(N)
    @test_throws InexactError mul!(FaerMatrix(zeros(2, 2)), FaerDiagonal([1.0, 2]), FaerMatrix(ones(2, 2)), 1im, 0)
    R=FaerMatrix(zeros(Float32, 2, 2))
    @test Matrix(mul!(R, FaerDiagonal([1.0, 2]), FaerMatrix(ones(2, 2))))==Float32[1 1; 2 2]
    # Triangular wrappers of native matrices are scaled without densifying.
    for W in (LowerTriangular, UpperTriangular, UnitLowerTriangular, UnitUpperTriangular)
        A=FaerMatrix([1.0 2 3; 4 5 6; 7 8 9])
        for j = 1:3, i = 1:3
            (W in (LowerTriangular, UnitLowerTriangular) ? i<j : i>j) && (A[i, j]=NaN)
            W in (UnitLowerTriangular, UnitUpperTriangular) && i==j && (A[i, j]=NaN)
        end
        expected=Matrix(W([1.0 2 3; 4 5 6; 7 8 9]))
        d=[2.0, 3, 4]
        @test Matrix(FaerDiagonal(d)*W(A))≈Diagonal(d)*expected
        @test Matrix(W(A)*Diagonal(d))≈expected*Diagonal(d)
        C=FaerMatrix(ones(3, 3))
        @test Matrix(mul!(C, W(A), FaerDiagonal(d), 2, 1))≈2expected*Diagonal(d) .+1
    end
    closed=FaerMatrix(ones(2, 2))
    close(closed)
    @test_throws ArgumentError FaerDiagonal([1.0, 2])*closed
    @test_throws ArgumentError mul!(FaerMatrix(ones(2, 2)), Diagonal([1.0, 2]), closed, 0, 0)
    D=FaerDiagonal([1.0, 2])
    close(D)
    @test_throws ArgumentError D*FaerMatrix(ones(2, 2))
    @test_throws DimensionMismatch FaerDiagonal([1.0, 2])*FaerMatrix(ones(3, 2))
    @test_throws DimensionMismatch FaerMatrix(ones(3, 2))*Diagonal([1.0, 2, 3])
    @test_throws DimensionMismatch mul!(
        FaerMatrix(ones(3, 2)),
        FaerDiagonal([1.0, 2]),
        FaerMatrix(ones(2, 2)),
    )
end
@testset "Diagonal times native column and row times diagonal" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), n in (0, 1, 3)
        d=T.(1:n) .+ 1
        T<:Complex && (d .+= T(0.5im))
        v=T.(n:-1:1)
        D=FaerDiagonal(d)
        x=D*FaerVector(v)
        @test x isa FaerVector{T}
        @test Vector(x)≈d .* v
        y=FaerRow(v)*D
        @test y isa FaerRow{T}
        @test Vector(y)≈v .* d
    end
    @test eltype(FaerDiagonal(Float32[1, 2])*FaerVector(ComplexF64[1, 1im]))==ComplexF64
    @test Vector(FaerRow([1.0, 2])*FaerDiagonal(ComplexF32[1im, 2]))≈[1im, 4]
    @test_throws DimensionMismatch FaerDiagonal([1.0, 2])*FaerVector([1.0])
    @test_throws DimensionMismatch FaerRow([1.0])*FaerDiagonal([1.0, 2])
end
@testset "Diagonal products stay off the Julia heap" begin
    n=256
    for T in (Float64, ComplexF32)
        D=FaerDiagonal(T.(1:n))
        J=Diagonal(T.(1:n))
        N=FaerMatrix(ones(T, n, n))
        C=FaerMatrix(ones(T, n, n))
        f1()=D*N
        f2()=N*J
        f3()=mul!(C, D, N, 2, 3)
        f4()=mul!(N, N, D)
        for f in (f1, f2, f3, f4)
            f()
            f()
            # The previous dense-snapshot path allocated several n^2 buffers.
            @test @allocated(f())<n*n*sizeof(T)÷4
        end
    end
end
