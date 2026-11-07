using Colmat, Test, LinearAlgebra

const STRUCTURED_TYPES=(Float32, Float64, ComplexF32, ComplexF64)
const WRAPPERS=(LowerTriangular, UpperTriangular, UnitLowerTriangular, UnitUpperTriangular)
_structured_sample(::Type{T}, m, n) where {T} =
    T<:Complex ? T.(reshape(1:(m*n), m, n) .+ 0.5im .* reshape((m*n):-1:1, m, n)) :
    T.(reshape(1:(m*n), m, n) ./ (m*n))
_block(dest, i, j) =
    dest===:full ? true :
    dest===:lower ? i>=j : dest===:upper ? i<=j : dest===:strict_lower ? i>j : i<j

@testset "Structured destination blocks" begin
    for T in STRUCTURED_TYPES,
        dest in (:full, :lower, :upper, :strict_lower, :strict_upper),
        n in (0, 1, 4)

        a=_structured_sample(T, n, n)
        b=reverse(a; dims = 2)
        for (α, β) in ((T(1), T(0)), (T(2), T(1)), (T(-1), T(3)), (T(0), T(2)))
            c0=fill(T(NaN), n, n)
            β!=0 && (c0=_structured_sample(T, n, n))
            C=FaerMatrix(β!=0 ? c0 : zeros(T, n, n))
            β==0 && foreach(I->C[I]=T(NaN), CartesianIndices(c0))
            A=LowerTriangular(FaerMatrix(a))
            B=FaerMatrix(b)
            @test structured_mul!(C, A, B, α, β; dest) === C
            full=α .* (Matrix(LowerTriangular(a))*b) .+ (β==0 ? 0 : β .* c0)
            got=Matrix(C)
            for j = 1:n, i = 1:n
                if _block(dest, i, j)
                    @test got[i, j] ≈ full[i, j]
                else
                    @test isequal(got[i, j], c0[i, j])
                end
            end
            foreach(close, (C, parent(A), B))
        end
    end
    C=FaerMatrix(zeros(2, 3))
    @test_throws DimensionMismatch structured_mul!(
        C,
        FaerMatrix(ones(2, 2)),
        FaerMatrix(ones(2, 3));
        dest = :lower,
    )
    @test_throws ArgumentError structured_mul!(
        C,
        FaerMatrix(ones(2, 2)),
        FaerMatrix(ones(2, 3));
        dest = :diagonal,
    )
    @test_throws ArgumentError structured_mul!(
        zeros(2, 3),
        FaerMatrix(ones(2, 2)),
        FaerMatrix(ones(2, 3)),
    )
    @test_throws ArgumentError structured_mul!(
        FaerMatrix(zeros(Float32, 2, 2)),
        FaerMatrix(ones(2, 2)),
        FaerMatrix(ones(2, 2));
        dest = :upper,
    )
    close(C)
end

@testset "Structured products borrow Julia and native parents" begin
    for T in STRUCTURED_TYPES, W in WRAPPERS
        a=_structured_sample(T, 5, 5)
        poisoned=copy(a)
        for j = 1:5, i = 1:5
            if (W in (LowerTriangular, UnitLowerTriangular) ? i<j : i>j) ||
               (W in (UnitLowerTriangular, UnitUpperTriangular) && i==j)
                poisoned[i, j]=T(NaN)
            end
        end
        b=_structured_sample(T, 5, 3)
        B=FaerMatrix(b)
        expected=Matrix(W(a))*b
        @test Matrix(W(poisoned)*B) ≈ expected
        @test (W(poisoned)*B) isa FaerMatrix{T}
        @test (W(transpose(FaerMatrix(a)))*B) isa FaerMatrix{T}
        # A strided Julia view and a non-strided adjoint parent.
        big=zeros(T, 10, 10)
        big[1:2:10, 2:2:10]=poisoned
        @test Matrix(W(view(big, 1:2:10, 2:2:10))*B) ≈ expected
        @test Matrix(W(copy(poisoned')')*B) ≈ expected
        # Transposed native parents.
        P=FaerMatrix(permutedims(a))
        @test Matrix(W(transpose(P))*B) ≈ expected
        @test Matrix(transpose(B)*W(transpose(P))') ≈ transpose(b)*Matrix(W(a))'
        foreach(close, (B, P))
    end
end

@testset "Structured vector products and in-place updates" begin
    for T in STRUCTURED_TYPES, W in WRAPPERS
        a=_structured_sample(T, 4, 4)
        x=_structured_sample(T, 4, 1)[:, 1]
        A=W(FaerMatrix(a))
        v=FaerVector(x)
        @test (A*v) isa FaerVector{T}
        @test Vector(A*v) ≈ Matrix(W(a))*x
        @test Vector(A*view(v, 4:-1:1)) ≈ Matrix(W(a))*reverse(x)
        y=FaerVector(ones(T, 4))
        @test mul!(y, A, v, T(2), T(3)) === y
        @test Vector(y) ≈ 2 .* (Matrix(W(a))*x) .+ 3
        # The destination is also the right operand.
        @test mul!(v, A, v) === v
        @test Vector(v) ≈ Matrix(W(a))*x
        foreach(close, (parent(A), v, y))
    end
end

@testset "Structured mul! aliasing, coefficients and promotion" begin
    for T in STRUCTURED_TYPES
        a=_structured_sample(T, 3, 3)
        A=FaerMatrix(a)
        B=FaerMatrix(ones(T, 3, 3))
        # The triangular operand's parent is the destination.
        mul!(A, UpperTriangular(A), B)
        @test Matrix(A) ≈ Matrix(UpperTriangular(a))*ones(T, 3, 3)
        A=FaerMatrix(a)
        V=view(A, 1:3, 1:3)
        mul!(V, LowerTriangular(A), transpose(A), T(1), T(1))
        @test Matrix(A) ≈ a .+ Matrix(LowerTriangular(a))*transpose(a)
        # Diagonal operands accumulate natively into views.
        big=FaerMatrix(zeros(T, 5, 5))
        D=FaerDiagonal(T[1, 2, 3])
        mul!(view(big, 5:-2:1, 1:3), D, FaerMatrix(a), T(2), T(0))
        @test Matrix(big)[5:-2:1, 1:3] ≈ 2 .* (Diagonal(T[1, 2, 3])*a)
        structured_mul!(
            big,
            Diagonal(ones(T, 5)),
            FaerMatrix(ones(T, 5, 5)),
            1,
            1;
            dest = :strict_upper,
        )
        ref=zeros(T, 5, 5)
        ref[5:-2:1, 1:3]=2 .* (Diagonal(T[1, 2, 3])*a)
        for j = 1:5, i = 1:(j-1)
            ref[i, j]+=1
        end
        @test Matrix(big) ≈ ref
        foreach(close, (A, B, big, D))
    end
    # A destination narrower than the promoted type uses the converting path.
    C=FaerMatrix(zeros(Float32, 2, 2))
    mul!(C, LowerTriangular(FaerMatrix([1.0 2; 3 4])), FaerMatrix(ones(2, 2)))
    @test Matrix(C) == Float32[1 1; 7 7]
    Z=FaerMatrix(zeros(ComplexF64, 2, 2))
    mul!(Z, UpperTriangular(FaerMatrix([1.0 2; 3 4])), FaerMatrix(ones(2, 2)), 1im, 0)
    @test Matrix(Z) ≈ 1im .* [3.0 3; 4 4]
    foreach(close, (C, Z))
end
