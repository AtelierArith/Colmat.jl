using Test, LinearAlgebra, SparseArrays, Random, Colmat

@testset "Native sparse triangular solves" begin
    rng=MersenneTwister(481)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        format in (:csc, :csr),
        C in (LowerTriangular, UpperTriangular, UnitLowerTriangular, UnitUpperTriangular)
        # Both halves are deliberately present; the wrapper must ignore one.
        a=randn(rng, T, 7, 7)
        a[diagind(a)].+=T(5)
        s=FaerSparseMatrix(sparse(a); format)
        tri=C(s)
        reference=C(a)
        for R in (T, ComplexF64)
            rhs=randn(rng, R, 7, 3)
            native=FaerMatrix(rhs)
            expected=reference\rhs
            @test Matrix(tri\rhs) ≈ expected
            @test Matrix(tri\native) ≈ expected
            @test Matrix(native)==rhs
            vector=rhs[:, 1]
            @test tri\vector ≈ reference\vector
            @test tri\sparse(vector) ≈ reference\vector
            owned=FaerVector(vector)
            result=tri\owned
            @test result isa FaerVector
            @test Vector(result) ≈ reference\vector
            if R===T
                copyrhs=copy(native)
                @test ldiv!(tri, copyrhs)===copyrhs
                @test Matrix(copyrhs) ≈ expected
                output=FaerMatrix(zeros(T, 7, 3))
                @test ldiv!(output, tri, native)===output
                @test Matrix(output) ≈ expected
                alias=copy(native)
                @test ldiv!(alias, tri, alias)===alias
                @test Matrix(alias) ≈ expected
                v=copy(vector)
                @test ldiv!(tri, v)===v
                @test v ≈ reference\vector
            end
        end
        @test Matrix(s)==a
        @test_throws DimensionMismatch tri\zeros(T, 6, 2)
        @test_throws DimensionMismatch tri\zeros(T, 6)
        @test_throws DimensionMismatch ldiv!(zeros(T, 7, 2), tri, zeros(T, 7, 3))
        close(s)
        @test_throws ArgumentError tri\ones(T, 7)
    end
end
@testset "Sparse triangular diagonal conventions and empty shapes" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), format in (:csc, :csr)
        # No stored diagonal: unit wrappers must supply it.
        a=T[0 8 0; 2 0 9; 3 4 0]
        s=FaerSparseMatrix(sparse(a); format)
        for C in (UnitLowerTriangular, UnitUpperTriangular)
            @test C(s)\ones(T, 3) ≈ C(a)\ones(T, 3)
        end
        for C in (LowerTriangular, UpperTriangular)
            @test_throws SingularException C(s)\ones(T, 3)
        end
        # Stored zero diagonal is also singular, while the unit wrapper ignores NaN.
        s[1, 1]=T(NaN)
        for C in (UnitLowerTriangular, UnitUpperTriangular)
            @test all(isfinite, C(s)\ones(T, 3))
        end
        s[1, 1]=T(2)
        for C in (LowerTriangular, UpperTriangular)
            try
                C(s)\ones(T, 3)
                @test false
            catch e
                @test e isa SingularException
                @test e.info==2
            end
        end
        empty=FaerSparseMatrix(spzeros(T, 0, 0); format)
        for C in
            (LowerTriangular, UpperTriangular, UnitLowerTriangular, UnitUpperTriangular)
            @test size(C(empty)\zeros(T, 0, 3))==(0, 3)
            @test isempty(C(empty)\T[])
        end
        identity=FaerSparseMatrix(sparse(Matrix{T}(I, 3, 3)); format)
        @test size(LowerTriangular(identity)\zeros(T, 3, 0))==(3, 0)
    end
    @test isempty(
        detect_ambiguities(Colmat, Base, LinearAlgebra, SparseArrays; recursive = true),
    )
end

@testset "Structured right hand sides and sparse scaling" begin
    rng=MersenneTwister(983)
    for T in (Float32, Float64, ComplexF32, ComplexF64), format in (:csc, :csr)
        a=randn(rng, T, 4, 4)
        a[diagind(a)].+=T(5)
        s=FaerSparseMatrix(sparse(a); format)
        b=randn(rng, T, 4, 4)
        rights=(
            LowerTriangular(b),
            UpperTriangular(b),
            UnitLowerTriangular(b),
            UnitUpperTriangular(b),
            Bidiagonal(b, :U),
            UpperHessenberg(b),
            sparse(b),
            FaerSparseMatrix(sparse(b); format),
            view(FaerMatrix(b), :, :),
        )
        for C in
            (LowerTriangular, UpperTriangular, UnitLowerTriangular, UnitUpperTriangular)
            for rhs in rights
                @test Matrix(C(s)\rhs) ≈ C(a)\Matrix(rhs)
            end
            v=FaerVector(b[:, 1])
            @test Vector(C(s)\view(v, :)) ≈ C(a)\b[:, 1]
        end
        for C in
            (LowerTriangular, UpperTriangular, UnitLowerTriangular, UnitUpperTriangular)
            @test Matrix(adjoint(C(s))\b) ≈ adjoint(C(a))\b
            @test Matrix(transpose(C(s))\b) ≈ transpose(C(a))\b
        end
        @test s.epoch==0
    end
    # A dense copy would require 20 GB. This path stores only O(n) entries.
    n=50_000
    for format in (:csc, :csr)
        s=FaerSparseMatrix(sparse(1:n, 1:n, fill(2.0, n), n, n); format)
        @test LowerTriangular(s)\ones(n)==fill(0.5, n)
        @test UnitUpperTriangular(s)\ones(n)==ones(n)
        @test nnz(s)==n
    end
end
