using Colmat, Test, LinearAlgebra

const DIAGONAL_TYPES=(Float32, Float64, ComplexF32, ComplexF64)
_diag_sample(::Type{T}, dims...) where {T} =
    T<:Complex ? T.(reshape(1:prod(dims), dims) .+ im .* reshape(prod(dims):-1:1, dims)) :
    T.(reshape(1:prod(dims), dims) .- prod(dims)/3)

@testset "Native diagonal products, solves and sums" begin
    for T in DIAGONAL_TYPES, (m, k) in ((0, 0), (1, 1), (3, 3), (3, 4))
        d=_diag_sample(T, m) .+ T(1)
        dk=_diag_sample(T, k) .+ T(2)
        a=_diag_sample(T, m, k)
        D, Dk, A=FaerDiagonal(d), FaerDiagonal(dk), FaerMatrix(a)
        V=view(A, :, :)
        for B in (A, V)
            @test (D*B) isa FaerMatrix{T}
            @test Matrix(D*B) ≈ Diagonal(d)*a
            @test Matrix(B*Dk) ≈ a*Diagonal(dk)
            @test Matrix(D\B) ≈ Diagonal(d)\a
            @test Matrix(B/Dk) ≈ a/Diagonal(dk)
            @test Matrix(Diagonal(d)*A) ≈ Diagonal(d)*a
            @test Matrix(A*Diagonal(dk)) ≈ a*Diagonal(dk)
        end
        @test Matrix(Diagonal(d)\A) ≈ Diagonal(d)\a
        @test Matrix(A/Diagonal(dk)) ≈ a/Diagonal(dk)
        if m==k
            @test Matrix(D+A) ≈ Diagonal(d)+a
            @test Matrix(A+D) ≈ a+Diagonal(d)
            @test Matrix(A-D) ≈ a-Diagonal(d)
            @test Matrix(D-V) ≈ Diagonal(d)-a
        else
            @test_throws DimensionMismatch D+A
            @test_throws DimensionMismatch Dk*A
            @test_throws DimensionMismatch A/D
        end
        @test Matrix(A) == a
        foreach(close, (D, Dk, A))
    end
end

@testset "Diagonal products of strided, adjoint and promoted operands" begin
    for T in DIAGONAL_TYPES
        a=_diag_sample(T, 5, 6)
        A=FaerMatrix(a)
        d=_diag_sample(T, 3) .+ T(1)
        D=FaerDiagonal(d)
        V=view(A, 5:-2:1, 2:2:6)
        @test Matrix(D*V) ≈ Diagonal(d)*a[5:-2:1, 2:2:6]
        @test Matrix(V*D) ≈ a[5:-2:1, 2:2:6]*Diagonal(d)
        @test Matrix(D\V) ≈ Diagonal(d)\a[5:-2:1, 2:2:6]
        # Promotion converts only the operand needing it.
        R=FaerDiagonal(Float32[1, 2, 4])
        P=promote_type(T, Float32)
        @test (R*V) isa FaerMatrix{P}
        @test Matrix(R*V) ≈ Diagonal(Float32[1, 2, 4])*a[5:-2:1, 2:2:6]
        @test Matrix(Diagonal([1, 2, 4])*A[1:3, 1:3]) ≈ Diagonal([1, 2, 4])*a[1:3, 1:3]
        foreach(close, (A, D, R))
    end
    A=FaerMatrix(ComplexF64[1+2im 3; 4im 5-1im])
    B=Matrix(A)
    D=FaerDiagonal(ComplexF64[2im, -1])
    @test Matrix(D*copy(A')) ≈ Diagonal([2im, -1])*B'
    close(A)
    close(D)
end

@testset "Julia Diagonal entries cross natively with non-finite values" begin
    for T in DIAGONAL_TYPES
        a=_diag_sample(T, 2, 2)
        A=FaerMatrix(a)
        d=T[Inf, NaN]
        @test isequal(Matrix(Diagonal(d)*A), Diagonal(d)*a)
        @test isequal(Matrix(A*Diagonal(d)), a*Diagonal(d))
        # A FaerDiagonal may hold non-finite entries after assignment.
        D=FaerDiagonal(T[1, 2])
        D[2, 2]=T(NaN)
        @test isequal(Matrix(D*A), Diagonal(T[1, NaN])*a)
        C=FaerMatrix{ComplexF64}(a)
        @test isequal(Matrix(D*C), Diagonal(ComplexF64.(T[1, NaN]))*ComplexF64.(a))
        foreach(close, (A, D, C))
    end
end

@testset "Native diagonal vector products and solves" begin
    for T in DIAGONAL_TYPES, n in (0, 1, 4)
        d=_diag_sample(T, n) .+ T(1)
        x=_diag_sample(T, n)
        D, v, r=FaerDiagonal(d), FaerVector(x), FaerRow(x)
        @test (D*v) isa FaerVector{T}
        @test Vector(D*v) ≈ Diagonal(d)*x
        @test Vector(D*view(v, :)) ≈ Diagonal(d)*x
        @test (r*D) isa FaerRow{T}
        @test Matrix(r*D) ≈ permutedims(x)*Diagonal(d)
        @test Vector(D\v) ≈ Diagonal(d)\x
        @test Matrix(r/D) ≈ permutedims(x)/Diagonal(d)
        n>1 && @test Vector(D*view(v, n:-1:1)) ≈ Diagonal(d)*reverse(x)
        @test_throws DimensionMismatch FaerDiagonal(T[1])*FaerVector(T[1, 2])
        foreach(close, (D, v, r))
    end
end

@testset "Singular diagonal solves throw and preserve inputs" begin
    for T in DIAGONAL_TYPES
        d=T[2, 0, 4]
        D=FaerDiagonal(d)
        a=_diag_sample(T, 3, 3)
        A=FaerMatrix(a)
        v=FaerVector(T[1, 2, 3])
        for f in (
            ()->D\A,
            ()->A/D,
            ()->D\v,
            ()->FaerRow(T[1, 2, 3])/D,
            ()->inv(D),
            ()->ldiv!(D, A),
            ()->rdiv!(A, D),
            ()->ldiv!(D, v),
            ()->Diagonal(d)\A,
            ()->ldiv!(Diagonal(d), A),
        )
            err=try
                f()
                nothing
            catch e
                e
            end
            @test err isa LinearAlgebra.SingularException
            @test err.info==2
        end
        @test Matrix(A)==a
        @test Vector(v)==T[1, 2, 3]
        @test Vector(diag(D))==d
        foreach(close, (D, A, v))
    end
end

@testset "Native diagonal inverse, adjoint copies and extraction" begin
    for T in DIAGONAL_TYPES, n in (0, 3)
        d=_diag_sample(T, n) .+ T(1)
        D=FaerDiagonal(d)
        I_=inv(D)
        @test I_ isa FaerDiagonal{T}
        @test Vector(diag(I_)) ≈ inv.(d)
        @test copy(D') isa FaerDiagonal{T}
        @test Vector(diag(copy(D'))) == conj.(d)
        @test Vector(diag(copy(transpose(D)))) == d
        a=_diag_sample(T, 3, n)
        A=FaerMatrix(a)
        @test diag(A) isa FaerVector{T}
        @test Vector(diag(A)) == diag(a)
        @test FaerDiagonal(A) isa FaerDiagonal{T}
        @test Vector(diag(FaerDiagonal(A))) == diag(a)
        @test Vector(diag(view(A, 3:-1:1, :))) == diag(a[3:-1:1, :])
        @test diag(A, 1) == diag(a, 1)
        foreach(close, (D, I_, A))
    end
end

@testset "In-place native diagonal updates" begin
    for T in DIAGONAL_TYPES
        a=_diag_sample(T, 3, 2)
        d3=_diag_sample(T, 3) .+ T(1)
        d2=_diag_sample(T, 2) .+ T(2)
        D3, D2=FaerDiagonal(d3), FaerDiagonal(d2)
        for (f, g) in (
            (A->lmul!(D3, A), A->lmul!(Diagonal(d3), A)),
            (A->rmul!(A, D2), A->rmul!(A, Diagonal(d2))),
            (A->ldiv!(D3, A), A->ldiv!(Diagonal(d3), A)),
            (A->rdiv!(A, D2), A->rdiv!(A, Diagonal(d2))),
        )
            A=FaerMatrix(a)
            ref=g(copy(a))
            @test f(A) === A
            @test Matrix(A) ≈ ref
            B=FaerMatrix(a)
            @test Matrix(g(B)) ≈ ref
            foreach(close, (A, B))
        end
        v=FaerVector(d3)
        lmul!(D3, v)
        @test Vector(v) ≈ d3 .* d3
        ldiv!(D3, v)
        @test Vector(v) ≈ d3
        r=FaerRow(d2)
        rmul!(r, D2)
        @test Matrix(r) ≈ permutedims(d2 .* d2)
        rdiv!(r, D2)
        @test Matrix(r) ≈ permutedims(d2)
        @test_throws Exception lmul!(D2, FaerVector(d3))
        # Aliased diagonal updates read a snapshot of the shared entries.
        E=FaerDiagonal(d3)
        lmul!(E, E)
        @test Vector(diag(E)) ≈ d3 .* d3
        rdiv!(E, E)
        @test Vector(diag(E)) ≈ ones(T, 3)
        W=FaerDiagonal(d3)
        ldiv!(W, diagonalview(W))
        @test Vector(diag(W)) ≈ ones(T, 3)
        # Views are updated through out-of-place native results.
        A=FaerMatrix(_diag_sample(T, 4, 4))
        ref=Matrix(A)
        V=view(A, 2:4, 1:2)
        lmul!(D3, V)
        ref[2:4, 1:2]=Diagonal(d3)*ref[2:4, 1:2]
        @test Matrix(A) ≈ ref
        rdiv!(V, D2)
        ref[2:4, 1:2]=ref[2:4, 1:2]/Diagonal(d2)
        @test Matrix(A) ≈ ref
        y=FaerVector{T}(undef, 3)
        ldiv!(y, D3, FaerVector(d3))
        @test Vector(y) ≈ ones(T, 3)
        foreach(close, (D3, D2, v, r, E, W, A, y))
    end
    # Real destinations reject nonreal diagonal entries before writing.
    A=FaerMatrix([1.0 2; 3 4])
    @test_throws InexactError lmul!(Diagonal([1im, 2]), A)
    @test Matrix(A)==[1.0 2; 3 4]
    close(A)
end

@testset "Diagonal operations require open native owners" begin
    D=FaerDiagonal([1.0, 2.0])
    A=FaerMatrix([1.0 2; 3 4])
    close(D)
    @test_throws ArgumentError D*A
    @test_throws ArgumentError inv(D)
    @test_throws ArgumentError lmul!(D, A)
    E=FaerDiagonal([1.0, 2.0])
    close(A)
    @test_throws ArgumentError E*A
    @test_throws ArgumentError lmul!(E, A)
    close(E)
end

@testset "Native outer products of vectors" begin
    for T in DIAGONAL_TYPES
        x=_diag_sample(T, 3)
        y=_diag_sample(T, 2) .+ T(1)
        v, w=FaerVector(x), FaerVector(y)
        @test (v*w') isa FaerMatrix{T}
        @test Matrix(v*w') ≈ x*y'
        @test Matrix(v*transpose(w)) ≈ x*transpose(y)
        @test Matrix(view(v, 3:-1:1)*w') ≈ reverse(x)*y'
        foreach(close, (v, w))
    end
end
