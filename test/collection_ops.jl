using Colmat, Test, LinearAlgebra

const COLLECTION_TYPES=(Float32, Float64, ComplexF32, ComplexF64)
_cop_sample(::Type{T}, dims...) where {T} =
    T<:Complex ? T.(reshape(1:prod(dims), dims) .+ im .* reshape(prod(dims):-1:1, dims)) :
    T.(reshape(1:prod(dims), dims) .- prod(dims)/3)
# The method serving `f(args...)` is Colmat's native one, not a generic fallback.
_colmat_method(f, args...) = parentmodule(which(f, typeof.(args))) === Colmat

@testset "Diagonal with diagonal operands stays a native diagonal" begin
    for T in COLLECTION_TYPES, n in (0, 1, 3)
        d=_cop_sample(T, n) .+ T(1)
        e=_cop_sample(T, n) .+ T(2)
        D, E=FaerDiagonal(d), FaerDiagonal(e)
        J=Diagonal(e)
        for (f, ref) in (
            (+, Diagonal(d)+J),
            (-, Diagonal(d)-J),
            (*, Diagonal(d)*J),
            (\, Diagonal(d)\J),
            (/, Diagonal(d)/J),
        )
            for (x, y, r) in ((D, J, ref), (D, E, ref), (Diagonal(d), E, ref))
                @test _colmat_method(f, x, y)
                z=f(x, y)
                @test z isa FaerDiagonal{T}
                @test Vector(diag(z)) ≈ diag(r)
            end
            @test Vector(diag(f(J, D))) ≈ diag(f(J, Diagonal(d)))
        end
        # Self division reads the operand before writing the fresh result.
        @test Vector(diag(D\D)) ≈ ones(T, n)
        @test Vector(diag(D)) == d
        foreach(close, (D, E))
    end
    # Promotion follows Julia, including integer Julia diagonals.
    D=FaerDiagonal(Float32[1, 2])
    @test (D+Diagonal(ComplexF64[1im, 2])) isa FaerDiagonal{ComplexF64}
    @test (Diagonal([1, 2])*D) isa FaerDiagonal{Float32}
    @test Vector(diag(Diagonal([1, 2])\D)) ≈ Float32[1, 1]
    @test_throws DimensionMismatch D+Diagonal([1.0])
    @test_throws DimensionMismatch D*FaerDiagonal([1.0])
    # Singular divisors throw with Julia's index and leave the operands.
    Z=FaerDiagonal([2.0, 0.0])
    for f in (()->Z\D, ()->D/Z, ()->Diagonal([2.0, 0.0])\D, ()->D/Diagonal([2.0, 0.0]))
        err=try
            f()
            nothing
        catch e
            e
        end
        @test err isa LinearAlgebra.SingularException
        @test err.info==2
    end
    @test Vector(diag(D))==Float32[1, 2]
    close(Z)
    close(D)
    @test_throws ArgumentError D+Diagonal(Float32[1, 2])
    @test_throws ArgumentError Diagonal(Float32[1, 2])\D
end

@testset "Transposed and adjoint native diagonals" begin
    for T in COLLECTION_TYPES
        d=_cop_sample(T, 3) .+ T(1)
        a=_cop_sample(T, 3, 3)
        x=_cop_sample(T, 3)
        D, A, v, r=FaerDiagonal(d), FaerMatrix(a), FaerVector(x), FaerRow(x)
        J=Diagonal(d)
        for (W, R) in ((D', J'), (transpose(D), transpose(J)))
            @test _colmat_method(*, W, A)
            @test (W*A) isa FaerMatrix{T}
            @test Matrix(W*A) ≈ R*a
            @test Matrix(A*W) ≈ a*R
            @test Matrix(W*view(A, 3:-1:1, :)) ≈ R*a[3:-1:1, :]
            @test Matrix(W\A) ≈ R\a
            @test Matrix(A/W) ≈ a/R
            @test Matrix(W+A) ≈ R+a
            @test Matrix(A-W) ≈ a-R
            @test (W*v) isa FaerVector{T}
            @test Vector(W*v) ≈ R*x
            @test Vector(W\v) ≈ R\x
            @test (r*W) isa FaerRow{T}
            @test Matrix(r*W) ≈ permutedims(x)*R
            @test (W*D) isa FaerDiagonal{T}
            @test Vector(diag(W*D)) ≈ diag(R*J)
            @test Vector(diag(D+W)) ≈ diag(J+R)
            @test Vector(diag(W\J)) ≈ diag(R\J)
            C=FaerMatrix(zeros(T, 3, 3))
            @test _colmat_method(mul!, C, W, A, true, false)
            @test Matrix(mul!(C, W, A)) ≈ R*a
            @test Matrix(mul!(C, A, W, T(2), T(0))) ≈ 2*a*R
            close(C)
        end
        foreach(close, (D, A, v, r))
        @test_throws ArgumentError D'*FaerMatrix(a)
        @test_throws ArgumentError transpose(D)*FaerVector(x)
    end
end

@testset "Diagonal scaling of transposed and adjoint dense operands" begin
    for T in COLLECTION_TYPES
        a=_cop_sample(T, 3, 4)
        A=FaerMatrix(a)
        d3, d4=_cop_sample(T, 3) .+ T(1), _cop_sample(T, 4) .+ T(2)
        D3, D4=FaerDiagonal(d3), FaerDiagonal(d4)
        for (W, w) in ((A', a'), (transpose(A), transpose(a)), (view(A, :, 4:-1:2)', a[:, 4:-1:2]'))
            k=size(w, 1)
            Dk=k==4 ? D4 : FaerDiagonal(d4[1:k])
            dk=k==4 ? d4 : d4[1:k]
            @test _colmat_method(*, Dk, W)
            @test (Dk*W) isa FaerMatrix{T}
            @test Matrix(Dk*W) ≈ Diagonal(dk)*w
            @test Matrix(W*D3) ≈ w*Diagonal(d3)
            @test Matrix(Diagonal(dk)*W) ≈ Diagonal(dk)*w
            C=FaerMatrix(zeros(T, size(w)))
            @test mul!(C, Dk, W, T(2), T(0)) === C
            @test Matrix(C) ≈ 2*Diagonal(dk)*w
            k==4 || close(Dk)
        end
        # A destination aliasing the transposed operand reads a snapshot.
        S=FaerMatrix(_cop_sample(T, 3, 3))
        s=Matrix(S)
        mul!(S, D3, S')
        @test Matrix(S) ≈ Diagonal(d3)*s'
        foreach(close, (A, D3, D4, S))
    end
end

@testset "Native diagonal products of mixed scalar types" begin
    for (S, T) in ((Float32, Float64), (Float32, ComplexF32), (Float64, ComplexF64))
        a=_cop_sample(S, 3, 3)
        A=FaerMatrix(a)
        d=_cop_sample(T, 3) .+ T(1)
        D=FaerDiagonal(d)
        @test (D*A) isa FaerMatrix{T}
        @test Matrix(D*A) ≈ Diagonal(d)*a
        @test Matrix(A*D) ≈ a*Diagonal(d)
        @test Matrix(D*A') ≈ Diagonal(d)*a'
        @test Matrix(D\view(A, 3:-1:1, :)) ≈ Diagonal(d)\a[3:-1:1, :]
        @test Matrix(A/D) ≈ a/Diagonal(d)
        @test Matrix(D+A) ≈ Diagonal(d)+a
        @test Matrix(A) == a
        foreach(close, (A, D))
    end
end

@testset "Diagonal products into native vector destinations" begin
    for T in COLLECTION_TYPES, n in (0, 3)
        d=_cop_sample(T, n) .+ T(1)
        x=_cop_sample(T, n)
        D, v=FaerDiagonal(d), FaerVector(x)
        y=FaerVector(fill(T(1), n))
        @test _colmat_method(mul!, y, D, v)
        @test mul!(y, D, v) === y
        @test Vector(y) ≈ d .* x
        @test mul!(y, D, v, T(2), T(3)) === y
        @test Vector(y) ≈ 2 .* d .* x .+ 3 .* (d .* x)
        @test mul!(y, Diagonal(d), view(v, :)) === y
        @test Vector(y) ≈ d .* x
        r, z=FaerRow(x), FaerRow(zeros(T, n))
        @test mul!(z, r, D) === z
        @test Matrix(z) ≈ permutedims(x .* d)
        # Destinations aliasing an operand read a snapshot.
        mul!(v, D, v)
        @test Vector(v) ≈ d .* x
        if n>0
            V=FaerVector(x)
            mul!(view(V, n:-1:1), D, V)
            @test Vector(V) ≈ reverse(d .* x)
            E=FaerDiagonal(d)
            mul!(diagonalview(E), E, FaerVector(x))
            @test Vector(diag(E)) ≈ d .* x
            foreach(close, (V, E))
        end
        # Row slices of matrices are (transposed) native vector destinations.
        A=FaerMatrix(_cop_sample(T, 2, n))
        a=Matrix(A)
        mul!(view(A, 2, :), D, v)
        a[2, :]=d .* Vector(v)
        @test Matrix(A) ≈ a
        mul!(view(A, 2, :), D, v, T(2), T(1))
        a[2, :]=2 .* d .* Vector(v) .+ a[2, :]
        @test Matrix(A) ≈ a
        @test Vector(D*view(A, 1, :)) ≈ d .* a[1, :]
        mul!(view(A, 2, :), D, view(A, 2, :))
        a[2, :]=d .* a[2, :]
        @test Matrix(A) ≈ a
        close(A)
        # Diagonal destinations.
        E=FaerDiagonal(fill(T(1), n))
        @test _colmat_method(mul!, E, D, Diagonal(d))
        @test mul!(E, D, Diagonal(d), T(2), T(1)) === E
        @test Vector(diag(E)) ≈ 2 .* d .* d .+ 1
        mul!(E, E, D)
        @test Vector(diag(E)) ≈ (2 .* d .* d .+ 1) .* d
        close(E)
        @test_throws DimensionMismatch mul!(FaerVector(zeros(T, n+1)), D, v)
        foreach(close, (D, v, y, r, z))
    end
end

@testset "Row-vector products and quotients with diagonals" begin
    for T in COLLECTION_TYPES
        d=_cop_sample(T, 3) .+ T(1)
        x=_cop_sample(T, 3)
        D, v=FaerDiagonal(d), FaerVector(x)
        for (f, g) in ((*, *), (/, /))
            @test _colmat_method(f, v', D)
            a=f(v', D)
            @test a isa LinearAlgebra.Adjoint{T,<:FaerVector{T}}
            @test collect(a) ≈ g(x', Diagonal(d))
            t=f(transpose(v), D)
            @test t isa LinearAlgebra.Transpose{T,<:FaerVector{T}}
            @test collect(t) ≈ g(transpose(x), Diagonal(d))
            @test collect(f(v', Diagonal(d))) ≈ g(x', Diagonal(d))
        end
        @test (v'*D*v) ≈ x'*Diagonal(d)*x
        foreach(close, (D, v))
    end
    D=FaerDiagonal(Float32[1, 2])
    @test (FaerVector(ComplexF64[1im, 2])'*D) isa LinearAlgebra.Adjoint{ComplexF64}
    @test_throws DimensionMismatch FaerVector([1.0])'*D
    close(D)
end

@testset "Native finiteness predicates and maps of collections" begin
    for T in COLLECTION_TYPES
        for C in (FaerVector, FaerRow, FaerDiagonal)
            x=C(T[1, 2, 3])
            @test _colmat_method(all, isfinite, x)
            @test all(isfinite, x)
            @test !any(isnan, x)
            x2=C(T[1, Inf, 3])
            @test !all(isfinite, x2)
            @test !any(isnan, x2)
            x3=C(T[1, NaN, 3])
            @test any(isnan, x3)
            foreach(close, (x, x2, x3))
            @test_throws ArgumentError all(isfinite, x)
        end
        d=_cop_sample(T, 3)
        D=FaerDiagonal(d)
        M=map(x->2x, D)
        @test M isa FaerDiagonal{T}
        @test Vector(diag(M)) == 2 .* d
        N=map(x->x+1, D)
        @test N isa FaerMatrix{T}
        @test Matrix(N) == map(x->x+1, Diagonal(d))
        @test map(abs, D) isa FaerDiagonal{real(T)}
        @test map(x->x!=0, D) isa Diagonal{Bool}
        foreach(close, (D, M, N))
    end
end

@testset "Row arithmetic with adjoint and transposed columns" begin
    for T in COLLECTION_TYPES
        x=_cop_sample(T, 3)
        y=_cop_sample(T, 3) .+ T(1)
        r, v=FaerRow(x), FaerVector(y)
        for (W, w) in ((v', y'), (transpose(v), transpose(y)))
            @test _colmat_method(+, r, W)
            @test (r+W) isa FaerRow{T}
            @test Matrix(r+W) ≈ permutedims(x)+w
            @test (W-r) isa FaerRow{T}
            @test Matrix(W-r) ≈ w-permutedims(x)
            @test Matrix(r-W) ≈ permutedims(x)-w
        end
        u=FaerVector(x)
        @test (u*r) isa FaerMatrix{T}
        @test Matrix(u*r) ≈ x*permutedims(x)
        @test_throws DimensionMismatch r+FaerVector(T[1, 2])'
        foreach(close, (r, v, u))
    end
    @test (FaerRow(Float32[1, 2])+FaerVector(ComplexF64[1, 2im])') isa FaerRow{ComplexF64}
end

@testset "Collection operations add no ambiguities" begin
    ambiguities=Test.detect_ambiguities(Colmat, Base, LinearAlgebra; recursive = true)
    @test isempty(filter(a->a[1].module===Colmat||a[2].module===Colmat, ambiguities))
end
