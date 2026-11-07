using Colmat, Test, LinearAlgebra, SparseArrays

const TRANSPOSED_TYPES=(Float32, Float64, ComplexF32, ComplexF64)
_tr_sample(::Type{T}, m, n) where {T} =
    T.([sin(i+2j) + (i==j)*m + (T<:Complex ? im*cos(i*j) : 0) for i = 1:m, j = 1:n])
_vkey(x) = (round(real(x); digits = 4), round(imag(x); digits = 4))

@testset "Factorizations and norms of transposed native matrices" begin
    for T in TRANSPOSED_TYPES, W in (adjoint, transpose)
        n=12
        a=_tr_sample(T, n, n)
        A=FaerMatrix(a)
        for X in (A, view(A, :, :))
            x=W(X)
            ref=W(a)
            tol=200eps(real(T))*n
            @test Matrix(x) == ref
            @test isapprox(det(lu(x)), det(lu(ref)); rtol = tol)
            @test isapprox(abs.(Matrix(qr(x).R)), abs.(qr(ref).R); rtol = tol)
            @test isapprox(svd(x).S, svd(ref).S; rtol = tol)
            @test isapprox(svdvals(x), svdvals(ref); rtol = tol)
            @test isapprox(
                sort(eigvals(x); by = _vkey),
                sort(eigvals(ref); by = _vkey);
                rtol = tol,
            )
            @test isapprox(
                sort(eigen(x).values; by = _vkey),
                sort(eigvals(ref); by = _vkey);
                rtol = tol,
            )
            @test isapprox(
                sort(schur(x).values; by = _vkey),
                sort(eigvals(ref); by = _vkey);
                rtol = tol,
            )
            @test isapprox(Matrix(pinv(x)), pinv(ref); rtol = sqrt(tol))
            @test rank(x) == rank(ref)
            for p in (1, 2, Inf)
                @test isapprox(opnorm(x, p), opnorm(ref, p); rtol = tol)
                @test isapprox(norm(x, p), norm(ref, p); rtol = tol)
            end
        end
        @test Matrix(A) == a
        close(A)
    end
end

@testset "mul! with transposed native operands writes in place" begin
    for T in TRANSPOSED_TYPES
        a=_tr_sample(T, 6, 5)
        b=_tr_sample(T, 6, 3)
        A, B=FaerMatrix(a), FaerMatrix(b)
        C=FaerMatrix(ones(T, 5, 3))
        @test mul!(C, A', B) === C
        @test Matrix(C) ≈ a'*b
        mul!(C, transpose(A), B, T(2), T(1))
        @test Matrix(C) ≈ a'*b .+ 2 .* transpose(a)*b
        D=FaerMatrix(zeros(T, 6, 6))
        mul!(D, A, A')
        @test Matrix(D) ≈ a*a'
        mul!(view(D, 1:5, 1:5), A', A)
        @test Matrix(D)[1:5, 1:5] ≈ a'*a
        y=FaerVector(zeros(T, 5))
        mul!(y, A', FaerVector(b[:, 1]), T(1), T(0))
        @test Vector(y) ≈ a'*b[:, 1]
        # The destination shares storage with an operand.
        S=FaerMatrix(_tr_sample(T, 4, 4))
        s=Matrix(S)
        mul!(S, S', S)
        @test Matrix(S) ≈ s'*s
        foreach(close, (A, B, C, D, y, S))
    end
    n=400
    A=FaerMatrix(_tr_sample(Float64, n, n))
    B=FaerMatrix(_tr_sample(Float64, n, 4))
    C=FaerMatrix(zeros(n, 4))
    mul!(C, A', B)
    lu(A')
    @test @allocated(mul!(C, A', B)) < 64*1024
    @test @allocated(lu(A')) < 64*1024
    @test isempty(
        Test.detect_ambiguities(
            Colmat,
            Base,
            LinearAlgebra,
            SparseArrays;
            recursive = true,
        ),
    )
    foreach(close, (A, B, C))
end
