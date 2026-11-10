using Colmat, Test, LinearAlgebra

const BALANCED_TYPES = (Float32, Float64, ComplexF32, ComplexF64)
function _balanced_sample(::Type{T}) where {T}
    a = [1.0 1e4 0 0; 1e-4 2 1e3 0; 0 1e-3 3 1e2; 0 0 1e-2 4]
    T <: Complex && (a = a .+ 0.25im .* [0 1e4 0 0; 0 0 0 0; 0 0 0 1; 0 0 0 0])
    return T.(a)
end
_residual(a, v, λ) = norm(a * v - v * Diagonal(λ)) / max(norm(a), 1)
_lresidual(a, w, λ) = norm(w' * a - Diagonal(λ) * w') / max(norm(a), 1)

@testset "Balanced general EVD" begin
    team = FaerTeam(2)
    for T in BALANCED_TYPES
        a = _balanced_sample(T)
        A = FaerMatrix(a)
        tol = 100eps(real(T))
        ref = eigvals(a)
        for kw in ((;), (; team), (; scratch=FaerScratch(eigen, A; side=:both)))
            E = eigen(A; permute=true, scale=true, kw...)
            @test E isa Eigen
            @test _residual(a, E.vectors, E.values) < tol
            @test all(j -> isapprox(norm(E.vectors[:, j]), 1; atol=tol), axes(E.vectors, 2))
            @test sort(E.values; by=x -> (real(x), imag(x))) ≈
                sort(complex.(ref); by=x -> (real(x), imag(x)))
            v = eigvals(A; permute=true, scale=true, kw...)
            @test v ≈ E.values
            L = eigen(A; side=:left, permute=true, scale=true, kw...)
            @test _lresidual(a, L.vectors, L.values) < tol
            B = eigen(A; side=:both, permute=true, scale=true, kw...)
            @test _residual(a, B.right, B.values) < tol
            @test _lresidual(a, B.left, B.values) < tol
            haskey(kw, :scratch) && close(kw.scratch)
        end
        # sortby orders values and vectors consistently.
        by = x -> -real(x)
        E = eigen(A; permute=true, scale=true, sortby=by)
        @test issorted(E.values; by)
        @test _residual(a, E.vectors, E.values) < tol
        @test issorted(eigvals(A; permute=true, scale=true, sortby=by); by)
        # The input is not modified and the default stays unbalanced.
        @test Matrix(A) == a
        @test eigvals(A) ≈ eigvals(A; permute=false, scale=false)
        @test_throws ArgumentError eigen(A; permute=true)
        @test_throws ArgumentError eigvals(A; scale=true)
        @test_throws ArgumentError eigen(A; side=:up, permute=true, scale=true)
        close(A)
    end
    E = eigen(FaerMatrix(zeros(0, 0)); permute=true, scale=true)
    @test isempty(E.values)
    bad = FaerMatrix([1.0 2; 3 4])
    bad[1, 2] = NaN
    @test_throws Exception eigen(bad; permute=true, scale=true)
    close(bad)
    close(team)
end
