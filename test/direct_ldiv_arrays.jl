using Colmat, Test, LinearAlgebra, Random

_dla_factor(kind, A) =
    kind===:lu ? lu!(A; scratch = FaerScratch(lu!, A)) :
    kind===:full_lu ? full_piv_lu!(A; scratch = FaerScratch(full_piv_lu!, A)) :
    kind===:qr ? qr!(A; scratch = FaerScratch(qr!, A)) :
    kind===:pivoted_qr ? qr!(A, ColumnNorm(); scratch = FaerScratch(qr!, A, ColumnNorm())) :
    kind===:cholesky ? cholesky!(Hermitian(A); scratch = FaerScratch(cholesky!, Hermitian(A))) :
    kind===:ldlt ? ldlt!(Hermitian(A); scratch = FaerScratch(ldlt!, Hermitian(A))) :
    bunchkaufman!(Hermitian(A); scratch = FaerScratch(bunchkaufman!, Hermitian(A)))

@testset "Direct factors solve Julia arrays and native vectors in place" begin
    rng=MersenneTwister(3)
    n=7
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        kind in (:lu, :full_lu, :qr, :pivoted_qr, :cholesky, :ldlt, :bunchkaufman)

        X=randn(rng, T, n, n)
        M=kind in (:cholesky, :ldlt, :bunchkaufman) ? X*X'+n*I : X+n*I
        F=_dla_factor(kind, FaerMatrix(copy(M)))
        tol=1000eps(real(T))
        rhs=Any[randn(rng, T, n), randn(rng, T, n, 3), FaerVector(randn(rng, T, n))]
        P=randn(rng, T, n+2, 3)
        push!(rhs, view(P, 2:(n+1), :))
        T<:Real && push!(rhs, randn(rng, Complex{T}, n), randn(rng, Complex{T}, n, 2))
        for b in rhs
            b0=Array(copy(b))
            @test ldiv!(F, b)===b
            @test isapprox(Array(b), M\b0; rtol = tol, atol = tol)
        end
        @test P[[1, n+2], :]==P[[1, n+2], :]  # rows outside the view untouched
        for (G, op) in ((F', adjoint), (transpose(F), transpose))
            b=randn(rng, T, n)
            b0=copy(b)
            ldiv!(G, b)
            @test isapprox(b, op(M)\b0; rtol = tol, atol = tol)
        end
        @test_throws DimensionMismatch ldiv!(F, zeros(T, n+1))
        T===Float64 && @test_throws ArgumentError ldiv!(F, zeros(Float32, n))
    end
    R=FaerMatrix(randn(rng, 6, 4))
    F=qr!(R; scratch = FaerScratch(qr!, R))
    @test_throws ArgumentError ldiv!(F, zeros(6))
    @test isempty(filter(Test.detect_ambiguities(Colmat; recursive = true)) do (m1, m2)
        any(m->occursin("direct_factorizations.jl", string(m.file)), (m1, m2))
    end)
end
