using Colmat, Test, LinearAlgebra, Random

_dfo_factor(kind, A) =
    kind===:lu ? lu!(A; scratch = FaerScratch(lu!, A)) :
    kind===:full_lu ? full_piv_lu!(A; scratch = FaerScratch(full_piv_lu!, A)) :
    kind===:qr ? qr!(A; scratch = FaerScratch(qr!, A)) :
    kind===:pivoted_qr ? qr!(A, ColumnNorm(); scratch = FaerScratch(qr!, A, ColumnNorm())) :
    kind===:cholesky ? cholesky!(Hermitian(A); scratch = FaerScratch(cholesky!, Hermitian(A))) :
    kind===:ldlt ? ldlt!(Hermitian(A); scratch = FaerScratch(ldlt!, Hermitian(A))) :
    bunchkaufman!(Hermitian(A); scratch = FaerScratch(bunchkaufman!, Hermitian(A)))

@testset "Direct factors as inverse-action operators and preconditioners" begin
    rng=MersenneTwister(8)
    n=9
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        kind in (:lu, :full_lu, :qr, :pivoted_qr, :cholesky, :ldlt, :bunchkaufman)

        X=randn(rng, T, n, n)
        hpd=kind in (:cholesky, :ldlt, :bunchkaufman)
        M=hpd ? X*X'+n*I : X+n*I
        A=FaerMatrix(copy(M))
        F=_dfo_factor(kind, A)
        op=FaerLinearOperator(F)
        @test size(op)==(n, n)
        @test eltype(op)===T
        tol=2000eps(real(T))
        b=randn(rng, T, n, 2)
        y=zeros(T, n, 2)
        @test isapprox(mul!(y, op, b), M\b; rtol = tol, atol = tol)
        @test isapprox(Matrix(op'*FaerMatrix(b)), M'\b; rtol = tol, atol = tol)
        # The operator owns a copy: overwriting A does not change it.
        A.=zero(T)
        @test isapprox(mul!(y, op, b), M\b; rtol = tol, atol = tol)
        # Preconditioning with the factor itself (converted automatically).
        G=_dfo_factor(kind, FaerMatrix(copy(M)))
        v=b[:, 1]
        r=hpd ? cg(FaerMatrix(M), v; M = G) : bicgstab(FaerMatrix(M), v; M = G)
        @test isapprox(r.x, M\v; rtol = sqrt(eps(real(T))), atol = sqrt(eps(real(T))))
        close(op)
    end
    R=FaerMatrix(randn(rng, 5, 3))
    @test_throws DimensionMismatch FaerLinearOperator(qr!(R; scratch = FaerScratch(qr!, R)))
end
