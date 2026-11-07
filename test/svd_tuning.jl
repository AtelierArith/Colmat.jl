using Colmat, Test, LinearAlgebra

const SVD_TUNING_TYPES=(Float32, Float64, ComplexF32, ComplexF64)
_svd_sample(::Type{T}, m, n) where {T} =
    T.([sin(i+3j) + (T<:Complex ? im*cos(2i-j) : 0) for i = 1:m, j = 1:n])
function _svd_res(a, F)
    k=length(F.S)
    return norm(F.U[:, 1:k]*Diagonal(F.S)*F.Vt[1:k, :]-a)/max(norm(a), 1)
end
const ALGS=(nothing, LinearAlgebra.DivideAndConquer(), LinearAlgebra.QRIteration())

@testset "SVD alg keyword" begin
    team=FaerTeam(2)
    for T in SVD_TUNING_TYPES, (m, n) in ((0, 0), (6, 4), (40, 150), (150, 140))
        a=_svd_sample(T, m, n)
        A=FaerMatrix(a)
        tol=1000eps(real(T))
        ref=svdvals(a)
        for alg in ALGS, kw in ((;), (; team)), full in (false, true)
            F=svd(A; alg, full, kw...)
            @test _svd_res(a, F) <= tol
            @test F.S ≈ ref
            @test svdvals(A; alg, kw...) ≈ ref
        end
        if m*n > 0
            B=copy(A)
            F=svd!(B; alg = LinearAlgebra.QRIteration())
            @test F.S ≈ ref
            C=copy(A)
            @test svdvals!(C; alg = LinearAlgebra.QRIteration()) ≈ ref
            foreach(close, (B, C))
        end
        @test Matrix(A) == a
        @test_throws ArgumentError svd(A; alg = :qr)
        @test_throws ArgumentError svdvals(A; alg = 1)
        close(A)
    end
    close(team)
end

@testset "with_svd_tuning reaches the driver and is scoped" begin
    A=FaerMatrix(_svd_sample(Float64, 300, 200))
    base=scratch_requirements(svd, A)
    qr=with_svd_tuning(()->scratch_requirements(svd, A); recursion_threshold = typemax(Int))
    @test qr.bytes != base.bytes
    # Scratch reserved inside the scope serves the tuned call.
    with_svd_tuning(;
        recursion_threshold = 16,
        qr_ratio_threshold = 1.5,
        bidiag_par_threshold = FaerParThreshold(:never),
    ) do
        work=FaerScratch(svd, A)
        F=svd(A; scratch = work)
        @test _svd_res(Matrix(A), F) < 1e-12
        close(work)
        # alg composes with the enclosing settings and restores them.
        svd(A; alg = LinearAlgebra.QRIteration())
        v, r, f=Colmat._svd_state()
        @test (Int(v[1]), r[1], f) == (16, 1.5, UInt8[1, 1, 1])
    end
    v, r, f=Colmat._svd_state()
    @test f == zeros(UInt8, 3)
    @test Colmat._svd_tuning_depth[] == 0
    @test_throws ErrorException with_svd_tuning(()->error("boom"); recursion_threshold = 8)
    @test Colmat._svd_state()[3] == zeros(UInt8, 3)
    for bad in (
        (; recursion_threshold = 1),
        (; qr_ratio_threshold = 0.5),
        (; qr_ratio_threshold = Inf),
        (; bidiag_par_threshold = 3),
    )
        @test_throws ArgumentError with_svd_tuning(()->nothing; bad...)
    end
    close(A)
end
