using Colmat, Test, LinearAlgebra, Random

const GC_TYPES = (Float32, Float64, ComplexF32, ComplexF64)

@testset "Generalized eigenvalue quotient conventions" begin
    for T in GC_TYPES
        A = T[1 2; 0 3]
        B = T[1 0; 0 0]  # one finite eigenvalue (1) and one infinite one
        v = eigvals(FaerMatrix(A), FaerMatrix(B))
        @test v[1] ≈ 1
        if T <: Real
            # Real pencils: the real infinite eigenvalue is Inf+0im (LAPACK: real Inf).
            @test v[2] === complex(T(Inf))
            @test real.(v) == eigvals(A, B)
        else
            # Complex pencils keep Julia's complex quotient, as LinearAlgebra does.
            @test isnan(real(v[2])) && isnan(imag(v[2]))
            @test isequal(v, eigvals(A, B))
        end
        h = eigvals(FaerMatrix(A), FaerMatrix(B); homogeneous=true)
        @test iszero(h.β[2]) && !iszero(h.α[2])
        E = eigen(FaerMatrix(A), FaerMatrix(B))
        @test isequal(E.values, v)
        # Eigenvectors have unit 2-norm (LAPACK ggev scales max |re|+|im| to 1).
        @test all(c -> norm(c) ≈ 1, eachcol(E.vectors))
    end
    # A real pencil with a complex pair keeps both nonzero imaginary parts.
    v = eigvals(FaerMatrix([0.0 -1; 1 0]), FaerMatrix([2.0 0; 0 1]))
    @test v ≈ [-im / sqrt(2), im / sqrt(2)]
end

@testset "Graded and larger generalized pencils" begin
    rng = MersenneTwister(11)
    for T in (Float64, ComplexF64), n in (40, 120)
        A0 = randn(rng, T, n, n)
        B0 = randn(rng, T, n, n) + n * I
        ref = eigvals(A0, B0)
        # Row and column grading leave the eigenvalues unchanged.
        for g in (1e-3, 1e3)
            D1 = Diagonal(T.(g .^ range(-1, 1; length=n)))
            D2 = Diagonal(T.(g .^ range(1, -1; length=n)))
            A = D1 * A0 * D2
            B = D1 * B0 * D2
            v = eigvals(FaerMatrix(A), FaerMatrix(B))
            @test sort(abs.(v)) ≈ sort(abs.(ref)) rtol = 1e-6
        end
        E = eigen(FaerMatrix(A0), FaerMatrix(B0))
        V = E.vectors
        resid =
            norm(A0 * V - B0 * V * Diagonal(E.values)) /
            (norm(A0) + norm(B0) * maximum(abs, E.values))
        @test resid < 1e3 * n * eps(real(T))
    end
end

# Tuning, wider extreme-scale and allocation regressions (follow-up to #109).
function _homogeneous_error(A, B, V, α, β; side=:right)
    nA, nB = opnorm(A, 1), opnorm(B, 1)
    return maximum(axes(V, 2); init=zero(real(eltype(A)))) do j
        v = view(V, :, j)
        r = if side === :right
            β[j] * (A * v) - α[j] * (B * v)
        else
            conj(β[j]) * (A' * v) - conj(α[j]) * (B' * v)
        end
        return norm(r, 1) / ((abs(β[j]) * nA + abs(α[j]) * nB) * norm(v, 1))
    end
end
function _pencil(T, n, seed)
    rng = Xoshiro(seed)
    A = randn(rng, T, n, n)
    B = randn(rng, T, n, n) + 2n * I
    return A, B
end
@testset "Generalized EVD follows with_schur_tuning" begin
    # n stays below 75: from there the pinned faer QZ can take seconds on such
    # pencils (docs/reproducers/faer-gevd-driver.md).
    for T in (Float32, Float64, ComplexF32, ComplexF64), n in (0, 5, 40, 74)
        A, B = _pencil(T, n, 3n + 1)
        a, b = FaerMatrix(A), FaerMatrix(B)
        base = eigen(a, b; homogeneous=true)
        same = with_schur_tuning(() -> eigen(a, b; homogeneous=true))
        @test isequal(same, base)
        default_bytes = scratch_requirements(eigen, a, b).bytes
        for width in (1, 2), bs in (1, 64), side in (:right, :both)
            FaerTeam(width) do team
                with_schur_tuning(; generalized_hessenberg_block_size=bs) do
                    req = scratch_requirements(eigen, a, b; team, side)
                    n > bs > 32 && @test req.bytes > default_bytes
                    s = FaerScratch(eigen, a, b; team, side)
                    @test scratch_capacity(s) == req
                    H = eigen(a, b; homogeneous=true, side, scratch=s, team)
                    tol = 10max(n, 8) * eps(real(T))
                    if side === :both
                        @test _homogeneous_error(A, B, H.right, H.α, H.β) < tol
                        @test _homogeneous_error(A, B, H.left, H.α, H.β; side=:left) < tol
                    else
                        @test _homogeneous_error(A, B, H.vectors, H.α, H.β) < tol
                    end
                    @test eigvals(a, b; scratch=s, team) ≈ eigvals(a, b) rtol = sqrt(
                        eps(real(T))
                    )
                    close(s)
                end
            end
        end
        if n == 74
            # The tuned Hessenberg kernel is actually used by the driver.
            tuned = with_schur_tuning(
                () -> eigen(a, b; homogeneous=true); generalized_hessenberg_block_size=1
            )
            @test !isequal(tuned.vectors, base.vectors)
        end
    end
    # QZ controls reach the driver with exactly queried scratch (faer#252 shape).
    for T in (Float64, ComplexF64)
        A, B = _pencil(T, 96, 11)
        a, b = FaerMatrix(A), FaerMatrix(B)
        with_schur_tuning(; deflation_window=32, shift_count=6) do
            s = FaerScratch(eigen, a, b)
            H = eigen(a, b; homogeneous=true, scratch=s)
            @test _homogeneous_error(A, B, H.vectors, H.α, H.β) < 960eps(real(T))
            close(s)
        end
    end
end
@testset "Wider extreme-scale generalized pencils" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        R = real(T)
        tol = 500eps(R)
        # Pencils whose A (or B) norm lies far below/above faer's safe range;
        # B stays nonsingular, so every β must be nonzero. The eigenvalues
        # (scaled by sa/sb) stay representable.
        tiny, huge = T <: Union{Float32,ComplexF32} ? (R(1e-30), R(1e30)) : (1e-300, 1e300)
        for n in (8, 32),
            (sa, sb) in ((tiny, one(R)), (huge, one(R)), (one(R), tiny), (tiny, tiny))

            A0, B0 = _pencil(T, n, n + 17)
            A, B = A0 * sa, B0 * sb
            a, b = FaerMatrix(A), FaerMatrix(B)
            reference = eigvals(FaerMatrix(A0), FaerMatrix(B0); homogeneous=true)
            H = eigen(a, b; homogeneous=true)
            @test all(isfinite, H.α) && all(isfinite, H.β)
            @test all(!iszero, H.β)
            @test all(isfinite, H.vectors)
            λ = sort(H.α ./ H.β .* (sb / sa); by=x -> (real(x), imag(x)))
            λ0 = sort(reference.α ./ reference.β; by=x -> (real(x), imag(x)))
            @test λ ≈ λ0 rtol = sqrt(eps(R))
            @test _homogeneous_error(A0, B0, H.vectors, H.α * (sb / sa), H.β) < tol
        end
        # Diagonal grading over many orders of magnitude: eigenvalues are those
        # of the ungraded pencil, and QZ stays normwise backward stable.
        decades = T <: Union{Float32,ComplexF32} ? 8 : 24
        n = 16
        A0, B0 = _pencil(T, n, 5)
        d1 = R(10) .^ range(-decades / 2, decades / 2; length=n)
        d2 = reverse(d1)
        A, B = Diagonal(d1) * A0 * Diagonal(d2), Diagonal(d1) * B0 * Diagonal(d2)
        FaerTeam(2) do team
            for side in (:right, :left)
                H = eigen(FaerMatrix(A), FaerMatrix(B); homogeneous=true, side, team)
                @test all(isfinite, H.α) && all(isfinite, H.β) && all(isfinite, H.vectors)
                @test _homogeneous_error(A, B, H.vectors, H.α, H.β; side) < tol
            end
        end
        # Nearly singular B: one eigenvalue of size ~1/δ, still finite pairs.
        n = 12
        rng = Xoshiro(23)
        U = Matrix(qr(randn(rng, T, n, n)).Q)
        V = Matrix(qr(randn(rng, T, n, n)).Q)
        A = randn(rng, T, n, n)
        for δ in (sqrt(eps(R)), eps(R), eps(R)^2, zero(R))
            B = U * Diagonal([ones(R, n - 1); δ]) * V'
            H = eigen(FaerMatrix(A), FaerMatrix(B); side=:both, homogeneous=true)
            @test all(isfinite, H.α) && all(isfinite, H.β)
            @test _homogeneous_error(A, B, H.right, H.α, H.β) < tol
            # ComplexF32 left vectors are computed in ComplexF64 around
            # tensor4all/faer#324 (#239); the outputs keep the pencil's type.
            @test _homogeneous_error(A, B, H.left, H.α, H.β; side=:left) < tol
            @test eltype(H.left) === Complex{R} && eltype(H.α) === Complex{R}
            @test minimum(abs, H.β ./ maximum(abs, H.β)) < 100eps(R) + δ * 100
        end
    end
    # One larger order for Float64.
    A, B = _pencil(Float64, 160, 2)
    H = eigen(FaerMatrix(A), FaerMatrix(B); homogeneous=true)
    @test _homogeneous_error(A, B, H.vectors, H.α, H.β) < 500eps()
    λ = eigvals(A, B)
    @test all(x -> minimum(abs, λ .- x) <= 1e-8abs(x), H.α ./ H.β)
end
@testset "Generalized EVD allocations: owners versus workspace" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), n in (16, 64)
        A, B = _pencil(T, n, n)
        a, b = FaerMatrix(A), FaerMatrix(B)
        C = Complex{real(T)}
        for (vectors, side) in
            ((false, :right), (true, :right), (true, :left), (true, :both))
            f = vectors ? eigen : eigvals
            s = FaerScratch(f, a, b; side)
            run() =
                if vectors
                    eigen(a, b; homogeneous=true, side, scratch=s)
                else
                    eigvals(a, b; homogeneous=true, scratch=s)
                end
            run()
            bytes = @allocated run()
            nvec = vectors ? (side === :both ? 2 : 1) : 0
            # α, β and the eigenvector matrices are the returned owners; the
            # A/B copies and packed real output live in the native scratch.
            # The rest is fixed per-call bookkeeping (about 12 KB), whatever n.
            owners = n * sizeof(C) + n * sizeof(T) + nvec * n * n * sizeof(C)
            if T === ComplexF32 && side !== :right
                # #239 workaround: ComplexF64 copies, outputs and rounding.
                @test owners <= bytes <= 4owners + 32768
            else
                @test owners <= bytes <= owners + 16384
            end
            close(s)
        end
    end
end
