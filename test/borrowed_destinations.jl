using Test, LinearAlgebra, Statistics, SparseArrays, Colmat

_dest_bytes(f) = (f(); @allocated f())

@testset "Native writes into borrowed and Julia destinations" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        tol = sqrt(eps(real(T))) * 100
        a = rand(T, 6, 4)
        b = rand(T, 4, 5)
        A, B = FaerMatrix(a), FaerMatrix(b)
        c0 = rand(T, 6, 5)
        alpha, beta = T(2), T(-0.5)
        expected = alpha * a * b + beta * c0
        # Every destination kind, with native, borrowed and Julia operands.
        for (x, y) in
            ((A, B), (faerview(a), B), (A, b), (a, faerview(b)), (faerview(a), faerview(b)))
            c = copy(c0)
            @test mul!(c, x, y, alpha, beta) === c
            @test c ≈ expected rtol = tol
            j = copy(c0)
            bc = faerview(j)
            @test mul!(bc, x, y, alpha, beta) === bc
            @test j ≈ expected rtol = tol
            f = FaerMatrix(c0)
            mul!(f, x, y, alpha, beta)
            @test Matrix(f) ≈ expected rtol = tol
            g = FaerMatrix(c0)
            mul!(view(g, :, :), x, y, alpha, beta)
            @test Matrix(g) ≈ expected rtol = tol
            c = copy(c0)
            mul!(c, x, y)
            @test c ≈ a * b rtol = tol
        end
        # Lazy transposes/adjoints and strided (reversed, padded) destinations.
        c = copy(c0)
        mul!(c, transpose(FaerMatrix(permutedims(a))), FaerMatrix(b')', alpha, beta)
        @test c ≈ expected rtol = tol
        big = zeros(T, 9, 7)
        dest = view(big, 7:-1:2, 1:5)
        mul!(dest, A, B)
        @test big[7:-1:2, 1:5] ≈ a * b rtol = tol
        @test all(iszero, big[8:9, :]) && all(iszero, big[:, 6:7]) && all(iszero, big[1, :])
        # Matrix-vector products into Julia and borrowed vectors.
        v = rand(T, 4)
        y = rand(T, 6)
        y0 = copy(y)
        mul!(y, A, FaerVector(v), alpha, beta)
        @test y ≈ alpha * a * v + beta * y0 rtol = tol
        z = zeros(T, 6)
        mul!(faerview(z), A, v)
        @test z ≈ a * v rtol = tol
        fv = FaerVector(zeros(T, 6))
        mul!(fv, A, faerview(v))
        @test Vector(fv) ≈ a * v rtol = tol

        # Aliased operands are snapshot natively before C is written.
        s = rand(T, 5, 5)
        expected_alias = s' * s
        js = copy(s)
        mul!(js, js', faerview(js))
        @test js ≈ expected_alias rtol = tol
        js = copy(s)
        bs = faerview(js)
        mul!(bs, bs, bs)
        @test js ≈ s * s rtol = tol
        fs = FaerMatrix(s)
        mul!(fs, fs, fs, alpha, beta)
        @test Matrix(fs) ≈ alpha * s * s + beta * s rtol = tol
        fs = FaerMatrix(s)
        mul!(view(fs, :, 1:2), fs, view(fs, :, 4:5))
        expected_partial = copy(s)
        expected_partial[:, 1:2] = s * s[:, 4:5]
        @test Matrix(fs) ≈ expected_partial rtol = tol

        # beta == 0 never reads C; alpha == 0 never reads A or B.
        c = fill(T(NaN), 6, 5)
        mul!(c, A, B, alpha, 0)
        @test c ≈ alpha * a * b rtol = tol
        nanA = faerview(fill(T(NaN), 6, 4))
        c = copy(c0)
        mul!(c, nanA, B, 0, beta)
        @test c ≈ beta * c0 rtol = tol
        mul!(c, nanA, B, 0, 0)
        @test iszero(c)
        # Failed dimension checks leave C untouched.
        c = copy(c0)
        @test_throws DimensionMismatch mul!(c, A, A)
        @test_throws DimensionMismatch mul!(faerview(c), B, A)
        @test c == c0

        # Same-scalar products allocate only small Julia descriptors.
        n = 200
        C = zeros(T, n, n)
        P, Q = FaerMatrix(rand(T, n, n)), FaerMatrix(rand(T, n, n))
        @test _dest_bytes(() -> mul!(C, P, Q)) < 10_000
        @test _dest_bytes(() -> mul!(faerview(C), P, Q, alpha, beta)) < 10_000
        F = FaerMatrix{T}(undef, n, n)
        @test _dest_bytes(() -> mul!(F, P, Q, alpha, beta)) < 10_000
        @test _dest_bytes(() -> mul!(view(F, :, :), P, Q)) < 10_000

        # copyto! from native sources writes Julia memory in place.
        j = zeros(T, n, n)
        copyto!(j, P)
        @test j == Matrix(P)
        @test _dest_bytes(() -> copyto!(j, P)) < 10_000
        bj = faerview(zeros(T, n, n))
        @test copyto!(bj, Q) === bj
        @test parent(bj) == Matrix(Q)
        @test _dest_bytes(() -> copyto!(bj, Q)) < 10_000
        j = zeros(T, 6, 3)
        copyto!(j, view(A, :, 4:-1:2))
        @test j == a[:, 4:-1:2]
        copyto!(view(j, 6:-1:1, :), view(A, :, 1:3))
        @test j == a[6:-1:1, 1:3]
        jv = zeros(T, 4)
        copyto!(jv, FaerVector(v))
        @test jv == v
        # Borrowed sources overlapping the destination are snapshot.
        w = copy(a)
        copyto!(faerview(view(w, :, 1:2)), faerview(view(w, :, 2:3)))
        @test w[:, 1:2] == a[:, 2:3]
        # Base linear-order semantics remain for other shapes and scalars.
        lin = zeros(T, 4, 6)
        copyto!(lin, A)
        @test vec(lin) == vec(a)
        long = zeros(T, 30)
        copyto!(long, A)
        @test long[1:24] == vec(a) && iszero(long[25:end])
        copyto!(faerview(lin), a')
        @test lin == a'
        if T <: Real
            @test_throws InexactError copyto!(zeros(T, 6, 4), FaerMatrix(complex.(a, a)))
        end
        # Closed sources still throw.
        closed = FaerMatrix(a)
        close(closed)
        @test_throws ArgumentError copyto!(zeros(T, 6, 4), closed)

        # Householder reflections apply in place to Julia and borrowed arrays.
        x = rand(T, 6)
        H = householder(x)
        dense = Matrix(H)
        for target in (copy(a), faerview(copy(a)))
            lmul!(H, target)
            @test Array(target) ≈ dense * a rtol = tol
        end
        r = rand(T, 5, 6)
        rr = copy(r)
        rmul!(rr, H)
        @test rr ≈ r * dense rtol = tol
        brr = faerview(copy(r))
        rmul!(brr, H)
        @test parent(brr) ≈ r * dense rtol = tol
        fa = FaerMatrix(a)
        lmul!(H, view(fa, :, 2:3))
        @test Matrix(fa)[:, 2:3] ≈ dense * a[:, 2:3] rtol = tol
        work = FaerScratch(1 << 16)
        team = FaerTeam(2)
        fa = FaerMatrix(a)
        lmul!(H, fa; scratch=work, team)
        @test Matrix(fa) ≈ dense * a rtol = tol
        ja = copy(a)
        lmul!(H, ja; scratch=work)
        @test ja ≈ dense * a rtol = tol
        close(team)
        close(work)
        big = rand(T, n, n)
        Hn = householder(rand(T, n))
        @test _dest_bytes(() -> lmul!(Hn, big)) < sizeof(big) ÷ 4
        @test _dest_bytes(() -> rmul!(big, Hn)) < sizeof(big) ÷ 4

        # Operator results are written straight into Julia memory.
        op = FaerLinearOperator(FaerMatrix(rand(T, 6, 6)))
        ja = copy(a)
        lmul!(op, ja)
        @test ja ≈ Matrix(op * FaerMatrix(a)) rtol = tol
    end

    # Mixed scalar types promote; non-representable results leave C intact.
    a = rand(Float32, 3, 3)
    c = zeros(ComplexF64, 3, 3)
    mul!(c, FaerMatrix(a), FaerMatrix(a))
    @test c ≈ ComplexF64.(a * a) rtol = 1e-5
    c = zeros(Float64, 3, 3)
    @test_throws InexactError mul!(c, FaerMatrix(complex.(a, a)), FaerMatrix(a))
    @test iszero(c)
    # Destinations that repeat elements fall back without writing aliases.
    rep = zeros(4)
    stuck = view(rep, StepRangeLen(1, 0, 3))
    @test_throws ArgumentError faerview(stuck)
    copyto!(stuck, FaerVector([1.0, 2.0, 3.0]))
    @test rep[1] == 3.0
    # Read-only NPY borrows are never written, natively or through fallbacks.
    payload = [1.0 3.0; 2.0 4.0]
    header = "{'descr': '<f8', 'fortran_order': True, 'shape': (2, 2), }"
    pad = 64 - (10 + length(header) + 1) % 64
    bytes = vcat(
        UInt8[0x93],
        Vector{UInt8}("NUMPY"),
        UInt8[0x01, 0x00],
        reinterpret(UInt8, [htol(UInt16(length(header) + pad + 1))]),
        Vector{UInt8}(header * " "^pad * "\n"),
        reinterpret(UInt8, vec(payload)),
    )
    ro = npyborrow(bytes)
    @test ro isa FaerBorrowedMatrix{Float64}
    before = copy(bytes)
    F = FaerMatrix([5.0 6; 7 8])
    @test_throws ArgumentError mul!(ro, F, F)
    @test_throws ArgumentError mul!(ro, F, F, 2, 3)
    @test_throws ArgumentError copyto!(ro, F)
    @test_throws ArgumentError copyto!(ro, [1.0 2; 3 4])
    @test_throws ArgumentError lmul!(householder([1.0, 2.0]), ro)
    @test_throws ArgumentError rmul!(ro, householder([1.0, 2.0]))
    @test bytes == before
    @test ro == payload
    # Diagonal products still read Julia diagonals (owner kind 4).
    d = Diagonal(rand(4))
    m = rand(4, 4)
    @test Matrix(d * FaerMatrix(m)) ≈ d * m

    ambiguities = Test.detect_ambiguities(
        Colmat, Base, LinearAlgebra, SparseArrays, Statistics; recursive=true
    )
    @test isempty(
        filter(a -> a[1].module === Colmat || a[2].module === Colmat, ambiguities)
    )
end
