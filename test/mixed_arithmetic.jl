using Test, LinearAlgebra, SparseArrays, Statistics, Colmat

_mixed_bytes(f) = (f(); @allocated f())

# Follow-up to #96/#197: lazy transposes, rows, diagonals and scalar multiples
# with Julia operands are computed natively without Julia staging.
@testset "Mixed arithmetic" begin
    @testset "Mixed native/Julia dense +, - and *" begin
        for T in (Float32, Float64, ComplexF32, ComplexF64)
            tol = sqrt(eps(real(T))) * 100
            n = 6
            a, b = rand(T, n, n), rand(T, n, n)
            A = FaerMatrix(a)
            natives = (
                "A" => (A, a),
                "view" => (view(A, 1:n, 1:n), a),
                "A'" => (A', a'),
                "transpose(A)" => (transpose(A), transpose(a)),
                "view'" => (view(A, :, :)', a'),
            )
            julias = (
                "J" => (b, b),
                "view(J)" => (view(copy(b), 1:n, 1:n), b),
                "J'" => (b', b'),
                "faerview'" => (faerview(copy(b))', b'),
            )
            for (_, (x, xm)) in natives, (_, (y, ym)) in julias
                for (f, op) in ((+, "+"), (-, "-"))
                    r = f(x, y)
                    @test r isa FaerMatrix{T}
                    @test Matrix(r) ≈ f(xm, ym) rtol = tol
                    s = f(y, x)
                    @test s isa FaerMatrix{T}
                    @test Matrix(s) ≈ f(ym, xm) rtol = tol
                end
                r = x * y
                @test r isa FaerMatrix{T}
                @test Matrix(r) ≈ xm * ym rtol = tol
                s = y * x
                @test s isa FaerMatrix{T}
                @test Matrix(s) ≈ ym * xm rtol = tol
            end
            # Native transposes with native operands.
            for (_, (x, xm)) in natives, (_, (y, ym)) in natives
                @test x + y isa FaerMatrix{T}
                @test Matrix(x + y) ≈ xm + ym rtol = tol
                @test Matrix(x - y) ≈ xm - ym rtol = tol
                @test Matrix(x * y) ≈ xm * ym rtol = tol
            end
            # Matrix-vector products with Julia and native vectors.
            v, w = rand(T, n), rand(T, n)
            V = FaerVector(v)
            for (_, (x, xm)) in (natives..., "J" => (b, b), "J'" => (b', b'))
                for y in (w, view(copy(w), 1:n), V, view(V, 1:n))
                    x isa Union{StridedMatrix,LinearAlgebra.Adjoint{<:Any,<:Matrix}} &&
                        y isa StridedVector &&
                        continue
                    r = x * y
                    @test r isa FaerVector{T}
                    @test Vector(r) ≈ xm * Vector(y) rtol = tol
                end
            end
            # Row vectors and adjoint vectors times matrices stay native.
            row = FaerRow(v)
            for (_, (y, ym)) in (natives..., julias...)
                r = row * y
                @test r isa FaerRow{T}
                @test vec(Matrix(r)) ≈ vec(transpose(v) * ym) rtol = tol
                s = V' * y
                @test parent(s) isa FaerVector{T}
                @test s isa LinearAlgebra.Adjoint
                @test Vector(parent(s)) ≈ vec((v' * ym)') rtol = tol
                t = transpose(V) * y
                @test parent(t) isa FaerVector{T}
                @test t isa LinearAlgebra.Transpose
                @test Vector(parent(t)) ≈ vec(transpose(transpose(v) * ym)) rtol = tol
            end
            # Mixed scalar types promote, with native results.
            c = rand(ComplexF64, n, n)
            r = A + c
            @test r isa FaerMatrix{promote_type(T, ComplexF64)}
            @test Matrix(r) ≈ a + c rtol = tol
            @test Matrix(A' * c) ≈ a' * c rtol = tol
            @test Matrix(c - A') ≈ c - a' rtol = tol
            @test_throws DimensionMismatch A + rand(T, n, n + 1)
            @test_throws DimensionMismatch A * rand(T, n + 1, 2)
            @test_throws DimensionMismatch row * rand(T, n + 1, 2)
        end
        # Julia-only and borrowed-only combinations keep Julia's results.
        j = rand(3, 3)
        @test faerview(j) + j isa Matrix
        @test faerview(j) * j isa Matrix
        @test j * j isa Matrix
        # Non-native scalars use Julia's generic paths.
        A = FaerMatrix(rand(3, 3))
        @test A + ones(Int, 3, 3) ≈ Matrix(A) .+ 1
        @test Matrix(A) * [1, 2, 3] ≈ A * [1, 2, 3]
    end

    @testset "Mixed products into native vector and row destinations" begin
        for T in (Float32, Float64, ComplexF32, ComplexF64)
            tol = sqrt(eps(real(T))) * 100
            n = 5
            a, b = rand(T, n, n), rand(T, n, n)
            v, w = rand(T, n), rand(T, n)
            A = FaerMatrix(a)
            V = FaerVector(v)
            alpha, beta = T(2), T(-0.5)
            y0 = rand(T, n)
            for (x, xm) in
                ((A, a), (A', a'), (transpose(A), transpose(a)), (b, b), (b', b'))
                for z in (w, view(copy(w), :), V, view(V, :), faerview(copy(w)))
                    x isa Union{StridedMatrix,LinearAlgebra.Adjoint{<:Any,<:Matrix}} &&
                        z isa StridedVector &&
                        continue
                    y = FaerVector(y0)
                    @test mul!(y, x, z, alpha, beta) === y
                    @test Vector(y) ≈ alpha * xm * Vector(z) + beta * y0 rtol = tol
                    mul!(y, x, z)
                    @test Vector(y) ≈ xm * Vector(z) rtol = tol
                end
            end
            # Row destinations: r = alpha*x*B + beta*r.
            for (x, xv) in
                ((FaerRow(v), transpose(v)), (V', v'), (transpose(V), transpose(v)))
                for (y, ym) in ((A, a), (A', a'), (b, b), (b', b'), (view(A, :, :), a))
                    r = FaerRow(y0)
                    @test mul!(r, x, y, alpha, beta) === r
                    @test vec(Matrix(r)) ≈ vec(alpha * xv * ym + beta * transpose(y0)) rtol =
                        tol
                end
            end
            # Aliasing destinations are snapshot first.
            r = FaerRow(v)
            mul!(r, r, A)
            @test vec(Matrix(r)) ≈ vec(transpose(v) * a) rtol = tol
            y = FaerVector(v)
            mul!(y, A', y)
            @test Vector(y) ≈ a' * v rtol = tol
            # Mixed scalar types into narrower destinations convert completely.
            y = FaerVector(y0)
            mul!(y, A, rand(n))
            @test y isa FaerVector{T}
        end
        # A closed destination is rejected before anything runs.
        y = FaerVector(zeros(3))
        close(y)
        @test_throws Exception mul!(y, FaerMatrix(rand(3, 3)), rand(3))
        @test_throws DimensionMismatch mul!(
            FaerVector(zeros(3)), FaerMatrix(rand(3, 3))', rand(4)
        )
        @test_throws DimensionMismatch mul!(FaerRow(zeros(3)), FaerRow(rand(3)), rand(4, 3))
        # Caller teams are lent to vector and row destinations too.
        team = FaerTeam(2)
        a = rand(6, 6)
        A = FaerMatrix(a)
        v = rand(6)
        y = FaerVector(zeros(6))
        @test mul!(y, A, v; team) === y
        @test Vector(y) ≈ a * v
        mul!(y, a, FaerVector(v), 2.0, 0.0; team)
        @test Vector(y) ≈ 2 * a * v
        r = FaerRow(zeros(6))
        @test mul!(r, FaerRow(v), A; team) === r
        @test vec(Matrix(r)) ≈ vec(transpose(v) * a)
        mul!(r, FaerVector(v)', a, 1.0, 0.0; team)
        @test vec(Matrix(r)) ≈ vec(v' * a)
        close(team)
    end

    @testset "Native scalar promotion and scalar multiples of views" begin
        types = (Float32, Float64, ComplexF32, ComplexF64)
        for S in types, T in types
            P = promote_type(S, T)
            a, b = rand(S, 4, 4), rand(T, 4, 4)
            A, B = FaerMatrix(a), FaerMatrix(b)
            tol = sqrt(eps(real(P))) * 100
            for (x, xm) in ((A, a), (A', a'), (view(A, 1:4, 1:4), a)),
                (y, ym) in ((B, b), (b, b), (B', b'), (b', b'))

                @test x + y isa FaerMatrix{P}
                @test Matrix(x + y) ≈ xm + ym rtol = tol
                @test Matrix(y - x) ≈ ym - xm rtol = tol
                @test x * y isa FaerMatrix{P}
                @test Matrix(x * y) ≈ xm * ym rtol = tol
            end
            v = rand(T, 4)
            @test A * v isa FaerVector{P}
            @test Vector(A * v) ≈ a * v rtol = tol
            @test Vector(parent(FaerVector(v)' * A)) ≈ vec((v' * a)') rtol = tol
            # Scalar multiples of views and lazy transposes stay native.
            x = T(2)
            R = promote_type(typeof(x), S)
            for (y, ym) in
                ((view(A, :, 2:3), a[:, 2:3]), (A', a'), (transpose(A), transpose(a)))
                @test x * y isa FaerMatrix{R}
                @test Matrix(x * y) ≈ x * ym rtol = tol
                @test Matrix(y * x) ≈ ym * x rtol = tol
                @test Matrix(y / x) ≈ ym / x rtol = tol
            end
            @test Matrix(-view(A, 2:3, :)) ≈ -a[2:3, :]
        end
        # Real factors keep Julia's componentwise Real*Complex semantics.
        c = FaerMatrix(ComplexF64[1 + 2im 3; 4im 5])
        @test isequal(Matrix(Inf * c'), Inf * Matrix(c)')
        # Non-native promotions keep Julia's exact generic results.
        B = FaerMatrix(rand(2, 2))
        @test big(2.0) * view(B, :, :) isa Matrix{BigFloat}
        @test big(2.0) * B' isa Matrix{BigFloat}
        @test view(B, :, :) / big(3.0) isa Matrix{BigFloat}
        @test B * Complex{BigFloat}[1, 2] isa Vector{Complex{BigFloat}}
        @test B * big.([1.0, 2.0]) ≈ Matrix(B) * [1.0, 2.0]
    end

    @testset "Mixed operands do not stage through Julia" begin
        n = 128
        a, b = rand(n, n), rand(n, n)
        A = FaerMatrix(a)
        v = rand(n)
        V = FaerVector(v)
        row = FaerRow(v)
        y = FaerVector(zeros(n))
        r = FaerRow(zeros(n))
        payload = sizeof(a)
        c = rand(ComplexF64, n, n)
        C32 = FaerMatrix(rand(ComplexF32, n, n))
        for f in (
            () -> A + b,
            () -> b - A,
            () -> A' + b,
            () -> A + A',
            () -> A' - view(A, :, :),
            () -> A * b,
            () -> b * A',
            () -> A * v,
            () -> A' * V,
            () -> row * A,
            () -> row * b,
            () -> V' * b,
            () -> V' * A,
            () -> mul!(y, A, v),
            () -> mul!(y, A', V),
            () -> mul!(y, b, V),
            () -> mul!(r, row, A),
            () -> mul!(r, V', b),
            () -> 2 * view(A, :, :),
        )
            @test _mixed_bytes(f) < payload ÷ 4
        end
        # Promotions are converted by Rust: no Julia copy of the widened operand.
        for f in (() -> A + c, () -> A' * C32, () -> 2im * A, () -> 2im * A')
            @test _mixed_bytes(f) < payload ÷ 2
        end
        ambiguities = Test.detect_ambiguities(
            Colmat, Base, LinearAlgebra, SparseArrays, Statistics; recursive=true
        )
        @test isempty(
            filter(a -> a[1].module === Colmat || a[2].module === Colmat, ambiguities)
        )
    end

    @testset "Native diagonals and rows with Julia operands" begin
        for T in (Float32, Float64, ComplexF32, ComplexF64)
            tol = sqrt(eps(real(T))) * 100
            n = 5
            d = rand(T, n) .+ 1
            D, Dj = FaerDiagonal(d), Diagonal(d)
            for m in (rand(T, n, n), rand(n, n), view(rand(T, n, n + 1), :, 1:n))
                for x in (m, transpose(m))
                    xm = Matrix(x)
                    P = promote_type(T, eltype(m))
                    for (r, e) in (
                        (D * x, Dj * xm),
                        (x * D, xm * Dj),
                        (D \ x, Dj \ xm),
                        (x / D, xm / Dj),
                        (D + x, Dj + xm),
                        (x + D, xm + Dj),
                        (x - D, xm - Dj),
                        (D - x, Dj - xm),
                    )
                        @test r isa FaerMatrix{P}
                        @test Matrix(r) ≈ e rtol = tol
                    end
                end
            end
            v = rand(T, n)
            @test D * v isa FaerVector{T}
            @test Vector(D * v) ≈ Dj * v rtol = tol
            @test Vector(D \ v) ≈ Dj \ v rtol = tol
            @test_throws DimensionMismatch D * rand(T, n + 1)
            @test_throws DimensionMismatch D + rand(T, n, n + 1)
            # A zero diagonal entry throws before anything is returned.
            @test_throws SingularException FaerDiagonal(zeros(T, n)) \ rand(T, n, n)
            row = FaerRow(v)
            w = rand(T, n)
            for y in (w, FaerVector(w), view(FaerVector(w), :), faerview(copy(w)))
                r = row * y
                @test r isa FaerVector{T}
                @test length(r) == 1
                @test r[1] ≈ transpose(v) * w rtol = tol
            end
        end
        # Julia Diagonals with Julia arrays keep Julia's results.
        @test Diagonal(rand(3)) * rand(3, 3) isa Matrix
        m = rand(128, 128)
        D = FaerDiagonal(rand(128) .+ 1)
        for f in (
            () -> D * m,
            () -> m * D,
            () -> D \ m,
            () -> m / D,
            () -> D + m,
            () -> m - D,
            () -> D * m[:, 1],
        )
            @test _mixed_bytes(f) < sizeof(m) ÷ 4
        end
    end

    @testset "Lazily conjugated native operands" begin
        a = rand(ComplexF64, 4, 4)
        A = FaerMatrix(a)
        v = rand(ComplexF64, 4)
        V = FaerVector(v)
        @test parent(transpose(V) * A') isa FaerVector{ComplexF64}
        @test Vector(parent(transpose(V) * A')) ≈ vec(transpose(v) * a')
        @test Matrix(transpose(A') * A) ≈ transpose(a') * a
        @test Matrix(a * adjoint(transpose(A))) ≈ a * adjoint(transpose(a))
        @test Vector(transpose(A') * v) ≈ transpose(a') * v
        b = rand(ComplexF64, 128, 128)
        B = FaerMatrix(b)
        W = FaerVector(rand(ComplexF64, 128))
        f = () -> transpose(W) * B'
        @test _mixed_bytes(f) < sizeof(b) ÷ 4
    end
end
