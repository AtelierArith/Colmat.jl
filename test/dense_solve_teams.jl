using Test, LinearAlgebra, Colmat
function dst_fixture(T, m, n)
    a = T[0.03sin(2i + j) for i in 1:m, j in 1:n]
    for i in 1:min(m, n)
        a[i, i] += T(3 + i)
    end
    T <: Complex && (a .+= T(0.02im) .* T[cos(i + 2j) for i in 1:m, j in 1:n])
    return a
end
function dst_check(f, a, team, T)
    for (g, ref) in ((f, a), (transpose(f), transpose(a)), (adjoint(f), adjoint(a)))
        b = fill(one(T), size(ref, 1), 2)
        out = zeros(T, size(ref, 2), 2)
        tol = 500eps(real(T)) * max(size(ref)..., 1)
        gold = pinv(Matrix(ref)) * b
        @test ldiv!(out, g, b; team) === out
        @test out ≈ gold atol = tol rtol = tol
        @test ldiv!(out, g, b) === out
        @test out ≈ gold atol = tol rtol = tol
        v = zeros(T, size(ref, 2))
        @test ldiv!(v, g, b[:, 1]; team) === v
        @test v ≈ gold[:, 1] atol = tol rtol = tol
        native = FaerMatrix(zeros(T, size(out)))
        rhs = FaerMatrix(b)
        @test ldiv!(native, g, rhs; team) === native
        @test Matrix(native) ≈ gold atol = tol rtol = tol
        close(native)
        close(rhs)
        b_right = fill(one(T), 2, size(ref, 2))
        out_right = zeros(T, 2, size(ref, 1))
        @test rdiv!(out_right, b_right, g; team) === out_right
        @test out_right ≈ b_right * pinv(Matrix(ref)) atol = tol rtol = tol
        if size(ref, 1) == size(ref, 2)
            aliased = copy(b)
            @test ldiv!(g, aliased; team) === aliased
            @test aliased ≈ gold atol = tol rtol = tol
            right = copy(b_right)
            @test rdiv!(right, g; team) === right
            @test right ≈ b_right * pinv(Matrix(ref)) atol = tol rtol = tol
        end
    end
end
@testset "retained dense solve teams" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), width in (1, 2)
        team = FaerTeam(width)
        for n in (0, 1, 5)
            a = dst_fixture(T, n, n)
            h = a * a' + I
            for (make, ref) in (
                (lu, a),
                (full_piv_lu, a),
                (qr, a),
                (a -> qr(a, ColumnNorm()), a),
                (cholesky, Matrix(h)),
                (ldlt, Matrix(h)),
                (bunchkaufman, Matrix(h)),
            )
                source = FaerMatrix(ref)
                f = make(source)
                close(source)
                dst_check(f, ref, team, T)
                close(f)
            end
        end
        for (m, n, deficient) in (
                (7, 3, false),
                (3, 7, false),
                (7, 3, true),
                (3, 7, true),
                (0, 4, false),
                (4, 0, false),
            ),
            pivot in (false, true)

            a = dst_fixture(T, m, n)
            deficient && (a[:, end] .= a[:, 1])
            f = pivot ? qr(FaerMatrix(a), ColumnNorm()) : qr(FaerMatrix(a))
            dst_check(f, a, team, T)
            @test_throws DimensionMismatch ldiv!(f, zeros(T, m, 1); team)
            close(f)
        end
        close(team)
    end
end
@testset "mixed scalars, views, team errors and writeback" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        make in
        (lu, full_piv_lu, qr, a -> qr(a, ColumnNorm()), cholesky, ldlt, bunchkaufman)

        a = dst_fixture(T, 4, 4)
        a = a * a' + I
        f = make(FaerMatrix(a))
        team = FaerTeam(2)
        for S in (Float32, Float64, ComplexF32, ComplexF64)
            b = fill(S <: Complex ? S(1 + 0.2im) : one(S), 4, 2)
            P = promote_type(T, S)
            out = zeros(P, 4, 2)
            tol = 500eps(real(T)) * 4
            @test ldiv!(out, f, b; team) === out
            @test out ≈ Matrix(a) \ b atol = tol rtol = tol
            backing = fill(P(42), 8, 4)
            viewout = view(backing, 1:2:8, 1:2:4)
            @test ldiv!(viewout, f, b; team) === viewout
            @test viewout ≈ out atol = tol rtol = tol
            @test all(==(P(42)), backing[2:2:8, :])
            v = FaerVector(zeros(P, 4))
            bv = FaerVector(b[:, 1])
            @test ldiv!(v, f, bv; team) === v
            @test Vector(v) ≈ out[:, 1] atol = tol rtol = tol
            close(v)
            close(bv)
        end
        bad = fill(T(42), 4, 2)
        saved = copy(bad)
        @test_throws ArgumentError ldiv!(bad, f, ones(T, 4, 2); team=1)
        @test bad == saved
        close(team)
        @test_throws ArgumentError ldiv!(bad, f, ones(T, 4, 2); team)
        @test bad == saved
        close(f)
    end
    f = lu(FaerMatrix(ComplexF64[3 im; im 4]))
    dest = fill(42.0, 2, 1)
    @test_throws InexactError ldiv!(dest, f, ones(ComplexF64, 2, 1); team=FaerTeam(2))
    @test dest == fill(42.0, 2, 1)
    close(f)
end
@testset "blocked solves, copies, reuse and metadata guards" begin
    team = FaerTeam(2)
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        a = dst_fixture(T, 129, 129)
        h = a * a' + I
        for (make, ref) in (
            (lu, a),
            (full_piv_lu, a),
            (qr, a),
            (a -> qr(a, ColumnNorm()), a),
            (cholesky, Matrix(h)),
            (ldlt, Matrix(h)),
            (bunchkaufman, Matrix(h)),
        )
            f = make(FaerMatrix(ref))
            dst_check(f, ref, team, T)
            g = deepcopy(f)
            close(f)
            b = ones(T, 129, 1)
            result = zeros(T, 129, 1)
            @test fetch(@async ldiv!(result, g, b; team)) === result
            @test ref * result ≈ b rtol = 500eps(real(T)) * 129
            close(g)
        end
    end
    f = lu(FaerMatrix([4.0 1; 2 3]))
    b = ones(2, 2)
    saved = fill(42.0, 2, 2)
    out = copy(saved)
    f.dims = (3, 2)
    @test_throws Colmat.RustCall.RustError ldiv!(out, f, b; team)
    @test out == saved
    f.dims = (2, 2)
    Colmat._native_call(
        Colmat.RustCrate.initialize,
        f,
        FaerMatrix([4.0 1; 2 3]),
        Int32(2),
        true,
        UInt(2),
        UInt(2),
        Int32(1),
        nothing,
    )
    @test_throws Colmat.RustCall.RustError ldiv!(out, f, b; team)
    @test out == saved
    close(f)
    f = lu(FaerMatrix(Float32[4 1; 2 3]))
    Colmat._native_call(
        Colmat.RustCrate.initialize,
        f,
        FaerMatrix(ComplexF32[4 1; 2 3]),
        Int32(0),
        true,
        UInt(2),
        UInt(2),
        Int32(2),
        nothing,
    )
    @test_throws Colmat.RustCall.RustError ldiv!(out, f, b; team)
    @test out == saved
    close(f)
    f = lu(FaerMatrix([1.0 1; 1 1]); check=false)
    @test_throws SingularException ldiv!(out, f, b; team)
    @test out == saved
    close(f)
    f = lu(FaerMatrix([4.0 1; 2 3]))
    outputs = [zeros(2, 2) for _ in 1:4]
    @sync for dst in outputs
        @async ldiv!(dst, f, b; team)
    end
    @test all(x -> x ≈ [4.0 1; 2 3] \ b, outputs)
    close(team)
    @test ldiv!(out, f, b) === out
    @test out ≈ [4.0 1; 2 3] \ b
    close(f)
end
@testset "unsuccessful factors and nonfinite RHS preserve output" begin
    FaerTeam(2) do team
        for T in (Float32, Float64, ComplexF32, ComplexF64)
            f = cholesky(FaerMatrix(T[-1 0; 0 1]); check=false)
            dest = fill(T(42), 2, 2)
            @test_throws SingularException ldiv!(dest, f, ones(T, 2, 2); team)
            @test dest == fill(T(42), 2, 2)
            close(f)
            f = lu(FaerMatrix(T[4 1; 2 3]))
            @test_throws Colmat.RustCall.RustError ldiv!(dest, f, fill(T(NaN), 2, 2); team)
            @test dest == fill(T(42), 2, 2)
            close(f)
            @test_throws ArgumentError ldiv!(dest, f, ones(T, 2, 2); team)
            @test dest == fill(T(42), 2, 2)
        end
    end
end
