using Test, LinearAlgebra, Random, Colmat

qzr_matrices(f) = (f.S, f.T, f.Q, f.Z)
qzr_snapshot(f) = (map(Matrix, qzr_matrices(f)), copy(f.α), copy(f.β))
function qzr_factor(T, s, t; q=Matrix{T}(I, size(s)...), z=Matrix{T}(I, size(s)...))
    n = size(s, 1)
    R = typeof(real(zero(T)))
    return GeneralizedSchur(
        FaerMatrix(T.(s)),
        FaerMatrix(T.(t)),
        zeros(Complex{R}, n),
        zeros(T, n),
        FaerMatrix(T.(q)),
        FaerMatrix(T.(z)),
    )
end
function qzr_properties(g, a, b, tol)
    n = size(g, 1)
    @test Matrix(g.Q * g.S * g.Z') ≈ a rtol = tol atol = tol
    @test Matrix(g.Q * g.T * g.Z') ≈ b rtol = tol atol = tol
    @test Matrix(g.Q)' * Matrix(g.Q) ≈ Matrix{eltype(g)}(I, n, n) rtol = tol atol = tol
    @test Matrix(g.Z)' * Matrix(g.Z) ≈ Matrix{eltype(g)}(I, n, n) rtol = tol atol = tol
    @test Matrix(g.S) == triu(Matrix(g.S), eltype(g) <: Real ? -1 : 0)
    @test Matrix(g.T) == triu(Matrix(g.T))
    @test all(isfinite, g.α) && all(isfinite, g.β)
end

@testset "Native generalized Schur ordering" begin
    rng = MersenneTwister(251)
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        tol = typeof(real(zero(T))) === Float32 ? 3e-3 : 3e-10
        for n in (0, 1, 2, 3, 7, 12)
            a = randn(rng, T, n, n)
            b = randn(rng, T, n, n) + T(4) * I
            f = schur(FaerMatrix(a), FaerMatrix(b))
            for mask in
                (falses(n), trues(n), BitVector(isodd.(1:n)), Vector{Bool}(iseven.(1:n)))
                selected = copy(mask)
                if T <: Real
                    s = Matrix(f.S)
                    for i in 1:(n - 1)
                        if !iszero(s[i + 1, i])
                            selected[i] = selected[i + 1] = selected[i] || selected[i + 1]
                        end
                    end
                end
                expected = vcat(f.values[selected], f.values[.!selected])
                saved = qzr_snapshot(f)
                g = ordschur(f, mask)
                qzr_properties(g, a, b, tol)
                @test g.values ≈ expected rtol = tol atol = tol
                @test qzr_snapshot(f) == saved
                @test all(qzr_matrices(g)[i] !== qzr_matrices(f)[i] for i in 1:4)
                h = copy(f)
                owners = qzr_matrices(h)
                views = map(m -> view(m, :, :), owners)
                @test ordschur!(h, mask) === h
                @test all(qzr_matrices(h)[i] === owners[i] for i in 1:4)
                @test map(Matrix, views) == map(Matrix, qzr_matrices(h))
                qzr_properties(h, a, b, tol)
                @test h.values ≈ expected rtol = tol atol = tol
                close(g)
                close(h)
            end
            close(f)
        end
    end
end

@testset "QZ conjugate masks, infinities and saved metadata" begin
    rng = MersenneTwister(522)
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        tol = typeof(real(zero(T))) === Float32 ? 3e-3 : 3e-10
        if T <: Real
            s = T[5 0.1 0.2 0; 0 0.2 -1 0.03; 0 1 0.2 0.05; 0 0 0 2]
            t = Matrix(Diagonal(T[0, 1, 1, 1]))
            mask = Bool[false, true, false, true]
            expected = Complex{T}[0.2 + im, 0.2 - im, 2]
        else
            s = T[5 0.1 0.2 0; 0 1 + 0.5im 0.2 0.03; 0 0 2 - 0.3im 0.05; 0 0 0 3 + 0.7im]
            t = Matrix(Diagonal(T[0, 1, 1, 1]))
            mask = Bool[false, true, false, true]
            expected = T[1 + 0.5im, 3 + 0.7im, 2 - 0.3im]
        end
        q = Matrix(qr(randn(rng, T, 4, 4)).Q)
        z = Matrix(qr(randn(rng, T, 4, 4)).Q)
        f = qzr_factor(T, s, t; q, z)
        a = q * s * z'
        b = q * t * z'
        saved = qzr_snapshot(f)
        g = ordschur(f, mask)
        qzr_properties(g, a, b, tol)
        @test g.values[T <: Real ? collect(1:3) : [1, 2, 4]] ≈ expected rtol = tol atol =
            tol
        infinite = T <: Real ? 4 : 3
        @test iszero(g.β[infinite]) && !iszero(g.α[infinite])
        @test qzr_snapshot(f) == saved
        # Recompute from current saved factors, even after both public metadata
        # vectors have been overwritten and the source factors have changed.
        f.S[4, 4] = T(7)
        fill!(f.α, 999)
        fill!(f.β, 999)
        g = ordschur(f, falses(4))
        @test g.values[4] ≈ 7
        @test all(==(999), f.α) && all(==(999), f.β)
        h = deepcopy(f)
        close(f)
        @test ordschur!(h, falses(4)) === h
        @test h.values[4] ≈ 7
        @test_throws ArgumentError ordschur(f, mask)
        @test_throws ArgumentError ordschur!(f, mask)
    end
end

@testset "QZ ordering validation and atomic updates" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        s = Matrix(Diagonal(T[1, 2, 3]))
        t = Matrix{T}(I, 3, 3)
        f = qzr_factor(T, s, t)
        saved = qzr_snapshot(f)
        for operation in (ordschur, ordschur!)
            @test_throws DimensionMismatch operation(f, Bool[true, false])
            @test_throws MethodError operation(f, [1, 0, 1])
            @test qzr_snapshot(f) == saved
        end
        for i in 1:4, j in (i + 1):4
            matrices = collect(qzr_matrices(f))
            matrices[j] = matrices[i]
            alias = GeneralizedSchur(
                matrices[1], matrices[2], copy(f.α), copy(f.β), matrices[3], matrices[4]
            )
            before = qzr_snapshot(alias)
            @test_throws ArgumentError ordschur!(alias, trues(3))
            @test qzr_snapshot(alias) == before
        end
        for shared in (false, true)
            data = ones(T, 4)
            alpha = shared ? view(data, 1:3) : ones(T, 3)
            beta = shared ? view(data, 2:4) : alpha
            alias = GeneralizedSchur(f.S, f.T, alpha, beta, f.Q, f.Z)
            @test_throws ArgumentError ordschur!(alias, trues(3))
            @test qzr_snapshot(f) == saved
        end
        malformed = GeneralizedSchur(f.S, f.T, copy(f.α[1:2]), copy(f.β), f.Q, f.Z)
        @test_throws DimensionMismatch ordschur!(malformed, trues(3))
        readonly = GeneralizedSchur(f.S, f.T, 1:3, copy(f.β), f.Q, f.Z)
        @test_throws ArgumentError ordschur!(readonly, trues(3))
        @test ordschur(readonly, falses(3)).values ≈ [1, 2, 3]
        # A real-valued alpha destination cannot hold complex eigenvalues.
        # Conversion must fail before any of the four factors is overwritten.
        s2 = T <: Real ? T[0 -1; 1 0] : T[1 + im 0.1; 0 2 - im]
        f2 = qzr_factor(T, s2, Matrix{T}(I, 2, 2))
        incompatible = GeneralizedSchur(
            f2.S, f2.T, ones(typeof(real(zero(T))), 2), copy(f2.β), f2.Q, f2.Z
        )
        before = qzr_snapshot(incompatible)
        @test_throws InexactError ordschur!(incompatible, trues(2))
        @test qzr_snapshot(incompatible) == before
        # A nonregular 0/0 block cannot be crossed. Previous native swaps
        # only affect temporary work factors, so f remains entirely unchanged.
        rejected = qzr_factor(T, Matrix(Diagonal(T[0, 1, 2])), Matrix(Diagonal(T[0, 1, 1])))
        before = qzr_snapshot(rejected)
        for operation in (ordschur, ordschur!)
            @test_throws r"generalized Schur reorder rejected at row" operation(
                rejected, Bool[false, false, true]
            )
            @test qzr_snapshot(rejected) == before
        end
        # With no crossing requested, homogeneous 0/0 metadata is retained.
        g = ordschur(rejected, falses(3))
        @test iszero(g.α[1]) && iszero(g.β[1])
        for bad in (:lower_s, :lower_t, :overlapping_pair, :nonfinite)
            g = copy(f)
            if bad === :lower_s
                g.S[3, 1] = one(T)
            elseif bad === :lower_t
                g.T[2, 1] = one(T)
            elseif bad === :overlapping_pair
                g.S[2, 1] = one(T)
                g.S[3, 2] = one(T)
            else
                g.S[1, 1] = T(NaN)
            end
            before = qzr_snapshot(g)
            @test_throws Colmat.RustCall.RustError ordschur!(g, trues(3))
            @test isequal(qzr_snapshot(g), before)
        end
        closed = copy(f)
        close(closed.Z)
        @test_throws ArgumentError ordschur(closed, trues(3))
        @test_throws ArgumentError ordschur!(closed, trues(3))
        wrong = GeneralizedSchur(
            f.S, FaerMatrix(zeros(T, 2, 2)), copy(f.α), copy(f.β), f.Q, f.Z
        )
        @test_throws DimensionMismatch ordschur(wrong, trues(3))
        @test_throws DimensionMismatch ordschur!(wrong, trues(3))
    end
end

@testset "Strided QZ metadata and saved-owner lifetime" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        R = typeof(real(zero(T)))
        for layout in (:separate, :interleaved, :reversed)
            f = qzr_factor(T, Matrix(Diagonal(T[1, 2, 3])), Matrix{T}(I, 3, 3))
            data = fill(T(42), 8)
            alpha, beta = if layout === :separate
                (view(data, 1:3), view(data, 5:7))
            elseif layout === :interleaved
                (view(data, 1:2:5), view(data, 2:2:6))
            else
                (view(data, 5:-2:1), view(data, 6:-2:2))
            end
            h = GeneralizedSchur(f.S, f.T, alpha, beta, f.Q, f.Z)
            @test ordschur!(h, Bool[false, false, true]) === h
            @test h.values ≈ [3, 1, 2]
            @test data[8] == T(42)
            layout !== :separate && @test data[7] == T(42)
        end
        a = FaerMatrix(Matrix(Diagonal(T[1, 2, 3])))
        b = FaerMatrix(Matrix{T}(I, 3, 3))
        f = schur(a, b)
        a[1, 1] = T(99)
        close(a)
        close(b)
        @test ordschur(f, Bool[false, false, true]).values ≈ [3, 1, 2]
        @sync for _ in 1:2
            @async begin
                g = ordschur(f, Bool[false, true, false])
                @test g.values ≈ [2, 1, 3]
            end
        end
        empty = qzr_factor(T, zeros(T, 0, 0), zeros(T, 0, 0))
        values = T[]
        shared_empty = GeneralizedSchur(empty.S, empty.T, values, values, empty.Q, empty.Z)
        @test ordschur!(shared_empty, Bool[]) === shared_empty
        @test isempty(shared_empty.values)
    end
end

@testset "QZ current native owners and shape guards" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        f = qzr_factor(T, Matrix(Diagonal(T[1, 2, 3])), Matrix{T}(I, 3, 3))
        matrices = collect(qzr_matrices(f))
        # Two distinct Julia wrappers can retain the same checked Rust owner.
        source = matrices[3]
        matrices[4] = typeof(source)(source.handle, size(source))
        alias = GeneralizedSchur(
            matrices[1], matrices[2], copy(f.α), copy(f.β), matrices[3], matrices[4]
        )
        before = qzr_snapshot(alias)
        @test_throws ArgumentError ordschur!(alias, trues(3))
        @test qzr_snapshot(alias) == before
        @test ordschur(alias, Bool[false, false, true]).values ≈ [3, 1, 2]
        a = FaerMatrix(Matrix{T}(I, 3, 3))
        b = typeof(a)(a.handle, size(a))
        for operation in (schur!, hessenberg!)
            @test_throws ArgumentError operation(a, b)
            @test Matrix(a) == Matrix{T}(I, 3, 3)
        end
        h = hessenberg(FaerMatrix(Matrix{T}(I, 3, 3)), FaerMatrix(Matrix{T}(I, 3, 3)))
        shared = typeof(h.Q)(h.Q.handle, size(h.Q))
        bad = FaerGeneralizedHessenberg(h.H, h.T, h.Q, shared)
        @test_throws ArgumentError schur!(bad)
        # Cached Julia dimensions do not justify constructing oversized raw
        # slices. The native dimensions must be checked before pointer access.
        f = qzr_factor(T, Matrix(Diagonal(T[1, 2, 3])), Matrix{T}(I, 3, 3))
        original = map(Matrix, qzr_matrices(f))
        resize!(f.α, 2)
        resize!(f.β, 2)
        metadata = (copy(f.α), copy(f.β))
        for matrix in qzr_matrices(f)
            matrix.dims = (2, 2)
        end
        for operation in (ordschur, ordschur!)
            @test_throws Colmat.RustCall.RustError operation(f, Bool[false, true])
            @test (f.α, f.β) == metadata
        end
        for operation in (schur, hessenberg, schur!, hessenberg!)
            @test_throws r"current native dimension" operation(f.S, f.T)
            @test_throws r"current native dimension" scratch_requirements(
                operation, f.S, f.T
            )
        end
        for matrix in qzr_matrices(f)
            matrix.dims = (3, 3)
        end
        @test map(Matrix, qzr_matrices(f)) == original
    end
end

@testset "QZ current scalar tags before output access" begin
    for T in (Float32, ComplexF32, ComplexF64)
        f = qzr_factor(T, Matrix(Diagonal(T[1, 2, 3])), Matrix{T}(I, 3, 3))
        before = qzr_snapshot(f)
        U = T === Float32 ? ComplexF64 : Float32
        replacement = Matrix{U}(I, 3, 3)
        # Deliberately reload all erased native owners coherently with another
        # dtype. Matching the owners to each other does not validate Julia's
        # requested output dtype: the explicit scalar tag must also agree.
        for matrix in qzr_matrices(f)
            GC.@preserve replacement Colmat._native_call(
                Colmat.RustCrate.load,
                matrix,
                Ptr{UInt8}(pointer(replacement)),
                UInt(3),
                UInt(3),
                Colmat._tag(U),
            )
        end
        for operation in (ordschur, ordschur!)
            @test_throws r"requested scalar type" operation(f, Bool[false, false, true])
        end
        for operation in (schur, hessenberg, schur!, hessenberg!)
            @test_throws r"requested scalar type" operation(f.S, f.T)
            @test_throws r"requested scalar type" scratch_requirements(operation, f.S, f.T)
        end
        for (matrix, data) in zip(qzr_matrices(f), before[1])
            GC.@preserve data Colmat._native_call(
                Colmat.RustCrate.load,
                matrix,
                Ptr{UInt8}(pointer(data)),
                UInt(3),
                UInt(3),
                Colmat._tag(T),
            )
        end
        @test qzr_snapshot(f) == before
    end
end
