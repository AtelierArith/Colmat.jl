using Colmat, Test, LinearAlgebra, Random

@testset "Pivoted Cholesky construction and saved-factor workspace" begin
    rng = MersenneTwister(8600)
    team = FaerTeam(2)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        n in (0, 1, 2, 17, 65), cutoff in (-1.0, 0.0, 1e-6), block in (1, 2, 7, 128),
        resource in (nothing, team)

        x = randn(rng, T, n, min(n, 7))
        a = x * x' + Matrix{T}(I, n, n)
        source = FaerMatrix(a)
        input = Hermitian(source, isodd(n + block) ? :L : :U)
        opts = (; tol=cutoff, block_size=block, team=resource)
        req = scratch_requirements(cholesky, input, RowMaximum(); opts...)
        @test req == scratch_requirements(cholesky!, input, RowMaximum(); opts...)
        work = FaerScratch(cholesky, input, RowMaximum(); opts...)
        f = cholesky(input, RowMaximum(); opts..., scratch=work)
        epsilon = 200max(n, 1) * eps(real(T))
        @test f.rank == n
        @test f.factors isa FaerMatrix{T}
        @test Matrix(source) == a
        reserve!(work, reconstruct, f; team=resource)
        rebuilt = reconstruct(f; scratch=work, team=resource)
        @test Matrix(rebuilt) ≈ a rtol = epsilon atol = epsilon
        close(rebuilt)
        b = randn(rng, T, n, 2)
        dest = copy(b)
        reserve!(work, ldiv!, f, b; team=resource)
        @test ldiv!(f, dest; scratch=work, team=resource) === dest
        @test a * dest ≈ b rtol = epsilon atol = epsilon
        fill!(dest, 0)
        @test ldiv!(dest, f, b; scratch=work, team=resource) === dest
        @test a * dest ≈ b rtol = epsilon atol = epsilon
        right = randn(rng, T, 2, n)
        right_original = copy(right)
        reserve!(work, rdiv!, f, right; team=resource)
        @test rdiv!(right, f; scratch=work, team=resource) === right
        @test right * a ≈ right_original rtol = epsilon atol = epsilon
        reserve!(work, inv, f; team=resource)
        inverse = inv(f; scratch=work, team=resource)
        @test Matrix(inverse) * a ≈ Matrix{T}(I, n, n) rtol = epsilon atol = epsilon
        close(inverse)
        close(source)
        @test Matrix(f) ≈ a rtol = epsilon atol = epsilon
        close(f)
        close(work)
    end
    close(team)
end

@testset "Pivoted Cholesky rank, selected triangles and staged failures" begin
    team = FaerTeam(2)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        cutoff in (-1.0, 0.0, 1e-6), block in (1, 2, 128),
        uplo in (:L, :U)

        source = FaerMatrix(Matrix(Diagonal(T[4, 1, 0])))
        source[uplo === :L ? 1 : 3, uplo === :L ? 3 : 1] = T(NaN)
        input = Hermitian(source, uplo)
        before = Matrix(source)
        opts = (; tol=cutoff, block_size=block, team)
        work = FaerScratch(cholesky, input, RowMaximum(); opts...)
        @test_throws RankDeficientException cholesky!(
            input, RowMaximum(); opts..., scratch=work
        )
        @test isequal(Matrix(source), before)
        f = cholesky(input, RowMaximum(); opts..., scratch=work, check=false)
        @test f.rank == 2
        reserve!(work, reconstruct, f; team)
        rebuilt = reconstruct(f; team, scratch=work)
        @test Matrix(rebuilt) ≈ Matrix(Diagonal(T[4, 1, 0]))
        @test_throws RankDeficientException scratch_requirements(inv, f; team)
        @test_throws RankDeficientException ldiv!(f, ones(T, 3); team, scratch=work)
        close(rebuilt)
        close(f)
        close(source)
        close(work)
    end
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        source = FaerMatrix(T[4 1; 1 3])
        original = Matrix(source)
        short = FaerScratch(0)
        for cutoff in (-1.0, 0.0), op in (cholesky, cholesky!)
            @test_throws ArgumentError op(
                Hermitian(source),
                RowMaximum();
                tol=cutoff,
                block_size=2,
                team,
                scratch=short,
            )
            @test Matrix(source) == original
        end
        for block in (0, -1, big(typemax(UInt)) + 1)
            @test_throws ArgumentError cholesky(source, RowMaximum(); block_size=block)
            @test_throws ArgumentError scratch_requirements(
                cholesky, source, RowMaximum(); block_size=block
            )
        end
        f = cholesky(source, RowMaximum(); tol=-1, team)
        b = ones(T, 2, 2)
        for op in (ldiv!, rdiv!)
            dest = copy(b)
            @test_throws ArgumentError (
                if op === ldiv!
                    op(f, dest; team, scratch=short)
                else
                    op(dest, f; team, scratch=short)
                end
            )
            @test dest == b
        end
        old_dims = f.factors.dims
        f.factors.dims = (1, 1)
        @test_throws Exception scratch_requirements(inv, f; team)
        f.factors.dims = old_dims
        p = copy(f.p)
        f.p[1] = f.p[2]
        @test_throws ArgumentError scratch_requirements(reconstruct, f; team)
        @test_throws ArgumentError inv(f; team)
        copyto!(f.p, p)
        for dest in (f.factors, view(f.factors, :, :))
            saved = Matrix(f.factors)
            @test_throws ArgumentError ldiv!(f, dest; team)
            @test Matrix(f.factors) == saved
        end
        b[1, 1] = T(NaN)
        dest = fill(T(42), 2, 2)
        @test_throws Exception ldiv!(dest, f, b; team)
        @test dest == fill(T(42), 2, 2)
        close(short)
        @test_throws ArgumentError reconstruct(f; scratch=short)
        close(f)
        @test_throws ArgumentError scratch_requirements(inv, f)
        close(source)
    end
    close(team)
    source = FaerMatrix([4.0 1; 1 3])
    @test_throws ArgumentError cholesky(source, RowMaximum(); team)
    @test_throws ArgumentError scratch_requirements(cholesky, source, RowMaximum(); team)
    close(source)
end

@testset "Pivoted saved factors: promotion, upper storage and native RHS" begin
    rng = MersenneTwister(8601)
    team = FaerTeam(2)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        P in (Float32, Float64, ComplexF32, ComplexF64),
        upper in (false, true)

        x = randn(rng, T, 5, 5)
        a = x * x' + Matrix{T}(I, 5, 5)
        source = FaerMatrix(a)
        f = cholesky(Hermitian(source), RowMaximum(); tol=-1)
        if upper
            g = CholeskyPivoted(
                copy(adjoint(f.factors)), 'U', copy(f.p), f.rank, f.tol, f.info
            )
            close(f)
            f = g
        end
        f.factors[upper ? 5 : 1, upper ? 1 : 5] = T(NaN)
        S = promote_type(T, P)
        epsilon = 10000eps(real(T))
        b = randn(rng, P, 5, 3)
        native = FaerMatrix(b)
        work = FaerScratch(\, f, native; team)
        answer = (\)(f, native; scratch=work, team)
        @test answer isa FaerMatrix{S}
        @test S.(a) * Matrix(answer) ≈ S.(b) atol = epsilon rtol = epsilon
        close(answer)
        close(native)
        right = randn(rng, P, 3, 5)
        native = FaerMatrix(right)
        reserve!(work, /, f, native; team)
        answer = (/)(native, f; scratch=work, team)
        @test answer isa FaerMatrix{S}
        @test Matrix(answer) * S.(a) ≈ S.(right) atol = epsilon rtol = epsilon
        close(answer)
        close(native)
        reserve!(work, ldiv!, f, b; team)
        dest = zeros(S, 5, 3)
        ldiv!(dest, f, b; scratch=work, team)
        @test S.(a) * dest ≈ S.(b) atol = epsilon rtol = epsilon
        close(work)
        close(source)
        close(f)
    end
    close(team)
    @test isempty(detect_ambiguities(Colmat, Base, LinearAlgebra; recursive=true))
end

struct PivotedToleranceEffect <: Real
    action::Function
    PivotedToleranceEffect(action::Function) = new(action)
end
Base.isfinite(::PivotedToleranceEffect) = true
Base.Float64(t::PivotedToleranceEffect) = (t.action(); 0.0)
struct PivotedRhsEffect{T} <: AbstractMatrix{T}
    data::Matrix{T}
    action::Function
end
Base.size(x::PivotedRhsEffect) = size(x.data)
Base.getindex(x::PivotedRhsEffect, i::Int, j::Int) = x.data[i, j]
Base.Matrix{P}(x::PivotedRhsEffect) where {P} = (x.action(); Matrix{P}(x.data))

struct PivotedInvalidTolerance <: Real end
Base.isfinite(::PivotedInvalidTolerance) = true
Base.Float64(::PivotedInvalidTolerance) = 0
struct PivotedWrongMatrix{S} <: AbstractMatrix{Float64}
    data::Matrix{S}
end
Base.size(::PivotedWrongMatrix) = (2, 2)
Base.getindex(::PivotedWrongMatrix, i::Int, j::Int) = 1.0
Base.Matrix{P}(a::PivotedWrongMatrix) where {P} = a.data
struct PivotedWrongVector{S} <: AbstractVector{Float64}
    data::Vector{S}
end
Base.size(::PivotedWrongVector) = (2,)
Base.getindex(::PivotedWrongVector, i::Int) = 1.0
Base.Vector{P}(a::PivotedWrongVector) where {P} = a.data

@testset "Pivoted workspace: views, conversions and query slices" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), uplo in (:L, :U)
        owner = FaerMatrix(Matrix{T}(I, 7, 7))
        v = view(owner, 7:-2:1, 7:-2:1)
        data = T[5 1 0 1; 1 4 1 0; 0 1 3 1; 1 0 1 4]
        T <: Complex && (data[2, 1] += T(im); data[1, 2]=conj(data[2, 1]))
        copyto!(v, data)
        auto = cholesky(v, RowMaximum(); block_size=2)
        @test Matrix(auto) ≈ data rtol = 1000eps(real(T)) atol = 1000eps(real(T))
        close(auto)
        parent_before = Matrix(owner)
        input = T <: Real && uplo === :L ? Symmetric(v, uplo) : Hermitian(v, uplo)
        work = FaerScratch(cholesky!, input, RowMaximum(); block_size=2)
        f = fetch(Threads.@spawn cholesky!(input, RowMaximum(); block_size=2, scratch=work))
        @test Matrix(f) ≈ data rtol = 1000eps(real(T)) atol = 1000eps(real(T))
        parent_before[7:-2:1, 7:-2:1] = uplo === :L ? Matrix(f.factors) : Matrix(f.factors)'
        @test Matrix(owner) == parent_before
        close(owner)
        @test Matrix(f) ≈ data rtol = 1000eps(real(T)) atol = 1000eps(real(T))
        close(f)
        close(work)
    end
    source = FaerMatrix([4.0 1; 1 3])
    converted = Ref(0)
    once_tol = PivotedToleranceEffect(() -> (converted[] += 1))
    scratch_requirements(cholesky, source, RowMaximum(); tol=once_tol)
    @test converted[] == 1
    converted[] = 0
    counted = cholesky(source, RowMaximum(); tol=once_tol)
    @test converted[] == 1
    @test Matrix(counted) ≈ Matrix(source)
    close(counted)
    original = source.dims
    tol = PivotedToleranceEffect(() -> (source.dims = (1, 1)))
    @test_throws Exception scratch_requirements(cholesky, source, RowMaximum(); tol)
    source.dims = original
    @test_throws ArgumentError scratch_requirements(
        cholesky, source, RowMaximum(); tol=PivotedInvalidTolerance()
    )
    f = cholesky(source, RowMaximum())
    dest = fill(42.0, 2, 2)
    for (bad, error) in (
        (PivotedWrongMatrix(ones(Int64, 2, 2)), ArgumentError),
        (PivotedWrongMatrix(ones(2, 1)), DimensionMismatch),
    )
        @test_throws error ldiv!(dest, f, bad)
        @test dest == fill(42.0, 2, 2)
    end
    for (bad, error) in (
        (PivotedWrongVector(ones(Int64, 2)), ArgumentError),
        (PivotedWrongVector(ones(3)), DimensionMismatch),
    )
        v = fill(42.0, 2)
        @test_throws error ldiv!(v, f, bad)
        @test v == fill(42.0, 2)
    end
    p = copy(f.p)
    rhs = PivotedRhsEffect(ones(2, 2), () -> (f.p[1] = f.p[2]))
    @test_throws ArgumentError ldiv!(dest, f, rhs)
    @test dest == fill(42.0, 2, 2)
    copyto!(f.p, p)
    @test_throws InexactError ldiv!(dest, f, fill(1im, 2, 2))
    @test dest == fill(42.0, 2, 2)
    work = FaerScratch(ldiv!, f, ones(2, 2))
    rhs = PivotedRhsEffect(ones(2, 2), () -> close(work))
    @test_throws ArgumentError ldiv!(dest, f, rhs; scratch=work)
    @test dest == fill(42.0, 2, 2)
    tiny = FaerMatrix(reshape([1e-300], 1, 1))
    small_factor = cholesky(tiny, RowMaximum())
    small_dest = fill(Float32(42), 1, 1)
    @test_throws ArgumentError ldiv!(small_dest, small_factor, ones(1, 1))
    @test small_dest == fill(Float32(42), 1, 1)
    close(small_factor)
    close(tiny)
    native_dest = FaerMatrix(fill(42.0, 2, 2))
    rhs = PivotedRhsEffect(ones(2, 2), () -> (native_dest.dims = (1, 1)))
    @test_throws Exception ldiv!(native_dest, f, rhs)
    native_dest.dims = (2, 2)
    @test Matrix(native_dest) == fill(42.0, 2, 2)
    close(native_dest)
    for factor in (true, false), len in (0, 1, 3)
        out = fill(UInt(43), len)
        @test_throws Colmat.RustCall.RustError (
            if factor
                Colmat._native_call(
                    Colmat.RustCrate.pivoted_llt_requirements,
                    UInt(2),
                    Int32(1),
                    0.0,
                    UInt(2),
                    nothing,
                    out,
                )
            else
                Colmat._native_call(
                    Colmat.RustCrate.pivoted_llt_apply_requirements,
                    UInt(2),
                    UInt(1),
                    Int32(1),
                    Int32(0),
                    nothing,
                    out,
                )
            end
        )
        @test out == fill(UInt(43), len)
    end
    out = fill(UInt(44), 2)
    @test_throws Colmat.RustCall.RustError Colmat._native_call(
        Colmat.RustCrate.pivoted_llt_apply_requirements,
        UInt(0),
        typemax(UInt),
        Int32(1),
        Int32(0),
        nothing,
        out,
    )
    @test out == UInt[44, 44]
    @test_throws ArgumentError Colmat._native_call(
        Colmat.RustCrate.pivoted_llt_requirements,
        UInt(2),
        Int32(1),
        0.0,
        UInt(2),
        nothing,
        zeros(Float64, 2),
    )
    close(f)
    close(source)
    @test isempty(detect_ambiguities(Colmat, Base, LinearAlgebra; recursive=true))
end
