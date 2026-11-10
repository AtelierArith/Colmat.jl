"""
    FaerSparseSymbolic(P::FaerSparsity; kind=:lu, uplo=:L)
    FaerSparseSymbolic(A::FaerSparseMatrix; kind=:lu, uplo=:L)

Own faer's symbolic sparse LU, QR or Cholesky analysis (`kind=:lu/:qr/:cholesky`).
`LinearAlgebra.factorize(plan,A)` computes independent numeric factors without repeating the
analysis. A must have exactly the original stored pattern, including stored
zeros; CSC/CSR orientation and numeric scalar type may differ. Wide sparse QR uses a square-prefix native analysis and saved-reflector projections.
Cholesky accesses `uplo`.

Cholesky accepts `ordering=:default/:amd/:identity` or a permutation, AMD tuning
`dense=10.0, aggressive=true`, and `algorithm=:auto/:simplicial/:supernodal`.
`plan.p` returns an independent FaerPermutation; `plan.algorithm` reports the
actual chosen algorithm. `plan.factor_storage` counts numeric scalar slots,
including supernodal padding. Explicit `kind=:qr, algorithm=:simplicial` accepts identity/default or custom
column ordering and exposes thin Q, sparse R and column permutation on numeric
factors. Explicit simplicial LU/QR additionally accept `ordering=:colamd` with
`dense_row=0.5, dense_col=0.5, aggressive=true`. COLAMD is computed once from the
stored pattern and retained by symbolic reuse and numeric updates. AMD tuning
remains Cholesky-only. Auto/supernodal LU/QR accepts default/COLAMD, identity or custom column ordering.
COLAMD accepts `dense_row`, `dense_col`, `aggressive`. QR exposes `F.p/F.pcol`.
Cholesky and auto/supernodal LU/QR accept faer's supernodal controls:
`supernodal_threshold` (`supernodal_flop_ratio_threshold`, nonnegative, `algorithm=:auto`
only; default 1.0) and `relax=[(n, z), ...]` (`SymbolicSupernodalParams.relax`, `0 ≤ z ≤ 1`;
omitted keeps faer's default, empty disables relaxation). Auto/supernodal QR also
accepts `blocking_threshold` (nonnegative integer, the dense front
`QrParams.blocking_threshold`, default 48²), kept by the plan for reuse and `qr!`;
QR plans and factors report `algorithm` (wide QR: its square head analysis).

`scratch=FaerScratch(FaerSparseSymbolic, P; kwargs...)` supplies the ordering
kernel workspace for Cholesky AMD, wide QR COLAMD, explicit simplicial
`ordering=:colamd` and the explicit simplicial QR symbolic kernels, with the
same analysis as omitting it; see
`scratch_requirements(FaerSparseSymbolic, P; kwargs...)` for what stays allocated.
"""
mutable struct FaerSparseSymbolic{K,H}
    handle::Union{Nothing,H}
    dims::Tuple{Int,Int}
    uplo::Symbol
end
"""
    FaerSparseFactorization

Reusable Rust-owned sparse factors returned by `lu`, `qr`, `cholesky` and
`LinearAlgebra.factorize(plan,A)` for FaerSparseMatrix. `F \\ B` reuses numeric factors; QR
solves full-column-rank tall/square problems, and minimum-norm wide problems
using native SVD of saved R. LU/Cholesky also support
transpose/adjoint solves. Sparse QR supports minimum-norm transpose/adjoint solves for full-column-rank
factors with every algorithm. Inputs and symbolic plans
may be closed independently.
Mixed precision RHS reuse converted saved factors for all sparse algorithms,
without reading or refactorizing the source snapshot. `ldiv!` stages the result before overwriting.
Sparse Cholesky additionally exposes independent sparse CSC `F.L` and `F.U`,
with `F.L * F.U ≈ A[F.p,F.p]`, and supports `Matrix(F)`, `FaerSparseMatrix(F)`,
`det`, `logdet` and `logabsdet`. Reconstruction returns the original row/column
order. Extracted components remain usable after F is updated or closed.
Sparse LU exposes independent CSC `F.L`, `F.U` and permutations `F.p`, `F.q`
for every algorithm, with `F.L * F.U ≈ A[F.p,F.q]`. It also supports `Matrix(F)`, `FaerMatrix(F)`,
`FaerSparseMatrix(F)` and `reconstruct(F)`, from current native L/U and inverse
row/column permutations. Sparse QR exposes independent thin `Q/Qthin`, square orthogonal/unitary `Qfull`,
sparse `R`, `p/pcol` and dense
reconstruction for all algorithms, using the current saved numeric buffers.
`F.Qoperator` is the lazy m×m Q (SPQR's `F.Q`): an independent `FaerLinearOperator`
applying saved Householder reflectors with optional caller scratch/team, and `F.prow`
is the identity row permutation, since Q is stored in the original row order.
Sparse QR also reports the actual `F.algorithm`. Sparse Cholesky exposes
independent `F.p`, the actual `F.algorithm` and `F.factor_storage` (numeric
scalar slots, including supernodal padding).
`cholesky(A; perm=p)` selects a custom ordering using Julia's sparse keyword;
`ordering=p` is an alternative. Numeric updates preserve the symbolic controls.
"""
mutable struct FaerSparseFactorization{T,K,H,P,A} <: LinearAlgebra.Factorization{T}
    handle::Union{Nothing,H}
    dims::Tuple{Int,Int}
    info::Int
    plan::P
    source::A
end
const _SparseNativeFactor = Union{FaerSparseSymbolic,FaerSparseFactorization}
function _handle(f::_SparseNativeFactor)
    h = getfield(f, :handle)
    isnothing(h) && throw(ArgumentError("sparse analysis/factorization has been closed"))
    return h
end
_native_argument(f::_SparseNativeFactor) = _handle(f)
_is_native_owner(::_SparseNativeFactor) = true
Base.size(f::_SparseNativeFactor) = getfield(f, :dims)
function Base.size(f::_SparseNativeFactor, d::Integer)
    if d < 1
        throw(ArgumentError("dimension must be positive"))
    elseif d <= 2
        size(f)[d]
    else
        1
    end
end
function Base.isopen(f::_SparseNativeFactor)
    return lock(() -> !isnothing(getfield(f, :handle)), _matrix_lock)
end
function Base.close(f::_SparseNativeFactor)
    lock(_matrix_lock) do
        f.handle = nothing
        if f isa FaerSparseFactorization
            close(f.plan)
            close(f.source)
        end
    end
    return nothing
end
LinearAlgebra.issuccess(f::FaerSparseFactorization) = iszero(f.info)
function _sparse_symbolic_options(
    p::Union{FaerSparsity,FaerSparseMatrix,FaerBorrowedSparseMatrix};
    kind=:lu,
    uplo=:L,
    ordering=:default,
    dense::Real=10.0,
    dense_row::Real=0.5,
    dense_col::Real=0.5,
    aggressive::Bool=true,
    algorithm=:auto,
    supernodal_threshold=nothing,
    relax=nothing,
    blocking_threshold=nothing,
)
    code = if kind === :lu
        0
    elseif kind === :qr
        1
    elseif kind === :cholesky
        2
    else
        throw(ArgumentError("kind must be :lu, :qr or :cholesky"))
    end
    uplo in (:L, :U) || throw(ArgumentError("uplo must be :L or :U"))
    kind !== :qr &&
        size(p, 1) != size(p, 2) &&
        throw(DimensionMismatch("sparse LU/Cholesky requires a square matrix"))
    threshold = Float64(dense)
    isfinite(threshold) || throw(ArgumentError("dense must be finite"))
    algo = if algorithm === :auto
        0
    elseif algorithm === :simplicial
        1
    elseif algorithm === :supernodal
        2
    else
        throw(ArgumentError("algorithm must be :auto, :simplicial or :supernodal"))
    end
    use_colamd = kind in (:lu, :qr) && algo == 1 && ordering === :colamd
    native_colamd = kind in (:lu, :qr) && algo != 1 && ordering in (:default, :colamd)
    native_explicit =
        kind in (:lu, :qr) &&
        algo != 1 &&
        (ordering === :identity || ordering isa AbstractVector{<:Integer})
    row, col = Float64(dense_row), Float64(dense_col)
    isfinite(row) && isfinite(col) ||
        throw(ArgumentError("COLAMD thresholds must be finite"))
    !(use_colamd || native_colamd) &&
        (row != 0.5 || col != 0.5) &&
        throw(
            ArgumentError(
                "dense_row/dense_col require COLAMD ordering for QR or explicit simplicial LU",
            ),
        )
    if kind !== :cholesky &&
        !(
            kind in (:lu, :qr) &&
            algo == 1 &&
            threshold == 10.0 &&
            (aggressive || use_colamd)
        ) &&
        !((native_colamd || (native_explicit && aggressive)) && threshold == 10.0) &&
        (ordering !== :default || threshold != 10.0 || !aggressive || algo != 0)
        throw(
            ArgumentError(
                "these controls require Cholesky or supported sparse LU/QR ordering options"
            ),
        )
    end
    kind in (:lu, :qr) &&
        ordering === :amd &&
        throw(
            ArgumentError(
                "LU/QR ordering supports :default, :identity, :colamd or a permutation"
            ),
        )
    ratio, relax_sizes, relax_fractions = _sparse_supernodal_tuning(
        kind, algo, supernodal_threshold, relax
    )
    blocking = _sparse_qr_blocking(kind, algo, blocking_threshold)
    custom = use_colamd || ordering isa AbstractVector{<:Integer}
    ordercode = if custom
        2
    elseif (native_colamd || ordering in (:default, :amd))
        0
    elseif ordering === :identity
        1
    else
        throw(
            ArgumentError(
                "ordering must be :default, :amd, :identity, :colamd or a permutation"
            ),
        )
    end
    return (;
        kind,
        code,
        uplo,
        ordering,
        threshold,
        aggressive,
        algo,
        row,
        col,
        use_colamd,
        custom,
        ordercode,
        ratio,
        relax_sizes,
        relax_fractions,
        blocking,
    )
end
# Dense `QrParams.blocking_threshold` for auto/supernodal sparse QR fronts.
function _sparse_qr_blocking(kind, algo, threshold)
    isnothing(threshold) && return nothing
    kind === :qr && algo != 1 || throw(
        ArgumentError(
            "blocking_threshold requires kind=:qr with algorithm=:auto or :supernodal"
        ),
    )
    threshold isa Integer && threshold >= 0 ||
        throw(ArgumentError("blocking_threshold must be a nonnegative integer"))
    return threshold >= typemax(UInt) ? typemax(UInt) : UInt(threshold)
end
# faer's supernodal_flop_ratio_threshold (auto selection) and
# SymbolicSupernodalParams.relax for sparse Cholesky and non-simplicial LU/QR.
function _sparse_supernodal_tuning(kind, algo, threshold, relax)
    ratio = NaN
    if !isnothing(threshold)
        algo == 0 || throw(ArgumentError("supernodal_threshold requires algorithm=:auto"))
        threshold isa Real ||
            throw(ArgumentError("supernodal_threshold must be a nonnegative real"))
        ratio = Float64(threshold)
        !isnan(ratio) && ratio >= 0 ||
            throw(ArgumentError("supernodal_threshold must be a nonnegative real"))
    end
    isnothing(relax) && return ratio, nothing, nothing
    algo != 1 || throw(ArgumentError("relax requires algorithm=:auto or :supernodal"))
    sizes = UInt[]
    fractions = Float64[]
    for item in relax
        item isa Tuple{Integer,Real} ||
            throw(ArgumentError("relax must contain (column count, zero fraction) tuples"))
        n, z = item
        n >= 0 || throw(ArgumentError("relax column counts must be nonnegative"))
        zf = Float64(z)
        0 <= zf <= 1 || throw(ArgumentError("relax zero fractions must lie in [0, 1]"))
        push!(sizes, n >= typemax(UInt) ? typemax(UInt) : UInt(n))
        push!(fractions, zf)
    end
    return ratio, sizes, fractions
end
function _sparse_supernodal_tuning!(h, o)
    isnothing(o.blocking) ||
        _native_call(RustCrate.sparse_symbolic_qr_blocking, h, o.blocking)
    isnan(o.ratio) && isnothing(o.relax_sizes) && return nothing
    custom = !isnothing(o.relax_sizes)
    sizes = custom ? Vector{UInt}(o.relax_sizes) : UInt[]
    fractions = custom ? Vector{Float64}(o.relax_fractions) : Float64[]
    GC.@preserve sizes fractions _native_call(
        RustCrate.sparse_symbolic_tuning,
        h,
        o.ratio,
        custom,
        sizes,
        fractions,
        UInt(length(sizes)),
    )
    return nothing
end
function _sparse_symbolic_requirements(p::FaerSparsity, o)
    if o.use_colamd
        # COLAMD runs first, then the analysis reuses the same buffer.
        a = _ordering_requirements(p, size(p), false)
        b = _sparse_native_requirements(p, o)
        return (bytes=max(a.bytes, b.bytes), alignment=max(a.alignment, b.alignment))
    end
    return _sparse_native_requirements(p, o)
end
function _sparse_native_requirements(p::FaerSparsity, o)
    info = zeros(UInt, 2)
    GC.@preserve info _native_call(
        RustCrate.sparse_analysis_requirements,
        p,
        UInt(size(p, 1)),
        UInt(size(p, 2)),
        Int32(o.code),
        Int32(o.ordercode),
        Int32(o.algo),
        info,
    )
    return (bytes=Int(info[1]), alignment=Int(info[2]))
end
# Patterns analyzed in place: a borrowed Julia CSC matrix (per-call 0-based
# buffers) or a CSC owner. The plan keeps the only pattern copy.
const _RawAnalysisSource = Union{FaerSparseMatrix,FaerBorrowedSparseMatrix}
_analysis_open(p::Union{FaerSparsity,FaerSparseMatrix}) = _handle(p)
_analysis_open(::FaerBorrowedSparseMatrix) = nothing
function _with_analysis_pattern(f, p::FaerBorrowedSparseMatrix)
    return _with_borrowed_sparse((m, n, ptr, idx, _, k, _) -> f(m, n, ptr, idx, k), p)
end
function _with_analysis_pattern(f, p::FaerSparseMatrix)
    return _with_raw_sparse((m, n, ptr, idx, _, k) -> f(m, n, ptr, idx, k), p)
end
function _sparse_symbolic_requirements(p::_RawAnalysisSource, o)
    info = zeros(UInt, 2)
    GC.@preserve info _native_call(
        RustCrate.sparse_analysis_requirements_shape,
        UInt(size(p, 1)),
        UInt(size(p, 2)),
        UInt(SparseArrays.nnz(p)),
        Int32(o.code),
        Int32(o.ordercode),
        Int32(o.algo),
        info,
    )
    return (bytes=Int(info[1]), alignment=Int(info[2]))
end
function _sparse_analyze!(h, p::FaerSparsity, scratch, args...)
    isnothing(scratch) && return _native_call(RustCrate.sparse_analyze, h, p, args...)
    return _native_call(
        RustCrate.sparse_analyze_workspace,
        h,
        p,
        scratch,
        UInt(size(p, 1)),
        UInt(size(p, 2)),
        args...,
    )
end
function _sparse_analyze!(h, p::_RawAnalysisSource, scratch, args...)
    _with_analysis_pattern(p) do pattern...
        return if isnothing(scratch)
            _native_call(RustCrate.sparse_analyze_raw, h, pattern..., args...)
        else
            _native_call(
                RustCrate.sparse_analyze_workspace_raw, h, pattern..., scratch, args...
            )
        end
    end
end
FaerSparseSymbolic(p::FaerSparsity; kwargs...) = _sparse_symbolic(p; kwargs...)
function _sparse_symbolic(p; scratch=nothing, kwargs...)
    o = _sparse_symbolic_options(p; kwargs...)
    # Explicit simplicial COLAMD orders a native pattern owner first.
    if p isa _RawAnalysisSource && o.use_colamd
        pattern = FaerSparsity(p)
        try
            return _sparse_symbolic(pattern; scratch, kwargs...)
        finally
            close(pattern)
        end
    end
    isnothing(scratch) ||
        scratch isa FaerScratch ||
        throw(ArgumentError("scratch must be a FaerScratch"))
    kind, uplo, row, col = o.kind, o.uplo, o.row, o.col
    lock(_matrix_lock) do
        _analysis_open(p)
        if !isnothing(scratch)
            req = _sparse_symbolic_requirements(p, o)
            cap = scratch_capacity(scratch)
            cap.bytes >= req.bytes && cap.alignment >= req.alignment || throw(
                ArgumentError(
                    "insufficient symbolic analysis scratch; call reserve!(scratch, FaerSparseSymbolic, A; kwargs...)",
                ),
            )
        end
        perm = if o.use_colamd
            colamd(p; dense_row=row, dense_col=col, o.aggressive, scratch)
        elseif o.custom
            _as_permutation(o.ordering)
        else
            Ptr{UInt8}(C_NULL)
        end
        try
            o.custom && _check_permutation(p, perm, true)
            h = RustCrate.SparseSymbolic()
            _sparse_supernodal_tuning!(h, o)
            _sparse_analyze!(
                h,
                p,
                scratch,
                Int32(o.code),
                uplo === :L,
                Int32(o.ordercode),
                perm,
                o.threshold,
                o.aggressive,
                Int32(o.algo),
                row,
                col,
            )
            return FaerSparseSymbolic{kind,typeof(h)}(h, size(p), uplo)
        finally
            o.use_colamd && close(perm)
        end
    end
end
# Borrowed Julia CSC and CSC owner inputs are analyzed in place; a CSR owner's
# pattern is converted to CSC once (faer's symbolic analyses are CSC-only).
const _SparseFactorInput{T} = Union{FaerSparseMatrix{T},FaerBorrowedSparseMatrix{T}}
function FaerSparseSymbolic(a::_SparseFactorInput; kwargs...)
    return if a isa FaerSparseMatrix && a.format === :csr
        FaerSparseSymbolic(FaerSparsity(a); kwargs...)
    else
        _sparse_symbolic(a; kwargs...)
    end
end
function Base.copy(p::FaerSparseSymbolic{K}) where {K}
    lock(_matrix_lock) do
        h = RustCrate.SparseSymbolic()
        _native_call(RustCrate.sparse_symbolic_clone, h, p)
        return FaerSparseSymbolic{K,typeof(h)}(h, size(p), p.uplo)
    end
end
Base.deepcopy_internal(p::FaerSparseSymbolic, d::IdDict) = get!(() -> copy(p), d, p)
function _sparse_factor_inputs(p::FaerSparseSymbolic{K}, a::FaerSparseMatrix{T}) where {K,T}
    _handle(p)
    _handle(a)
    kind = if K === :lu
        0
    elseif K === :qr
        1
    elseif K === :cholesky
        2
    else
        throw(ArgumentError("unsupported sparse factor kind"))
    end
    p.uplo in (:L, :U) || throw(ArgumentError("invalid sparse factor triangle metadata"))
    csr = _sparse_format(a.format)
    _native_call(
        RustCrate.sparse_factor_matrix_validate,
        a,
        UInt(size(a, 1)),
        UInt(size(a, 2)),
        _tag(T),
        csr,
        UInt(a.stored),
    )
    _native_call(
        RustCrate.sparse_factor_plan_validate,
        p,
        UInt(size(p, 1)),
        UInt(size(p, 2)),
        Int32(kind),
        p.uplo === :L,
    )
    return nothing
end
function LinearAlgebra.factorize(
    p::FaerSparseSymbolic{K},
    a::FaerSparseMatrix{T};
    check::Bool=true,
    scratch=nothing,
    team=nothing,
) where {K,T}
    size(p) == size(a) ||
        throw(DimensionMismatch("symbolic analysis and numeric matrix shapes must match"))
    lock(_matrix_lock) do
        _sparse_factor_inputs(p, a)
        (isnothing(scratch) && isnothing(team)) ||
            _sparse_factor_resource_validate(p, a, scratch, team)
        return _factorize_snapshot(p, copy(a); check, scratch, team)
    end
end
# A borrowed Julia matrix is copied once into the owned snapshot the factors keep.
function LinearAlgebra.factorize(
    p::FaerSparseSymbolic,
    a::FaerBorrowedSparseMatrix;
    check::Bool=true,
    scratch=nothing,
    team=nothing,
)
    size(p) == size(a) ||
        throw(DimensionMismatch("symbolic analysis and numeric matrix shapes must match"))
    lock(_matrix_lock) do
        _handle(p)
        snapshot = FaerSparseMatrix(a)
        _sparse_factor_inputs(p, snapshot)
        (isnothing(scratch) && isnothing(team)) ||
            _sparse_factor_resource_validate(p, snapshot, scratch, team)
        return _factorize_snapshot(p, snapshot, a; check, scratch, team)
    end
end
function _sparse_values_finite(p::FaerSparseSymbolic{K}, a::FaerSparseMatrix) where {K}
    K === :cholesky || return all(isfinite, Vector(SparseArrays.nonzeros(a)))
    rows, cols, values = SparseArrays.findnz(a)
    return all(
        i -> (p.uplo === :L ? rows[i] < cols[i] : rows[i] > cols[i]) || isfinite(values[i]),
        eachindex(values),
    )
end
# Reads the borrowed parent in place (no value snapshot); same accessed entries.
function _sparse_values_finite(
    p::FaerSparseSymbolic{K}, a::FaerBorrowedSparseMatrix
) where {K}
    s = parent(a)
    rows, values = SparseArrays.rowvals(s), SparseArrays.nonzeros(s)
    for j in 1:size(s, 2), q in SparseArrays.nzrange(s, j)
        K === :cholesky && (p.uplo === :L ? rows[q] < j : rows[q] > j) && continue
        isfinite(values[q]) || return false
    end
    return true
end
# `source` is the matrix the snapshot was copied from, checked for finiteness.
function _factorize_snapshot(
    p::FaerSparseSymbolic{K},
    snapshot::FaerSparseMatrix{T},
    source=snapshot;
    check,
    scratch,
    team,
) where {K,T}
    lock(_matrix_lock) do
        _sparse_values_finite(p, source) || throw(
            ArgumentError(
                "sparse factorization requires finite values in the accessed entries"
            ),
        )
        plan = copy(p)
        h = RustCrate.SparseFactor()
        info = try
            if isnothing(scratch) && isnothing(team)
                Int(_native_call(RustCrate.sparse_factorize, h, plan, snapshot))
            else
                _sparse_factor_resource_workspace(h, plan, snapshot, scratch, team)
            end
        catch e
            if e isa RustCall.RustError && occursin(
                "numeric matrix sparsity pattern must match", sprint(showerror, e)
            )
                throw(
                    ArgumentError(
                        "numeric matrix sparsity pattern must match the symbolic analysis exactly",
                    ),
                )
            end
            rethrow()
        end
        if check && !iszero(info)
            K === :cholesky && throw(LinearAlgebra.PosDefException(info))
            throw(LinearAlgebra.SingularException(info))
        end
        return FaerSparseFactorization{T,K,typeof(h),typeof(plan),typeof(snapshot)}(
            h, size(snapshot), info, plan, snapshot
        )
    end
end
function LinearAlgebra.lu(
    a::_SparseFactorInput;
    check::Bool=true,
    algorithm=:auto,
    ordering=:default,
    dense_row::Real=0.5,
    dense_col::Real=0.5,
    aggressive::Bool=true,
    supernodal_threshold=nothing,
    relax=nothing,
    scratch=nothing,
    team=nothing,
)
    return LinearAlgebra.factorize(
        FaerSparseSymbolic(
            a;
            kind=:lu,
            algorithm,
            ordering,
            dense_row,
            dense_col,
            aggressive,
            supernodal_threshold,
            relax,
        ),
        a;
        check,
        scratch,
        team,
    )
end
function LinearAlgebra.lu(a::_SparseFactorInput, ::LinearAlgebra.RowMaximum; kwargs...)
    return LinearAlgebra.lu(a; kwargs...)
end
function LinearAlgebra.qr(
    a::_SparseFactorInput;
    algorithm=:auto,
    ordering=:default,
    dense_row::Real=0.5,
    dense_col::Real=0.5,
    aggressive::Bool=true,
    supernodal_threshold=nothing,
    relax=nothing,
    blocking_threshold=nothing,
    scratch=nothing,
    team=nothing,
)
    return LinearAlgebra.factorize(
        FaerSparseSymbolic(
            a;
            kind=:qr,
            algorithm,
            ordering,
            dense_row,
            dense_col,
            aggressive,
            supernodal_threshold,
            relax,
            blocking_threshold,
        ),
        a;
        scratch,
        team,
    )
end
const _SparseSelfAdjoint = Union{
    LinearAlgebra.Hermitian{<:Any,<:_SparseFactorInput},
    LinearAlgebra.Symmetric{<:Real,<:_SparseFactorInput},
}
function LinearAlgebra.cholesky(
    a::_SparseSelfAdjoint;
    check::Bool=true,
    perm=nothing,
    ordering=:default,
    scratch=nothing,
    team=nothing,
    kwargs...,
)
    !isnothing(perm) &&
        ordering !== :default &&
        throw(ArgumentError("specify either perm or ordering, not both"))
    selected = isnothing(perm) ? ordering : perm
    source = parent(a)
    return LinearAlgebra.factorize(
        FaerSparseSymbolic(
            source;
            kind=:cholesky,
            uplo=a.uplo == 'L' ? :L : :U,
            ordering=selected,
            kwargs...,
        ),
        source;
        check,
        scratch,
        team,
    )
end
function LinearAlgebra.cholesky(a::_SparseFactorInput; check::Bool=true, kwargs...)
    LinearAlgebra.ishermitian(a) || throw(
        ArgumentError(
            "Cholesky requires Hermitian input; use Hermitian(A,:L) or Hermitian(A,:U) to select a stored triangle",
        ),
    )
    return LinearAlgebra.cholesky(LinearAlgebra.Hermitian(a, :L); check, kwargs...)
end
function LinearAlgebra.factorize(a::_SparseFactorInput)
    return size(a, 1) == size(a, 2) ? LinearAlgebra.lu(a) : LinearAlgebra.qr(a)
end
function Base.:\(
    a::_SparseFactorInput, b::Union{AbstractVector{<:Number},AbstractMatrix{<:Number}}
)
    return LinearAlgebra.factorize(a) \ b
end
function Base.copy(f::FaerSparseFactorization{T,K}) where {T,K}
    lock(_matrix_lock) do
        _handle(f)
        h = RustCrate.SparseFactor()
        LinearAlgebra.issuccess(f) && _native_call(RustCrate.sparse_factor_clone, h, f)
        plan = copy(f.plan)
        source = if isopen(f.source)
            copy(f.source)
        else
            typeof(f.source)(
                nothing, f.source.dims, f.source.stored, f.source.format, f.source.epoch
            )
        end
        return FaerSparseFactorization{T,K,typeof(h),typeof(plan),typeof(source)}(
            h, size(f), f.info, plan, source
        )
    end
end
Base.deepcopy_internal(f::FaerSparseFactorization, d::IdDict) = get!(() -> copy(f), d, f)
const _SparseTransposedFactor = Union{
    LinearAlgebra.TransposeFactorization{<:Any,<:FaerSparseFactorization},
    LinearAlgebra.AdjointFactorization{<:Any,<:FaerSparseFactorization},
}
function _sparse_factor_solve(
    f::FaerSparseFactorization{T,K}, b::AbstractMatrix, trans=false, adj=false
) where {T,K}
    lock(_matrix_lock) do
        _handle(f)
        size(b, 1) == size(f, trans ? 2 : 1) ||
            throw(DimensionMismatch("sparse factor and RHS rows must match"))
        LinearAlgebra.issuccess(f) || (
            if K === :cholesky
                throw(LinearAlgebra.PosDefException(f.info))
            else
                throw(LinearAlgebra.SingularException(f.info))
            end
        )
        K === :qr &&
            trans &&
            !_native_call(RustCrate.sparse_qr_has_components, f) &&
            throw(ArgumentError("transpose/adjoint sparse QR components are unavailable"))
        if K === :qr && _native_call(RustCrate.sparse_qr_needs_spectral_solve, f)
            oriented = trans ? (adj ? adjoint(f) : transpose(f)) : f
            return _sparse_qr_minimum_norm_solve(oriented, b)
        end
        P = promote_type(T, _sparse_input_type(eltype(b)))
        if P !== T
            base = FaerLinearOperator(f)
            action = trans ? (adj ? adjoint(base) : transpose(base)) : base
            try
                return action * b
            finally
                action === base || close(action)
                close(base)
            end
        end
        rhs = b isa FaerMatrix ? _coerce(b, T) : FaerMatrix{T}(b)
        dims = (size(f, trans ? 1 : 2), size(b, 2))
        args = (f, rhs, trans, adj)
        return if T === Float64
            _matrix_result(RustCrate.sparse_factor_solve, dims, args...)
        else
            _typed_result(T, RustCrate.sparse_factor_solve, dims, args...)
        end
    end
end
_sparse_solve(f::FaerSparseFactorization, b) = _sparse_factor_solve(f, b)
function _sparse_solve(f::_SparseTransposedFactor, b)
    return _sparse_factor_solve(
        parent(f), b, true, f isa LinearAlgebra.AdjointFactorization
    )
end
function LinearAlgebra.ishermitian(a::FaerSparseMatrix)
    lock(_matrix_lock) do
        return _native_call(RustCrate.sparse_is_selfadjoint, a, true)
    end
end
function LinearAlgebra.issymmetric(a::FaerSparseMatrix)
    lock(_matrix_lock) do
        return _native_call(RustCrate.sparse_is_selfadjoint, a, false)
    end
end
function _refactor_sparse!(
    f::FaerSparseFactorization{T},
    a::_SparseFactorInput{T};
    check=true,
    scratch=nothing,
    team=nothing,
) where {T}
    lock(_matrix_lock) do
        _handle(f)
        next = LinearAlgebra.factorize(f.plan, a; check, scratch, team)
        oldplan = f.plan
        oldsource = f.source
        f.handle = next.handle
        f.dims = next.dims
        f.info = next.info
        f.plan = next.plan
        f.source = next.source
        close(oldplan)
        close(oldsource)
        return f
    end
end
function LinearAlgebra.lu!(
    f::FaerSparseFactorization{T,:lu},
    a::_SparseFactorInput{T};
    check::Bool=true,
    scratch=nothing,
    team=nothing,
) where {T}
    return _refactor_sparse!(f, a; check, scratch, team)
end
"""
    qr!(F::FaerSparseFactorization{T,:qr}, A::FaerSparseMatrix{T}; scratch=nothing, team=nothing)

Replace the numeric QR revision using F's retained symbolic analysis and ordering.
A must match its stored pattern. Optional scratch/team use the same construction
kernel and requirements as `factorize(F.plan,A; scratch,team)`, for every shape
and algorithm. Failures preserve F. Snapshot and owned output allocation remain
separate; the update stages a new factor before replacing F's numeric revision.
"""
function LinearAlgebra.qr!(
    f::FaerSparseFactorization{T,:qr},
    a::_SparseFactorInput{T};
    scratch=nothing,
    team=nothing,
) where {T}
    return _refactor_sparse!(f, a; scratch, team)
end
function LinearAlgebra.cholesky!(
    f::FaerSparseFactorization{T,:cholesky},
    a::_SparseFactorInput{T};
    check::Bool=true,
    scratch=nothing,
    team=nothing,
) where {T}
    return _refactor_sparse!(f, a; check, scratch, team)
end

"""Return an independent copy of the permutation used by sparse Cholesky analysis."""
function FaerPermutation(p::FaerSparseSymbolic{:cholesky})
    return _permutation_result(size(p, 1), RustCrate.sparse_cholesky_permutation, p)
end
function FaerPermutation(f::FaerSparseFactorization{T,:cholesky}) where {T}
    lock(_matrix_lock) do
        _handle(f)
        return _permutation_result(
            size(f, 1), RustCrate.sparse_numeric_cholesky_permutation, f
        )
    end
end
function Base.getproperty(p::FaerSparseSymbolic{:cholesky}, name::Symbol)
    name === :p && return FaerPermutation(p)
    name === :factor_storage &&
        return Int(_native_call(RustCrate.sparse_cholesky_storage, p))
    if name === :algorithm
        return if _native_call(RustCrate.sparse_cholesky_algorithm, p) == 1
            :simplicial
        else
            :supernodal
        end
    end
    return getfield(p, name)
end
function Base.propertynames(p::FaerSparseSymbolic{:cholesky}, private::Bool=false)
    return if private
        (fieldnames(typeof(p))..., :p, :algorithm, :factor_storage)
    else
        (:dims, :uplo, :p, :algorithm, :factor_storage)
    end
end
function Base.getproperty(f::FaerSparseFactorization{T,:cholesky}, name::Symbol) where {T}
    name === :p && return FaerPermutation(f)
    name === :L && return _sparse_cholesky_component(f, false)
    name === :U && return _sparse_cholesky_component(f, true)
    if name === :factor_storage
        return lock(_matrix_lock) do
            _handle(f)
            return f.plan.factor_storage
        end
    end
    if name === :algorithm
        return lock(_matrix_lock) do
            _handle(f)
            return f.plan.algorithm
        end
    end
    return getfield(f, name)
end
function Base.propertynames(
    f::FaerSparseFactorization{T,:cholesky}, private::Bool=false
) where {T}
    return if private
        (fieldnames(typeof(f))..., :p, :L, :U, :algorithm, :factor_storage)
    else
        (:dims, :info, :p, :L, :U, :algorithm, :factor_storage)
    end
end

const _SparseCholesky = FaerSparseFactorization{<:Any,:cholesky}
function _check_sparse_cholesky(f::_SparseCholesky)
    _handle(f)
    LinearAlgebra.issuccess(f) || throw(LinearAlgebra.PosDefException(f.info))
    return nothing
end
function _sparse_cholesky_component(
    f::FaerSparseFactorization{T,:cholesky}, upper
) where {T}
    lock(_matrix_lock) do
        _check_sparse_cholesky(f)
        return _sparse_result(
            T, size(f), :csc, RustCrate.sparse_cholesky_component, f, upper
        )
    end
end
function _sparse_cholesky_diagonal(f::FaerSparseFactorization{T,:cholesky}) where {T}
    lock(_matrix_lock) do
        _check_sparse_cholesky(f)
        out = Vector{T}(undef, size(f, 1))
        GC.@preserve out _native_call(
            RustCrate.sparse_cholesky_diagonal,
            f,
            Ptr{UInt8}(pointer(out)),
            UInt(length(out)),
        )
        return out
    end
end
function FaerSparseMatrix(f::_SparseCholesky)
    lock(_matrix_lock) do
        _check_sparse_cholesky(f)
        inverse = invperm(FaerPermutation(f))
        return (f.L * f.U)[inverse, inverse]
    end
end
function FaerSparseMatrix{T}(f::_SparseCholesky) where {T<:_NativeScalar}
    return FaerSparseMatrix{T}(FaerSparseMatrix(f))
end
FaerMatrix(f::_SparseCholesky) = FaerMatrix(FaerSparseMatrix(f))
function FaerMatrix{T}(f::_SparseCholesky) where {T<:_NativeScalar}
    return FaerMatrix{T}(FaerSparseMatrix(f))
end
Base.Matrix(f::_SparseCholesky) = Matrix(FaerSparseMatrix(f))
Base.Matrix{T}(f::_SparseCholesky) where {T} = Matrix{T}(FaerSparseMatrix(f))
function SparseArrays.SparseMatrixCSC(f::_SparseCholesky)
    return SparseArrays.SparseMatrixCSC(FaerSparseMatrix(f))
end
SparseArrays.sparse(f::_SparseCholesky) = SparseArrays.SparseMatrixCSC(f)
function LinearAlgebra.logdet(f::_SparseCholesky)
    d = real.(_sparse_cholesky_diagonal(f))
    return 2sum(log, d; init=zero(eltype(d)))
end
LinearAlgebra.det(f::_SparseCholesky) = exp(LinearAlgebra.logdet(f))
function LinearAlgebra.logabsdet(f::_SparseCholesky)
    return (LinearAlgebra.logdet(f), one(_realtype(eltype(f))))
end

function Base.getproperty(f::FaerSparseFactorization{T,:lu}, name::Symbol) where {T}
    if name in (:L, :U, :p, :q)
        return lock(_matrix_lock) do
            _handle(f)
            LinearAlgebra.issuccess(f) || throw(LinearAlgebra.SingularException(f.info))
            name in (:p, :q) && return _permutation_result(
                size(f, 1), RustCrate.sparse_lu_permutation, f, name === :q
            )
            return _sparse_result(
                T, size(f), :csc, RustCrate.sparse_lu_component, f, name === :U
            )
        end
    end
    return getfield(f, name)
end
function Base.propertynames(
    f::FaerSparseFactorization{T,:lu}, private::Bool=false
) where {T}
    return if private
        (fieldnames(typeof(f))..., :L, :U, :p, :q)
    else
        (:dims, :info, :L, :U, :p, :q)
    end
end

function Base.getproperty(p::FaerSparseSymbolic{:qr}, name::Symbol)
    if name === :algorithm
        return if _native_call(RustCrate.sparse_qr_algorithm, p) == 1
            :simplicial
        else
            :supernodal
        end
    end
    return getfield(p, name)
end
function Base.propertynames(p::FaerSparseSymbolic{:qr}, private::Bool=false)
    return private ? (fieldnames(typeof(p))..., :algorithm) : (:dims, :uplo, :algorithm)
end
function Base.getproperty(f::FaerSparseFactorization{T,:qr}, name::Symbol) where {T}
    if name === :algorithm
        return lock(_matrix_lock) do
            _handle(f)
            return f.plan.algorithm
        end
    end
    if name in (:Q, :Qthin, :Qfull, :Qoperator, :R, :p, :pcol, :prow)
        return lock(_matrix_lock) do
            _handle(f)
            LinearAlgebra.issuccess(f) || throw(LinearAlgebra.SingularException(f.info))
            name === :Qoperator && return _sparse_qr_q_operator(f)
            # Q is stored in the original row order (SPQR's prow is identity here).
            name === :prow && return FaerPermutation(1:size(f, 1))
            name in (:p, :pcol) &&
                return _permutation_result(size(f, 2), RustCrate.sparse_qr_permutation, f)
            name === :R && return _sparse_result(
                T, (min(size(f)...), size(f, 2)), :csc, RustCrate.sparse_qr_r, f
            )
            dims =
                name === :Qfull ? (size(f, 1), size(f, 1)) : (size(f, 1), min(size(f)...))
            return if T === Float64
                _matrix_result(RustCrate.sparse_qr_q, dims, f, name === :Qfull)
            else
                _typed_result(T, RustCrate.sparse_qr_q, dims, f, name === :Qfull)
            end
        end
    end
    return getfield(f, name)
end
function Base.propertynames(
    f::FaerSparseFactorization{T,:qr}, private::Bool=false
) where {T}
    return if private
        (
            fieldnames(typeof(f))...,
            :Q,
            :Qthin,
            :Qfull,
            :Qoperator,
            :R,
            :p,
            :pcol,
            :prow,
            :algorithm,
        )
    else
        (:dims, :info, :Q, :Qthin, :Qfull, :Qoperator, :R, :p, :pcol, :prow, :algorithm)
    end
end
# Lazy m×m Q (SPQR's `F.Q`): an independent native snapshot of the saved
# Householder reflectors; applications never form dense Q.
function _sparse_qr_q_operator(f::FaerSparseFactorization{T,:qr}) where {T}
    h = RustCrate.OwnedLinearOperator()
    _native_call(RustCrate.operator_sparse_qr_q, h, f)
    m = size(f, 1)
    return FaerLinearOperator{T,typeof(h)}(h, (m, m), :factorization, false, false)
end
function reconstruct(f::FaerSparseFactorization{T,:qr}) where {T}
    lock(_matrix_lock) do
        _handle(f)
        return if T === Float64
            _matrix_result(RustCrate.sparse_qr_reconstruct, size(f), f)
        else
            _typed_result(T, RustCrate.sparse_qr_reconstruct, size(f), f)
        end
    end
end
FaerMatrix(f::FaerSparseFactorization{T,:qr}) where {T} = reconstruct(f)
function Base.Matrix(f::FaerSparseFactorization{T,:qr}) where {T}
    a = reconstruct(f)
    try
        return Matrix(a)
    finally
        close(a)
    end
end

function FaerMatrix{T}(f::FaerSparseFactorization{S,:qr}) where {T<:_NativeScalar,S}
    a = reconstruct(f)
    T === S && return a
    try
        return FaerMatrix{T}(a)
    finally
        close(a)
    end
end
function Base.Matrix{T}(f::FaerSparseFactorization{S,:qr}) where {T,S}
    a = reconstruct(f)
    try
        return Matrix{T}(a)
    finally
        close(a)
    end
end
