# faer's dense self-adjoint factorization parameters (`LltParams`,
# `LdltParams`, `LbltParams`) and the per-call options that change the result
# (dynamic regularization, LBLᴴ pivoting strategy). The native state is read by
# every Colmat LLᴴ/LDLᴴ/LBLᴴ construction and its scratch queries.

const _cholesky_tuning_names = (
    :llt_recursion_threshold,
    :llt_block_size,
    :ldlt_recursion_threshold,
    :ldlt_block_size,
    :lblt_block_size,
    :lblt_par_threshold,
)

"""
    with_cholesky_tuning(f; llt_recursion_threshold=nothing, llt_block_size=nothing,
                         ldlt_recursion_threshold=nothing, ldlt_block_size=nothing,
                         lblt_block_size=nothing, lblt_par_threshold=nothing)

Run `f()` with faer's dense self-adjoint factorization parameters overridden
for every `cholesky`/`ldlt`/`bunchkaufman` construction in the scope: the
retained factors (owned, typed and borrowed input, `team=`), the staged and
direct in-place `cholesky!`/`ldlt!`/`bunchkaufman!` paths, and their scratch
queries, `FaerScratch` construction and `reserve!`, so query and execution
agree. `nothing` keeps faer's default for that field. These fields only change
blocking and parallel scheduling; the factorization is the same up to
rounding.

- `llt_recursion_threshold`, `llt_block_size`: `LltParams` of `cholesky`
  (faer's defaults 64 and 128). Must be positive.
- `ldlt_recursion_threshold`, `ldlt_block_size`: `LdltParams` of `ldlt`
  (faer's defaults 64 and 128). Must be positive.
- `lblt_block_size`: `LbltParams.block_size` of `bunchkaufman` (faer's default
  64; zero, one or at least the order select the unblocked kernel). The
  direct path's workspace grows with it.
- `lblt_par_threshold`: `LbltParams.par_threshold` as a
  [`FaerParThreshold`](@ref); faer reads it only for `pivoting=:full`.

Regularization and the LBLᴴ pivoting strategy change the factored matrix or
the permutation, so they are per-call keywords of `cholesky`, `ldlt` and
`bunchkaufman` instead. Pivoted `cholesky(A, RowMaximum())` keeps its own
`block_size` keyword. The scope holds Colmat's native-call lock, so other tasks
wait instead of seeing the overrides; `f` must call Colmat on the current task.
Previous settings, including those of an enclosing scope, are restored even if
`f` throws.
"""
function with_cholesky_tuning(
    f;
    llt_recursion_threshold = nothing,
    llt_block_size = nothing,
    ldlt_recursion_threshold = nothing,
    ldlt_block_size = nothing,
    lblt_block_size = nothing,
    lblt_par_threshold = nothing,
)
    positive=(llt_recursion_threshold, llt_block_size, ldlt_recursion_threshold, ldlt_block_size)
    for (name, x) in zip(_cholesky_tuning_names, positive)
        x isa Union{Nothing,Integer} && (isnothing(x) || 1<=x<=typemax(UInt)) || throw(
            ArgumentError("$name must be nothing or an integer between one and typemax(UInt)"),
        )
    end
    lblt_block_size isa Union{Nothing,Integer} &&
        (isnothing(lblt_block_size) || 0<=lblt_block_size<=typemax(UInt)) ||
        throw(ArgumentError("lblt_block_size must be nothing or a nonnegative integer"))
    lblt_par_threshold isa Union{Nothing,FaerParThreshold} ||
        throw(ArgumentError("lblt_par_threshold must be a FaerParThreshold or nothing"))
    settings=(
        positive...,
        lblt_block_size,
        isnothing(lblt_par_threshold) ? nothing : lblt_par_threshold.value,
    )
    values=UInt[isnothing(x) ? 0 : x for x in settings]
    flags=UInt8[isnothing(x) ? 0 : 1 for x in settings]
    lock(_matrix_lock) do
        old_values=zeros(UInt, 6)
        old_flags=zeros(UInt8, 6)
        _native_call(RustCrate.selfadjoint_tuning_state, old_values, old_flags)
        _native_call(RustCrate.set_selfadjoint_tuning, values, flags)
        try
            return f()
        finally
            _native_call(RustCrate.set_selfadjoint_tuning, old_values, old_flags)
        end
    end
end

# Per-call options: `pivoting` is the native LBLᴴ strategy code (0 keeps the
# kind's default), `delta`/`epsilon` faer's dynamic regularization (both zero:
# off), `signs` LDLᴴ's expected pivot signs or `nothing`.
const _NoSelfAdjointOptions = (pivoting = Int32(0), delta = 0.0, epsilon = 0.0, signs = nothing)

const _lblt_strategies =
    (partial = 1, partial_diag = 2, rook = 3, rook_diag = 4, full = 5)

function _regularization_options(n, delta, epsilon, signs)
    (delta isa Real && epsilon isa Real) ||
        throw(ArgumentError("regularization_delta and regularization_epsilon must be real"))
    (isfinite(delta) && isfinite(epsilon) && delta>=0 && epsilon>=0) || throw(
        ArgumentError(
            "regularization_delta and regularization_epsilon must be finite and nonnegative",
        ),
    )
    active=delta>0 || epsilon>0
    # faer silently skips the regularization unless both are positive.
    active && !(delta>0 && epsilon>0) && throw(
        ArgumentError(
            "dynamic regularization needs both regularization_delta and " *
            "regularization_epsilon to be positive",
        ),
    )
    if !isnothing(signs)
        active || throw(
            ArgumentError(
                "regularization_signs requires regularization_delta and regularization_epsilon",
            ),
        )
        signs isa AbstractVector{<:Integer} ||
            throw(ArgumentError("regularization_signs must be a vector of -1, 0 or 1"))
        length(signs)==n ||
            throw(DimensionMismatch("regularization_signs must have one entry per row"))
        all(s -> -1<=s<=1, signs) ||
            throw(ArgumentError("regularization_signs entries must be -1, 0 or 1"))
        signs=Int8.(signs)
    end
    return (pivoting = Int32(0), delta = Float64(delta), epsilon = Float64(epsilon), signs)
end
_cholesky_options(n, delta, epsilon) = _regularization_options(n, delta, epsilon, nothing)
_ldlt_options(n, delta, epsilon, signs) = _regularization_options(n, delta, epsilon, signs)
# The strategy code and Julia's `F.rook` for `bunchkaufman(A, rook; pivoting)`.
function _lblt_options(rook::Bool, pivoting)
    isnothing(pivoting) && return _NoSelfAdjointOptions, rook
    pivoting isa Symbol && haskey(_lblt_strategies, pivoting) || throw(
        ArgumentError(
            "pivoting must be nothing, :partial, :partial_diag, :rook, :rook_diag or :full",
        ),
    )
    rook &&
        !(pivoting in (:rook, :rook_diag)) &&
        throw(ArgumentError("rook=true conflicts with pivoting=:$pivoting"))
    options=(_NoSelfAdjointOptions..., pivoting = Int32(_lblt_strategies[pivoting]))
    return options, pivoting in (:rook, :rook_diag)
end

_default_options(options) =
    options.pivoting==0 && options.delta==0 && options.epsilon==0 && isnothing(options.signs)
# Install `options` natively for `f()` (one factorization or scratch query)
# under the native-call lock, and clear them afterwards.
function _with_selfadjoint_options(f, options)
    _default_options(options) && return f()
    lock(_matrix_lock) do
        signs=isnothing(options.signs) ? UInt8[] : reinterpret(UInt8, options.signs)
        _native_call(
            RustCrate.set_selfadjoint_call_options,
            options.pivoting,
            options.delta,
            options.epsilon,
            collect(signs),
            !isnothing(options.signs),
        )
        try
            return f()
        finally
            _native_call(RustCrate.set_selfadjoint_call_options, Int32(0), 0.0, 0.0, UInt8[], false)
        end
    end
end
# faer's `dynamic_regularization_count` of the construction that just ran.
_regularization_count(K, info) =
    K in (:cholesky, :ldlt) && info==0 ?
    Int(_native_call(RustCrate.selfadjoint_regularization_count)) : 0
# Keyword forms shared by the typed, borrowed, wrapper and destructive methods;
# an unsupported keyword is a MethodError here.
_cholesky_kw(n; regularization_delta::Real = 0, regularization_epsilon::Real = 0) =
    _cholesky_options(n, regularization_delta, regularization_epsilon)
_ldlt_kw(
    n;
    regularization_delta::Real = 0,
    regularization_epsilon::Real = 0,
    regularization_signs = nothing,
) = _ldlt_options(n, regularization_delta, regularization_epsilon, regularization_signs)
_lblt_kw(rook::Bool; pivoting = nothing) = _lblt_options(rook, pivoting)
