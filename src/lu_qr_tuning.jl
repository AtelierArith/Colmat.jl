# faer's dense LU and QR factorization parameters (`PartialPivLuParams`,
# `FullPivLuParams`, `QrParams`, `ColPivQrParams`). The native state is read by
# every Colmat LU/QR construction and its scratch queries. The two scopes own
# disjoint fields of one native state, so they nest in either order.

const _lu_qr_tuning_len = 5

# Override the native fields at `indices` with `settings` (`nothing` keeps
# faer's default) for `f()`, keeping the other fields of an enclosing scope.
function _with_lu_qr_tuning(f, indices, settings)
    lock(_matrix_lock) do
        old_values = zeros(UInt, _lu_qr_tuning_len)
        old_flags = zeros(UInt8, _lu_qr_tuning_len)
        _native_call(RustCrate.lu_qr_tuning_state, old_values, old_flags)
        values = copy(old_values)
        flags = copy(old_flags)
        for (i, x) in zip(indices, settings)
            values[i] = isnothing(x) ? 0 : x
            flags[i] = isnothing(x) ? 0 : 1
        end
        _native_call(RustCrate.set_lu_qr_tuning, values, flags)
        try
            return f()
        finally
            _native_call(RustCrate.set_lu_qr_tuning, old_values, old_flags)
        end
    end
end
function _par_threshold_value(name, x)
    return if x isa Union{Nothing,FaerParThreshold}
        (isnothing(x) ? nothing : x.value)
    else
        throw(ArgumentError("$name must be a FaerParThreshold or nothing"))
    end
end

"""
    with_lu_tuning(f; partial_recursion_threshold=nothing, partial_par_threshold=nothing,
                   full_par_threshold=nothing)

Run `f()` with faer's dense LU parameters overridden for every `lu` and
`full_piv_lu` construction in the scope: the retained factors (owned, typed and
borrowed input, `team=`), the staged and direct in-place `lu!`/`full_piv_lu!`
paths (and `inv!(A)`), the promoted refactorizations of mixed-scalar solves,
and their scratch queries, `FaerScratch` construction and `reserve!`, so query and
execution agree. `nothing` keeps faer's default. These fields only change
recursion and parallel scheduling; the factorization is the same up to
rounding.

- `partial_recursion_threshold`: `PartialPivLuParams.recursion_threshold`, the
  column count below which partial-pivot LU runs its unblocked kernel (faer's
  default 8 for `Float64`, 16 otherwise). Must be positive.
- `partial_par_threshold`: `PartialPivLuParams.par_threshold` as a
  [`FaerParThreshold`](@ref) (trailing update and row-swap regions).
- `full_par_threshold`: `FullPivLuParams.par_threshold` (a fixed floor counts
  elements of the trailing block).

faer's rank-truncation fields (`max_rank`, `abs_tol`, `rel_tol`) stop the
factorization early, and faer's solve, inverse and reconstruct routines must
not be given such a factor, so they are not exposed. `PartialPivLuParams.
block_size` is not read by faer's kernel at the pinned revision. LU used
internally by other Colmat operations (`A \\ B` without a saved factor,
`det(A)`, QZ/Schur, sparse factorizations) keeps faer's defaults. The scope holds Colmat's native-call
lock, so other tasks wait instead of seeing the overrides; `f` must call Colmat
on the current task. Previous settings, including those of an enclosing scope
(and the QR fields of an enclosing [`with_qr_tuning`](@ref)), are restored even
if `f` throws.
"""
function with_lu_tuning(
    f;
    partial_recursion_threshold=nothing,
    partial_par_threshold=nothing,
    full_par_threshold=nothing,
)
    r = partial_recursion_threshold
    r isa Union{Nothing,Integer} && (isnothing(r) || 1 <= r <= typemax(UInt)) || throw(
        ArgumentError(
            "partial_recursion_threshold must be nothing or an integer between one and typemax(UInt)",
        ),
    )
    settings = (
        r,
        _par_threshold_value("partial_par_threshold", partial_par_threshold),
        _par_threshold_value("full_par_threshold", full_par_threshold),
    )
    return _with_lu_qr_tuning(f, (1, 2, 3), settings)
end

"""
    with_qr_tuning(f; blocking_threshold=nothing, pivoted_par_threshold=nothing)

Run `f()` with faer's dense QR parameters overridden for every `qr` and
`qr(A, ColumnNorm())` construction in the scope: the retained factors (owned,
typed and borrowed input, `team=`), the staged and direct in-place `qr!` paths,
and their scratch queries, `FaerScratch` construction and `reserve!`. `nothing` keeps faer's default.
These fields only change blocking and parallel scheduling; the factorization
is the same up to rounding, and the block Householder coefficients keep faer's
recommended block size.

- `blocking_threshold`: `QrParams.blocking_threshold` of unpivoted QR, the
  `m*n` size below which a panel is factored without the blocked recursion
  (faer's default `48*48`). Any nonnegative integer.
- `pivoted_par_threshold`: `ColPivQrParams.par_threshold` of column-pivoted QR
  as a [`FaerParThreshold`](@ref) (`:never` keeps the whole call sequential).

`ColPivQrParams.blocking_threshold` is not read by faer's column-pivoted
kernel at the pinned revision. [`with_svd_tuning`](@ref)'s `qr_blocking` is
the separate `QrParams` of the SVD's internal QR, which this scope does not
change (nor does `with_svd_tuning` change these factorizations). The scope
holds Colmat's native-call lock; previous settings, including those of an
enclosing scope (and the LU fields of an enclosing [`with_lu_tuning`](@ref)),
are restored even if `f` throws.
"""
function with_qr_tuning(f; blocking_threshold=nothing, pivoted_par_threshold=nothing)
    b = blocking_threshold
    b isa Union{Nothing,Integer} && (isnothing(b) || 0 <= b <= typemax(UInt)) ||
        throw(ArgumentError("blocking_threshold must be nothing or a nonnegative integer"))
    settings = (b, _par_threshold_value("pivoted_par_threshold", pivoted_par_threshold))
    return _with_lu_qr_tuning(f, (4, 5), settings)
end
