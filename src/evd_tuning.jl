const _evd_tuning_depth=Ref(0)
# Untuned calls without resources use the high-level native path; inside a
# `with_evd_tuning` scope they use the parameterized workspace driver.
_evd_default_path(scratch, team) =
    isnothing(scratch) && isnothing(team) && iszero(_evd_tuning_depth[])
# The general driver's multishift QR step also follows `with_schur_tuning`.
_general_evd_default_path(scratch, team) =
    _evd_default_path(scratch, team) && iszero(_schur_tuning_depth[])

_evd_tuning_value(x::Nothing) = (UInt(0), UInt8(0))
_evd_tuning_value(x::FaerParThreshold) = (x.value, UInt8(1))
function _evd_tuning_value(x::Integer)
    x>0 || throw(ArgumentError("EVD thresholds must be positive"))
    return (UInt(x), UInt8(1))
end
_evd_tuning_value(x) =
    throw(ArgumentError("EVD tuning values must be integers, FaerParThreshold or nothing"))

"""
    with_evd_tuning(f; hessenberg_par_threshold=nothing, hessenberg_blocking_threshold=nothing,
                    evd_recursion_threshold=nothing, self_adjoint_recursion_threshold=nothing,
                    tridiagonal_par_threshold=nothing)

Run `f()` with the parameters inside faer's EVD drivers overridden for every
general and self-adjoint `eigen`/`eigvals` call in the scope, including their
scratch queries, `FaerScratch` and `reserve!`. `nothing` keeps faer's default.

- General EVD (`EvdParams`): `hessenberg_par_threshold::FaerParThreshold`,
  `hessenberg_blocking_threshold`, and `evd_recursion_threshold` for the
  eigenvectors-from-Schur step; its multishift QR step follows
  [`with_schur_tuning`](@ref), which also selects the workspace driver below.
- Self-adjoint EVD (`SelfAdjointEvdParams`): `tridiagonal_par_threshold::FaerParThreshold`
  and `self_adjoint_recursion_threshold` for divide and conquer.

Inside the scope, calls without `scratch`/`team` run on the parameterized
workspace driver instead of the high-level one. Like `with_schur_tuning`, the
scope holds Colmat's native-call lock, `f` must call Colmat on the current
task, and previous settings return even if `f` throws. Destructive
`eigen!`/`eigvals!` return the same tuned results; the general form's
written-back Schur form is computed separately with the scope's Hessenberg
settings (`evd_recursion_threshold` affects only returned eigenvectors).
"""
function with_evd_tuning(
    f;
    hessenberg_par_threshold = nothing,
    hessenberg_blocking_threshold = nothing,
    evd_recursion_threshold = nothing,
    self_adjoint_recursion_threshold = nothing,
    tridiagonal_par_threshold = nothing,
)
    for x in (hessenberg_par_threshold, tridiagonal_par_threshold)
        x isa Union{Nothing,FaerParThreshold} ||
            throw(ArgumentError("parallel thresholds must be FaerParThreshold or nothing"))
    end
    hessenberg_blocking_threshold isa Integer &&
        hessenberg_blocking_threshold<0 &&
        throw(ArgumentError("hessenberg_blocking_threshold must be nonnegative"))
    blocking=hessenberg_blocking_threshold isa Integer ?
             (UInt(hessenberg_blocking_threshold), UInt8(1)) :
             _evd_tuning_value(hessenberg_blocking_threshold)
    pairs=(
        _evd_tuning_value(hessenberg_par_threshold),
        blocking,
        _evd_tuning_value(evd_recursion_threshold),
        _evd_tuning_value(self_adjoint_recursion_threshold),
        _evd_tuning_value(tridiagonal_par_threshold),
    )
    values=UInt[first(p) for p in pairs]
    flags=UInt8[last(p) for p in pairs]
    lock(_matrix_lock) do
        old_values=zeros(UInt, 5)
        old_flags=zeros(UInt8, 5)
        _native_call(RustCrate.evd_tuning_state, old_values, old_flags)
        _native_call(RustCrate.set_evd_tuning, values, flags)
        _evd_tuning_depth[]+=1
        try
            return f()
        finally
            _evd_tuning_depth[]-=1
            _native_call(RustCrate.set_evd_tuning, old_values, old_flags)
        end
    end
end
