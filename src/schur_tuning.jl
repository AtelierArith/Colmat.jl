const _schur_tuning_depth=Ref(0)
"""
    with_schur_tuning(f; shift_count=0, deflation_window=0, blocking_threshold=0,
                      nibble_threshold=0, shift_chase_cost=0,
                      generalized_hessenberg_block_size=0,
                      generalized_hessenberg_blocking_threshold=0)

Run `f()` with faer's multishift QR (`SchurParams`), QZ
(`GeneralizedSchurParams`) and generalized Hessenberg
(`GeneralizedHessenbergParams`) controls overridden for every Schur and QZ call
in the scope: `schur`/`schur!` of matrices, saved Hessenberg and compact
Hessenberg factors, `hessenberg(A, B)`/`hessenberg!(A, B)`,
`schur(A, B)`/`schur!(A, B)`, the multishift QR step of the general
`eigen`/`eigvals`/`eigen!`/`eigvals!(A)` driver, and their scratch queries,
`FaerScratch` construction and `reserve!`, so query and execution agree. Zero
keeps faer's default for that control.

- `shift_count`: shifts per sweep (even, at least two, capped by the active block).
- `deflation_window`: aggressive early deflation window (at least two).
- `blocking_threshold`: size below which the unblocked code is used.
- `nibble_threshold`: percentage (0–100) of the deflation window that must
  converge to skip a sweep.
- `shift_chase_cost`: QZ only; relative cost of the near-diagonal shift chase
  to the sweep's matrix products.
- `generalized_hessenberg_block_size`: QZ construction, `hessenberg(A, B)` and
  the general `eigen`/`eigvals(A, B)` driver; block size of the generalized Hessenberg reduction (faer's default is
  32; one, or at least the order, selects the unblocked kernel). Its workspace
  grows with the block size.
- `generalized_hessenberg_blocking_threshold`: the matching
  `GeneralizedHessenbergParams.blocking_threshold`, forwarded as is; the
  pinned faer's reduction does not read it.

Inside the scope, general `eigen`/`eigvals`/`eigen!`/`eigvals!(A)` calls
without `scratch`/`team` use the parameterized workspace driver, as in
[`with_evd_tuning`](@ref). The general generalized `eigen`/`eigvals(A, B)`
driver receives the generalized Hessenberg and QZ controls through
`GevdParams`; its query also reserves the tuned Hessenberg step, which faer's
`gevd_scratch` sizes with the default block size.

faer takes the shift count, deflation window and shift-chase cost as
`extern "C"` function pointers, so overrides are stored natively for the
scope. The scope holds Colmat's native-call lock, so other tasks wait instead
of seeing the overrides; `f` must call Colmat on the current task. Previous
settings, including those of an enclosing scope, are restored even if `f`
throws.
"""
function with_schur_tuning(
    f;
    shift_count::Integer = 0,
    deflation_window::Integer = 0,
    blocking_threshold::Integer = 0,
    nibble_threshold::Integer = 0,
    shift_chase_cost::Integer = 0,
    generalized_hessenberg_block_size::Integer = 0,
    generalized_hessenberg_blocking_threshold::Integer = 0,
)
    values=(
        shift_count,
        deflation_window,
        blocking_threshold,
        nibble_threshold,
        shift_chase_cost,
        generalized_hessenberg_block_size,
        generalized_hessenberg_blocking_threshold,
    )
    all(>=(0), values) || throw(ArgumentError("Schur tuning values must be nonnegative"))
    shift_count==1 &&
        throw(ArgumentError("shift_count must be zero (default) or at least two"))
    deflation_window==1 &&
        throw(ArgumentError("deflation_window must be zero (default) or at least two"))
    nibble_threshold<=100 ||
        throw(ArgumentError("nibble_threshold is a percentage between 0 and 100"))
    lock(_matrix_lock) do
        previous=zeros(UInt, 7)
        _native_call(RustCrate.schur_tuning_state, previous)
        _native_call(RustCrate.set_schur_tuning, map(UInt, values)...)
        _schur_tuning_depth[]+=1
        try
            return f()
        finally
            _schur_tuning_depth[]-=1
            _native_call(RustCrate.set_schur_tuning, previous...)
        end
    end
end
