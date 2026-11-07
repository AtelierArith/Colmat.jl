const _svd_tuning_depth=Ref(0)
_svd_default_path(scratch, team) =
    isnothing(scratch) && isnothing(team) && iszero(_svd_tuning_depth[])
function _svd_state()
    v=zeros(UInt, 3)
    r=zeros(Float64, 1)
    f=zeros(UInt8, 4)
    _native_call(RustCrate.svd_tuning_state, v, r, f)
    return v, r, f
end
function _svd_scope(f, recursion, ratio, par, qr_blocking = nothing)
    recursion isa Union{Nothing,Integer} ||
        throw(ArgumentError("recursion_threshold must be an integer or nothing"))
    ratio isa Union{Nothing,Real} ||
        throw(ArgumentError("qr_ratio_threshold must be a real number or nothing"))
    par isa Union{Nothing,FaerParThreshold} ||
        throw(ArgumentError("bidiag_par_threshold must be a FaerParThreshold or nothing"))
    qr_blocking isa Union{Nothing,Integer} ||
        throw(ArgumentError("qr_blocking_threshold must be an integer or nothing"))
    isnothing(qr_blocking) ||
        qr_blocking>=0 ||
        throw(ArgumentError("qr_blocking_threshold must be nonnegative"))
    isnothing(recursion) ||
        recursion>=2 ||
        throw(ArgumentError("recursion_threshold must be at least two"))
    isnothing(ratio) ||
        (isfinite(ratio) && ratio>=1) ||
        throw(ArgumentError("qr_ratio_threshold must be finite and at least one"))
    lock(_matrix_lock) do
        old=_svd_state()
        # Unspecified controls keep the enclosing scope's values.
        v, r, flags=deepcopy(old)
        isnothing(recursion) || (v[1] = UInt(min(recursion, typemax(Int))); flags[1] = 1)
        isnothing(ratio) || (r[1] = Float64(ratio); flags[2] = 1)
        isnothing(par) || (v[2] = par.value; flags[3] = 1)
        isnothing(qr_blocking) ||
            (v[3] = UInt(min(qr_blocking, typemax(Int))); flags[4] = 1)
        _native_call(RustCrate.set_svd_tuning, v[1], r[1], v[2], v[3], flags)
        _svd_tuning_depth[]+=1
        try
            return f()
        finally
            _svd_tuning_depth[]-=1
            _native_call(
                RustCrate.set_svd_tuning,
                old[1][1],
                old[2][1],
                old[1][2],
                old[1][3],
                old[3],
            )
        end
    end
end
"""
    with_svd_tuning(f; recursion_threshold=nothing, qr_ratio_threshold=nothing,
                    bidiag_par_threshold=nothing, qr_blocking_threshold=nothing)

Run `f()` with faer's `SvdParams` overridden for every `svd`/`svdvals`/
`svd!`/`svdvals!` of native matrices in the scope, including scratch queries,
`FaerScratch` and `reserve!`. `nothing` keeps the enclosing scope's (or faer's
default) value.

- `recursion_threshold`: bidiagonal sizes below it use the implicit QR
  algorithm, larger ones divide and conquer (faer's default is 128).
- `qr_ratio_threshold`: aspect ratio above which a QR factorization precedes
  bidiagonalization (at least one).
- `bidiag_par_threshold::FaerParThreshold`: bidiagonalization parallel floor.
- `qr_blocking_threshold`: `QrParams.blocking_threshold` of that preceding QR
  (nonnegative; while `m*n` is below it the QR factors its panels without
  sub-blocking or look-ahead, faer's default is 48*48). faer's QR workspace does not depend on it, so
  it changes execution but not scratch queries.

Inside the scope, calls without `scratch`/`team` use the parameterized
workspace driver. The scope holds Colmat's native-call lock; `f` must call
Colmat on the current task; previous settings return even if `f` throws.
"""
with_svd_tuning(
    f;
    recursion_threshold = nothing,
    qr_ratio_threshold = nothing,
    bidiag_par_threshold = nothing,
    qr_blocking_threshold = nothing,
) = _svd_scope(
    f,
    recursion_threshold,
    qr_ratio_threshold,
    bidiag_par_threshold,
    qr_blocking_threshold,
)

# `alg` follows LinearAlgebra: DivideAndConquer() is faer's default driver and
# QRIteration() forces the bidiagonal implicit QR algorithm at every size.
_svd_alg(f, ::Nothing) = f()
_svd_alg(f, ::LinearAlgebra.DivideAndConquer) = f()
_svd_alg(f, ::LinearAlgebra.QRIteration) = _svd_scope(f, typemax(Int), nothing, nothing)
_svd_alg(f, alg) =
    throw(ArgumentError("alg must be LinearAlgebra.DivideAndConquer() or QRIteration()"))
