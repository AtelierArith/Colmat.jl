const _OrderingFunction=Union{typeof(amd),typeof(colamd)}
function _with_ordering_pattern(f, a)
    pattern=_ordering_pattern(a)
    try
        return f(pattern)
    finally
        pattern===a || close(pattern)
    end
end
function _ordering_requirements(pattern, shape, symmetric)
    info=zeros(UInt, 2)
    GC.@preserve info _native_call(
        RustCrate.ordering_requirements,
        pattern,
        UInt(shape[1]),
        UInt(shape[2]),
        symmetric,
        pointer(info),
    )
    return (bytes = Int(info[1]), alignment = Int(info[2]))
end
"""
    scratch_requirements(amd, A; dense=10.0, aggressive=true, statistics=false)
    scratch_requirements(colamd, A; dense_row=0.5, dense_col=0.5, aggressive=true)

Return `(bytes, alignment)` for the sequential AMD/COLAMD kernel on A's stored
pattern. Includes stored zeros and supports CSC/CSR, symbolic and numeric owners,
Julia CSC, and the same sparse wrappers as `amd`/`colamd`. Pattern conversion,
query buffers and the returned owning permutation allocate outside scratch.
Controls do not alter the kernel requirement. Inputs must be open and AMD square.
"""
function scratch_requirements(
    ::typeof(amd),
    a::_OrderingInput;
    dense::Real = 10.0,
    aggressive::Bool = true,
    statistics::Bool = false,
)
    isfinite(Float64(dense)) || throw(ArgumentError("dense must be finite"))
    size(a, 1)==size(a, 2) ||
        throw(DimensionMismatch("AMD requires a square sparse pattern"))
    lock(_matrix_lock) do
        _with_ordering_pattern(a) do pattern
            _ordering_requirements(pattern, size(a), true)
        end
    end
end
function scratch_requirements(
    ::typeof(colamd),
    a::_OrderingInput;
    dense_row::Real = 0.5,
    dense_col::Real = 0.5,
    aggressive::Bool = true,
)
    isfinite(Float64(dense_row))&&isfinite(Float64(dense_col)) ||
        throw(ArgumentError("COLAMD thresholds must be finite"))
    lock(_matrix_lock) do
        _with_ordering_pattern(a) do pattern
            _ordering_requirements(pattern, size(a), false)
        end
    end
end
function FaerScratch(fun::_OrderingFunction, a::_OrderingInput; kwargs...)
    req=scratch_requirements(fun, a; kwargs...)
    return FaerScratch(req.bytes; alignment = req.alignment)
end
function reserve!(s::FaerScratch, fun::_OrderingFunction, a::_OrderingInput; kwargs...)
    req=scratch_requirements(fun, a; kwargs...)
    return reserve!(s, req.bytes; alignment = req.alignment)
end
function _ordering_options(a, symmetric, row, col, aggressive, statistics, scratch)
    isnothing(scratch)||scratch isa FaerScratch ||
        throw(ArgumentError("scratch must be a FaerScratch"))
    lock(_matrix_lock) do
        _with_ordering_pattern(a) do pattern
            req=_ordering_requirements(pattern, size(a), symmetric)
            work=isnothing(scratch) ? FaerScratch(req.bytes; alignment = req.alignment) :
                 scratch
            try
                cap=scratch_capacity(work)
                cap.bytes>=req.bytes&&cap.alignment>=req.alignment || throw(
                    ArgumentError(
                        "scratch capacity/alignment is insufficient; call reserve!(scratch, amd/colamd, A)",
                    ),
                )
                stats=zeros(3)
                p=GC.@preserve stats _permutation_result(
                    size(a, 2),
                    RustCrate.sparse_ordering,
                    pattern,
                    work,
                    UInt(size(a, 1)),
                    UInt(size(a, 2)),
                    symmetric,
                    row,
                    col,
                    aggressive,
                    pointer(stats),
                )
                return statistics ?
                       (
                    permutation = p,
                    n_div = stats[1],
                    n_mult_subs_ldl = stats[2],
                    n_mult_subs_lu = stats[3],
                ) : p
            finally
                isnothing(scratch) && close(work)
            end
        end
    end
end
const _SymbolicOrderingInput=Union{FaerSparsity,FaerSparseMatrix}
"""
    scratch_requirements(FaerSparseSymbolic, A; kind=:lu, uplo=:L, ordering=:default, algorithm=:auto, dense=10.0, dense_row=0.5, dense_col=0.5, aggressive=true)

Return `(bytes, alignment)` for the fill-reducing ordering workspace used inside
`FaerSparseSymbolic(A; scratch=s, kwargs...)`, with the same keywords. A is a
`FaerSparsity` or `FaerSparseMatrix` (CSC/CSR, stored zeros included).
Cholesky `:default`/`:amd` runs faer's AMD (same input and controls as faer's
symbolic analysis) in the caller buffer, plus exact factor column counts for
`algorithm=:auto` so the simplicial/supernodal choice matches the no-scratch
path. Wide QR default COLAMD and explicit simplicial LU/QR `ordering=:colamd`
use the COLAMD kernel requirement. Identity/custom orderings and square/tall
auto/supernodal LU/QR return zero bytes: the latter's COLAMD runs inside faer's
allocating symbolic API, which accepts no caller stack. The kernels are
sequential. Pattern conversion to CSC, faer's post-ordering symbolic buffer,
the retained permutation and the symbolic structure allocate separately.
"""
function scratch_requirements(
    ::Type{<:FaerSparseSymbolic},
    a::_SymbolicOrderingInput;
    kwargs...,
)
    lock(_matrix_lock) do
        _with_ordering_pattern(a) do pattern
            o=_sparse_symbolic_options(pattern; kwargs...)
            return _sparse_symbolic_requirements(pattern, o)
        end
    end
end
function FaerScratch(t::Type{<:FaerSparseSymbolic}, a::_SymbolicOrderingInput; kwargs...)
    req=scratch_requirements(t, a; kwargs...)
    return FaerScratch(req.bytes; alignment = req.alignment)
end
function reserve!(
    s::FaerScratch,
    t::Type{<:FaerSparseSymbolic},
    a::_SymbolicOrderingInput;
    kwargs...,
)
    req=scratch_requirements(t, a; kwargs...)
    return reserve!(s, req.bytes; alignment = req.alignment)
end
