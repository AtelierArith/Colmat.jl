# Caller-owned construction scratch for saved QR inverse/pseudoinverse actions.
function _construction_options(scratch, team)
    isnothing(scratch) ||
        scratch isa FaerScratch ||
        throw(ArgumentError("scratch must be a FaerScratch"))
    isnothing(team) || team isa FaerTeam || throw(ArgumentError("team must be a FaerTeam"))
    isnothing(team) || _handle(team)
    isnothing(scratch) || _handle(scratch)
    return nothing
end
function _with_construction_scratch(f, req, scratch)
    work=isnothing(scratch) ? FaerScratch(req.bytes; alignment = req.alignment) : scratch
    try
        cap=scratch_capacity(work)
        cap.bytes>=req.bytes && cap.alignment>=req.alignment || throw(
            ArgumentError(
                "operator construction scratch capacity/alignment is insufficient; call reserve!(scratch, FaerLinearOperator, F)",
            ),
        )
        return f(work)
    finally
        isnothing(scratch) && close(work)
    end
end
function _qr_pseudoinverse_requirement(f, team)
    info=zeros(UInt, 2)
    args=(UInt(size(f, 1)), UInt(size(f, 2)), _tag(eltype(f)), info)
    GC.@preserve info begin
        _native_call(RustCrate.operator_qr_pseudoinverse_requirements, args..., team)
    end
    return (bytes = Int(info[1]), alignment = Int(info[2]))
end
function _sparse_qr_inverse_requirement(f, team)
    info=zeros(UInt, 2)
    GC.@preserve info begin
        _native_call(RustCrate.operator_sparse_qr_inverse_requirements, f, info, team)
    end
    return (bytes = Int(info[1]), alignment = Int(info[2]))
end
const _NoConstructionScratch = (bytes = 0, alignment = 1)

"""
    scratch_requirements(FaerLinearOperator, F::FaerFactorization{T,:qr}; atol=0, rtol=nothing, team=nothing)
    scratch_requirements(FaerLinearOperator, F::FaerSparseFactorization; team=nothing)

Query native construction scratch for `FaerLinearOperator(F; scratch, team)`.
Pass the same tolerances and team as the construction.
For dense (optionally pivoted) QR pseudoinverse/truncated actions this holds
the SVD of saved R: singular values, thin singular vectors and faer's SVD
workspace. Only the singular columns above the cutoff are copied into the
operator; Householder factors and the permutation are copied as operator state.
Square QR without tolerances keeps the nonsingular inverse action and needs
zero bytes. For sparse factors only wide QR needs scratch: dense R, its SVD and
the pseudoinverse reconstruction; the retained `pinv(R)`, the head factor
and the permutation are operator output. Sparse LU/Cholesky and square/tall QR
only copy factors (zero bytes). Rank-deficient sparse QR is routed to the
independent dense `pinv(F)`, which allocates its own temporaries; it reports
zero bytes and does not consume the scratch or team.
faer's matrix products inside the SVD may still allocate their own packing
buffers above small sizes (t4a-faer upstream-reports/gemm-001).
"""
function scratch_requirements(
    ::Type{<:FaerLinearOperator},
    f::_DenseQrFactor;
    atol::Real = 0,
    rtol = nothing,
    team = nothing,
)
    lock(_matrix_lock) do
        _handle(f)
        _spectral_tolerances(atol, isnothing(rtol) ? _default_spectral_rtol(f, atol) : rtol)
        _construction_options(nothing, team)
        size(f, 1)==size(f, 2) &&
            iszero(atol) &&
            isnothing(rtol) &&
            return _NoConstructionScratch
        return _qr_pseudoinverse_requirement(f, team)
    end
end
function scratch_requirements(
    ::Type{<:FaerLinearOperator},
    f::FaerSparseFactorization{T,K};
    team = nothing,
) where {T,K}
    K in (:lu, :cholesky, :qr) ||
        throw(ArgumentError("sparse inverse operator requires LU/QR or Cholesky"))
    lock(_matrix_lock) do
        _handle(f)
        _construction_options(nothing, team)
        K===:qr || return _NoConstructionScratch
        LinearAlgebra.issuccess(f) || throw(LinearAlgebra.SingularException(f.info))
        _native_call(RustCrate.sparse_qr_needs_spectral_solve, f) &&
            return _NoConstructionScratch
        return _sparse_qr_inverse_requirement(f, team)
    end
end
const _OperatorConstructionSource = Union{_DenseQrFactor,FaerSparseFactorization}
function FaerScratch(t::Type{<:FaerLinearOperator}, f::_OperatorConstructionSource; kwargs...)
    req=scratch_requirements(t, f; kwargs...)
    return FaerScratch(req.bytes; alignment = req.alignment)
end
function reserve!(
    s::FaerScratch,
    t::Type{<:FaerLinearOperator},
    f::_OperatorConstructionSource;
    kwargs...,
)
    req=scratch_requirements(t, f; kwargs...)
    return reserve!(s, req.bytes; alignment = req.alignment)
end
