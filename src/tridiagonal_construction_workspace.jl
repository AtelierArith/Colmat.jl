const _TridiagonalReductionFunction=Union{typeof(tridiagonalize),typeof(tridiagonalize!)}
function _tridiagonalize_plan(a, blocksize)
    _tridiagonal_open(a)
    _qz_validate_operand(a)
    n=LinearAlgebra.checksquare(a)
    T=_reflector_type(eltype(a))
    bs=Int(blocksize)
    0<=bs<=max(1, n-1) || throw(
        ArgumentError("blocksize must be zero (automatic) or between 1 and max(1,n-1)"),
    )
    return n, T, bs
end
function _tridiagonalize_requirement(n, T, bs, team, policy; compact = false)
    info=zeros(UInt, 2)
    GC.@preserve info begin
        args=(UInt(n), _tag(T), UInt(bs), policy.value, pointer(info))
        _native_call(
            (
                compact ? RustCrate.compact_tridiagonal_requirements :
                RustCrate.tridiagonalize_requirements
            ),
            args...,
            team,
        )
    end
    return (bytes = Int(info[1]), alignment = Int(info[2]))
end
"""
    scratch_requirements(tridiagonalize, A; compact=false, blocksize=0, team=nothing, par_threshold=FaerParThreshold())
    scratch_requirements(tridiagonalize!, A; compact=false, blocksize=0, team=nothing, par_threshold=FaerParThreshold())

Query reusable workspace for Hermitian tridiagonal reduction and explicit Q
formation, including the canonical work matrix and compact coefficients. The
reduction/Q stages reuse one kernel region and the actual scoped team token.
Julia input conversion, returned Q/H and staged writeback allocate separately.
With `compact=true`, the returned Q owns the packed work matrix, compact factors
and phases; scratch then covers only the reduction kernel. Use the same
`compact` option for query and computation.
Use the same `par_threshold` as computation. `FaerParThreshold(:never)` queries
sequential numerical kernels while still validating a supplied team.
"""
function scratch_requirements(
    ::_TridiagonalReductionFunction,
    a::AbstractMatrix{<:Number};
    blocksize::Integer = 0,
    team = nothing,
    par_threshold = FaerParThreshold(),
    compact::Bool = false,
)
    lock(_matrix_lock) do
        _iterative_resource_check(nothing, team)
        policy=_par_threshold(par_threshold)
        n, T, bs=_tridiagonalize_plan(a, blocksize)
        return _tridiagonalize_requirement(n, T, bs, team, policy; compact)
    end
end
function FaerScratch(
    f::_TridiagonalReductionFunction,
    a::AbstractMatrix{<:Number};
    kwargs...,
)
    req=scratch_requirements(f, a; kwargs...)
    return FaerScratch(req.bytes; alignment = req.alignment)
end
function reserve!(
    s::FaerScratch,
    f::_TridiagonalReductionFunction,
    a::AbstractMatrix{<:Number};
    kwargs...,
)
    req=scratch_requirements(f, a; kwargs...)
    return reserve!(s, req.bytes; alignment = req.alignment)
end
const _TridiagonalHessenbergFunction=Union{
    typeof(LinearAlgebra.hessenberg),
    typeof(LinearAlgebra.hessenberg!),
}
scratch_requirements(
    ::_TridiagonalHessenbergFunction,
    a::_TridiagonalSelfAdjoint;
    kwargs...,
) = scratch_requirements(tridiagonalize, a; kwargs...)
function FaerScratch(
    f::_TridiagonalHessenbergFunction,
    a::_TridiagonalSelfAdjoint;
    kwargs...,
)
    req=scratch_requirements(f, a; kwargs...)
    return FaerScratch(req.bytes; alignment = req.alignment)
end
function reserve!(
    s::FaerScratch,
    f::_TridiagonalHessenbergFunction,
    a::_TridiagonalSelfAdjoint;
    kwargs...,
)
    req=scratch_requirements(f, a; kwargs...)
    return reserve!(s, req.bytes; alignment = req.alignment)
end
