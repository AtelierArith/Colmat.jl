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
function _tridiagonalize_requirement(n, T, bs, team)
    info=zeros(UInt, 2)
    GC.@preserve info begin
        args=(UInt(n), _tag(T), UInt(bs), pointer(info))
        isnothing(team) ? _native_call(RustCrate.tridiagonalize_requirements, args...) :
        _native_call(RustCrate.tridiagonalize_team_requirements, team, args...)
    end
    return (bytes = Int(info[1]), alignment = Int(info[2]))
end
"""
    scratch_requirements(tridiagonalize, A; blocksize=0, team=nothing)
    scratch_requirements(tridiagonalize!, A; blocksize=0, team=nothing)

Query reusable workspace for Hermitian tridiagonal reduction and explicit Q
formation, including the canonical work matrix and compact coefficients. The
reduction/Q stages reuse one kernel region and the actual scoped team token.
Julia input conversion, returned Q/H and staged writeback allocate separately.
"""
function scratch_requirements(
    ::_TridiagonalReductionFunction,
    a::AbstractMatrix{<:Number};
    blocksize::Integer = 0,
    team = nothing,
)
    lock(_matrix_lock) do
        _iterative_resource_check(nothing, team)
        n, T, bs=_tridiagonalize_plan(a, blocksize)
        return _tridiagonalize_requirement(n, T, bs, team)
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
