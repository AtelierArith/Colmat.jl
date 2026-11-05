const _BidiagonalReductionFunction=Union{typeof(bidiagonalize),typeof(bidiagonalize!)}
function _bidiagonalize_plan(a, blocksize)
    _tridiagonal_open(a)
    _qz_validate_operand(a)
    m, n=size(a)
    T=_reflector_type(eltype(a))
    bs=Int(blocksize)
    0<=bs<=max(1, min(m, n)) || throw(
        ArgumentError(
            "blocksize must be zero (automatic) or between 1 and max(1,min(size(A)...))",
        ),
    )
    return m, n, T, bs
end
function _bidiagonalize_requirement(m, n, T, full, bs, team)
    info=zeros(UInt, 2)
    GC.@preserve info begin
        args=(UInt(m), UInt(n), _tag(T), full, UInt(bs), pointer(info))
        isnothing(team) ? _native_call(RustCrate.bidiagonalize_requirements, args...) :
        _native_call(RustCrate.bidiagonalize_team_requirements, team, args...)
    end
    return (bytes = Int(info[1]), alignment = Int(info[2]))
end
"""
    scratch_requirements(bidiagonalize, A; full=false, blocksize=0, team=nothing)
    scratch_requirements(bidiagonalize!, A; full=false, blocksize=0, team=nothing)

Query reusable bidiagonal reduction and explicit U/V workspace: canonical work
matrix, compact coefficients and a region reused by reduction and both basis
applications. Query and kernels use the actual scoped team shape/token. Input
conversion, returned owners and staged bang writeback allocate separately.
"""
function scratch_requirements(
    ::_BidiagonalReductionFunction,
    a::AbstractMatrix{<:Number};
    full::Bool = false,
    blocksize::Integer = 0,
    team = nothing,
)
    lock(_matrix_lock) do
        _iterative_resource_check(nothing, team)
        m, n, T, bs=_bidiagonalize_plan(a, blocksize)
        return _bidiagonalize_requirement(m, n, T, full, bs, team)
    end
end
function FaerScratch(
    f::_BidiagonalReductionFunction,
    a::AbstractMatrix{<:Number};
    kwargs...,
)
    req=scratch_requirements(f, a; kwargs...)
    return FaerScratch(req.bytes; alignment = req.alignment)
end
function reserve!(
    s::FaerScratch,
    f::_BidiagonalReductionFunction,
    a::AbstractMatrix{<:Number};
    kwargs...,
)
    req=scratch_requirements(f, a; kwargs...)
    return reserve!(s, req.bytes; alignment = req.alignment)
end
