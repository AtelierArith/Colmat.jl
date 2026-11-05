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
function _bidiagonalize_requirement(m, n, T, full, bs, team, policy; compact = false)
    info=zeros(UInt, 2)
    GC.@preserve info begin
        args=compact ? (UInt(m), UInt(n), _tag(T), UInt(bs), policy.value, pointer(info)) :
             (UInt(m), UInt(n), _tag(T), full, UInt(bs), policy.value, pointer(info))
        isnothing(team) ?
        _native_call(
            (
                compact ? RustCrate.compact_bidiagonal_requirements :
                RustCrate.bidiagonalize_tuned_requirements
            ),
            args...,
        ) :
        _native_call(
            (
                compact ? RustCrate.compact_bidiagonal_team_requirements :
                RustCrate.bidiagonalize_tuned_team_requirements
            ),
            team,
            args...,
        )
    end
    return (bytes = Int(info[1]), alignment = Int(info[2]))
end
"""
    scratch_requirements(bidiagonalize, A; compact=false, full=false, blocksize=0, team=nothing, par_threshold=FaerParThreshold())
    scratch_requirements(bidiagonalize!, A; compact=false, full=false, blocksize=0, team=nothing, par_threshold=FaerParThreshold())

Query reusable bidiagonal reduction and explicit U/V workspace: canonical work
matrix, compact coefficients and a region reused by reduction and both basis
applications. Query and kernels use the actual scoped team shape/token. Input
conversion, returned owners and staged bang writeback allocate separately.
Use the same `par_threshold` as computation; `FaerParThreshold(:never)` queries sequential kernels
while still validating a supplied team. `compact=true` queries only the
reduction kernel: the returned pair independently owns packed reflector/factor
storage, and preserves the requested thin/full basis shapes.
"""
function scratch_requirements(
    ::_BidiagonalReductionFunction,
    a::AbstractMatrix{<:Number};
    full::Bool = false,
    blocksize::Integer = 0,
    team = nothing,
    par_threshold = FaerParThreshold(),
    compact::Bool = false,
)
    lock(_matrix_lock) do
        _iterative_resource_check(nothing, team)
        policy=_par_threshold(par_threshold)
        m, n, T, bs=_bidiagonalize_plan(a, blocksize)
        return _bidiagonalize_requirement(m, n, T, full, bs, team, policy; compact)
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
