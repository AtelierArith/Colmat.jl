using LinearAlgebra: lmul!, rmul!, mul!, ldiv!, rdiv!

# (requirements, apply); the apply function takes the optional team last.
_householder_workspace_functions(::FaerHouseholder) =
    (RustCrate.householder_requirements, RustCrate.householder_workspace)
_householder_workspace_functions(::FaerHouseholderSequence) =
    (RustCrate.sequence_requirements, RustCrate.sequence_workspace)
_householder_workspace_metadata(h::FaerHouseholder) = (UInt(size(h, 1)), _tag(eltype(h)))
_householder_workspace_metadata(h::FaerHouseholderSequence) =
    (UInt(size(h, 1)), UInt(h.k), UInt(h.blocksize), _tag(eltype(h)))
_householder_orientation(::FaerHouseholder) = (false, false)
_householder_orientation(q::FaerHouseholderSequence) = (q.transposed, q.conjugated)
function _householder_workspace_query(h, extent, right; team = nothing)
    _handle(h)
    isnothing(team) || team isa FaerTeam || throw(ArgumentError("team must be a FaerTeam"))
    isnothing(team) || _handle(team)
    info=zeros(UInt, 2)
    t, c=_householder_orientation(h)
    GC.@preserve info _native_call(
        first(_householder_workspace_functions(h)),
        h,
        _householder_workspace_metadata(h)...,
        UInt(extent),
        right,
        t,
        info,
    )
    return (bytes = Int(info[1]), alignment = Int(info[2]))
end
function _householder_resource_check(h, a, P, right; scratch = nothing, team = nothing)
    # Validate original native metadata before a promoted clone could mask it.
    extent=right ? size(a, 1) : ndims(a)==1 ? 1 : size(a, 2)
    _householder_workspace_query(h, extent, right; team)
    req=_householder_workspace_query(_householder_coerce(h, P), extent, right; team)
    if !isnothing(scratch)
        scratch isa FaerScratch || throw(ArgumentError("scratch must be a FaerScratch"))
        cap=scratch_capacity(scratch)
        cap.bytes>=req.bytes && cap.alignment>=req.alignment || throw(
            ArgumentError(
                "Householder scratch capacity/alignment is insufficient; call reserve!",
            ),
        )
    end
    return req
end
# Caller scratch/team are lent to faer's in-place application in the native
# owner itself (no result staging); without them the owner's kernel runs.
function _householder_resource_apply!(h, a, right; scratch = nothing, team = nothing)
    isnothing(scratch) && isnothing(team) && return _householder_native_apply!(h, a, right)
    _householder_apply_into!(h, a, right; scratch, team)
    return a
end
"""
    scratch_requirements(lmul!, H, A; team=nothing)
    scratch_requirements(rmul!, A, H; team=nothing)
    scratch_requirements(mul!, X, H, A[, alpha,beta]; team=nothing)

Query native scratch for elementary or compact-sequence Householder application.
Right products use `(X,A,H)`. `ldiv!`/`rdiv!` query the adjoint action.
`FaerScratch(fun,args...;team)` and `reserve!(work,fun,args...;team)` use these
queries. Applying functions accept `scratch=` and `team=`; insufficient or closed
resources fail before destination writes. Arrays, views, aliasing, scalar
promotion and transformed sequences keep their existing contracts. Result
staging, coefficient promotion and Julia conversions allocate outside this query;
native application borrows the team and does not retain its token.
"""
function scratch_requirements(
    ::typeof(lmul!),
    h::AbstractFaerHouseholder,
    a::_HouseholderArray;
    team = nothing,
)
    lock(_matrix_lock) do
        _householder_check(h, a, false)
        return _householder_resource_check(h, a, eltype(a), false; team)
    end
end
function scratch_requirements(
    ::typeof(rmul!),
    a::_HouseholderMatrix,
    h::AbstractFaerHouseholder;
    team = nothing,
)
    lock(_matrix_lock) do
        _householder_check(h, a, true)
        return _householder_resource_check(h, a, eltype(a), true; team)
    end
end
function _householder_mul_requirements(dest, h, a, alpha, beta, right; team = nothing)
    lock(_matrix_lock) do
        _householder_check(h, a, right)
        _householder_check(h, dest, right)
        size(dest)==size(a) ||
            throw(DimensionMismatch("Householder destination shape mismatch"))
        eltype(dest)<:_NativeScalar ||
            throw(ArgumentError("Householder destination requires native scalar"))
        P=promote_type(
            eltype(h),
            _reflector_type(eltype(a)),
            eltype(dest),
            typeof(alpha),
            typeof(beta),
        )
        P<:_NativeScalar || throw(ArgumentError("unsupported Householder product scalar"))
        return _householder_resource_check(h, a, P, right; team)
    end
end
scratch_requirements(
    ::typeof(mul!),
    dest::_HouseholderArray,
    h::AbstractFaerHouseholder,
    a::_HouseholderArray,
    alpha::Number = 1,
    beta::Number = 0;
    team = nothing,
) = _householder_mul_requirements(dest, h, a, alpha, beta, false; team)
scratch_requirements(
    ::typeof(mul!),
    dest::_HouseholderMatrix,
    a::_HouseholderMatrix,
    h::AbstractFaerHouseholder,
    alpha::Number = 1,
    beta::Number = 0;
    team = nothing,
) = _householder_mul_requirements(dest, h, a, alpha, beta, true; team)
scratch_requirements(
    ::typeof(ldiv!),
    h::AbstractFaerHouseholder,
    a::_HouseholderArray;
    team = nothing,
) = scratch_requirements(lmul!, adjoint(h), a; team)
scratch_requirements(
    ::typeof(rdiv!),
    a::_HouseholderMatrix,
    h::AbstractFaerHouseholder;
    team = nothing,
) = scratch_requirements(rmul!, a, adjoint(h); team)
scratch_requirements(
    ::typeof(ldiv!),
    dest::_HouseholderArray,
    h::AbstractFaerHouseholder,
    a::_HouseholderArray;
    team = nothing,
) = scratch_requirements(mul!, dest, adjoint(h), a; team)
for (fn, signature, args) in (
    (:lmul!, :(h::AbstractFaerHouseholder, a::_HouseholderArray), :(h, a)),
    (:rmul!, :(a::_HouseholderMatrix, h::AbstractFaerHouseholder), :(a, h)),
    (:ldiv!, :(h::AbstractFaerHouseholder, a::_HouseholderArray), :(h, a)),
    (:rdiv!, :(a::_HouseholderMatrix, h::AbstractFaerHouseholder), :(a, h)),
    (
        :ldiv!,
        :(dest::_HouseholderArray, h::AbstractFaerHouseholder, a::_HouseholderArray),
        :(dest, h, a),
    ),
    (
        :mul!,
        :(
            dest::_HouseholderArray,
            h::AbstractFaerHouseholder,
            a::_HouseholderArray,
            scales::Number...,
        ),
        :(dest, h, a, scales...),
    ),
    (
        :mul!,
        :(
            dest::_HouseholderMatrix,
            a::_HouseholderMatrix,
            h::AbstractFaerHouseholder,
            scales::Number...,
        ),
        :(dest, a, h, scales...),
    ),
)
    @eval function FaerScratch(::typeof($fn), $(signature.args...); team = nothing)
        req=scratch_requirements($fn, $(args.args...); team)
        return FaerScratch(req.bytes; alignment = req.alignment)
    end
    @eval function reserve!(
        work::FaerScratch,
        ::typeof($fn),
        $(signature.args...);
        team = nothing,
    )
        req=scratch_requirements($fn, $(args.args...); team)
        return reserve!(work, req.bytes; alignment = req.alignment)
    end
end
