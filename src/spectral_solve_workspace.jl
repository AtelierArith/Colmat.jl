function _spectral_solve_workspace_query(
    f::FaerSpectralFactor{T,K},
    P,
    op,
    right,
    extent;
    team = nothing,
) where {T,K}
    _handle(f)
    isnothing(team) || team isa FaerTeam || throw(ArgumentError("team must be a FaerTeam"))
    isnothing(team) || _handle(team)
    info=zeros(UInt, 2)
    args=(
        f,
        UInt(size(f, 1)),
        UInt(size(f, 2)),
        _tag(T),
        _tag(P),
        K===:eigen,
        Int32(op),
        right,
        UInt(extent),
        info,
    )
    GC.@preserve info begin
        _native_call(RustCrate.spectral_solve_requirements, args..., team)
    end
    return (bytes = Int(info[1]), alignment = Int(info[2]))
end
function _spectral_solve_requirements(
    f,
    b;
    atol = 0,
    rtol = _default_spectral_rtol(_spectral_parent(f), atol),
    team = nothing,
    right = false,
)
    lock(_matrix_lock) do
        original=_spectral_parent(f)
        _handle(original)
        _spectral_rhs_open(b)
        _dense_rhs_validate(b)
        Base.require_one_based_indexing(b)
        _spectral_tolerances(atol, rtol)
        (right ? size(b, 2)==size(f, 2) : size(b, 1)==size(f, 1)) ||
            throw(DimensionMismatch("spectral factor and RHS dimensions do not match"))
        P=promote_type(eltype(original), eltype(b))
        P<:_NativeScalar ||
            throw(ArgumentError("unsupported spectral solve scalar promotion"))
        return _spectral_solve_workspace_query(
            original,
            P,
            _spectral_op(f),
            right,
            right ? size(b, 1) : ndims(b)==1 ? 1 : size(b, 2);
            team,
        )
    end
end
"""
    scratch_requirements(\\, F::FaerSpectralFactor, B; atol=0, rtol, team=nothing)
    scratch_requirements(ldiv!, X, F::FaerSpectralFactor, B; atol=0, rtol, team=nothing)

Query scratch for saved SVD/self-adjoint EVD solves. Right division uses `(B,F)`
or `(X,B,F)`; square two-argument ldiv!/rdiv! forms and `inv(F)` are supported. Allocate
with `FaerScratch(fun,args...;kwargs...)`, grow with `reserve!`, and pass `scratch=`
to the corresponding solve. Native projection, reciprocal weights and promoted
U/V copies are inside scratch. Results and Julia/RHS conversions allocate outside
it. Cutoffs are computed in the original factor precision; no decomposition or
explicit pseudoinverse is formed. Queries validate current owners and teams,
including zero-sized inputs. Tolerances are validated but do not change capacity.
"""
scratch_requirements(::typeof(\), f::_SpectralSolveFactor, b::AbstractVecOrMat; kwargs...) =
    _spectral_solve_requirements(f, b; kwargs...)
scratch_requirements(::typeof(/), b::AbstractMatrix, f::_SpectralSolveFactor; kwargs...) =
    _spectral_solve_requirements(f, b; right = true, kwargs...)
function scratch_requirements(
    ::typeof(LinearAlgebra.ldiv!),
    dest::AbstractVecOrMat,
    f::_SpectralSolveFactor,
    b::AbstractVecOrMat;
    kwargs...,
)
    lock(_matrix_lock) do
        _spectral_rhs_open(dest)
        _dense_rhs_validate(dest)
        size(dest)==(ndims(b)==1 ? (size(f, 2),) : (size(f, 2), size(b, 2))) ||
            throw(DimensionMismatch("spectral solve destination shape mismatch"))
        return _spectral_solve_requirements(f, b; kwargs...)
    end
end
function scratch_requirements(
    ::typeof(LinearAlgebra.rdiv!),
    dest::AbstractMatrix,
    b::AbstractMatrix,
    f::_SpectralSolveFactor;
    kwargs...,
)
    lock(_matrix_lock) do
        _spectral_rhs_open(dest)
        _dense_rhs_validate(dest)
        size(dest)==(size(b, 1), size(f, 1)) ||
            throw(DimensionMismatch("right spectral solve destination shape mismatch"))
        return _spectral_solve_requirements(f, b; right = true, kwargs...)
    end
end
function scratch_requirements(
    ::typeof(LinearAlgebra.ldiv!),
    f::_SpectralSolveFactor,
    b::AbstractVecOrMat;
    kwargs...,
)
    size(f, 1)==size(f, 2) ||
        throw(DimensionMismatch("two-argument spectral solve requires square factors"))
    return scratch_requirements(LinearAlgebra.ldiv!, b, f, b; kwargs...)
end
function scratch_requirements(
    ::typeof(LinearAlgebra.rdiv!),
    b::AbstractMatrix,
    f::_SpectralSolveFactor;
    kwargs...,
)
    size(f, 1)==size(f, 2) || throw(
        DimensionMismatch("two-argument right spectral solve requires square factors"),
    )
    return scratch_requirements(LinearAlgebra.rdiv!, b, b, f; kwargs...)
end
function scratch_requirements(
    ::typeof(LinearAlgebra.inv),
    f::_SpectralSolveFactor;
    atol = 0,
    rtol = _default_spectral_rtol(_spectral_parent(f), atol),
    team = nothing,
)
    lock(_matrix_lock) do
        original=_spectral_parent(f)
        _handle(original)
        _spectral_tolerances(atol, rtol)
        n=LinearAlgebra.checksquare(original)
        return _spectral_solve_workspace_query(
            original,
            eltype(original),
            _spectral_op(f),
            false,
            n;
            team,
        )
    end
end

for (fn, signature, args) in (
    (:(LinearAlgebra.inv), :((f::_SpectralSolveFactor,)), :(f,)),
    (:(\), :(f::_SpectralSolveFactor, b::AbstractVecOrMat), :(f, b)),
    (:(/), :(b::AbstractMatrix, f::_SpectralSolveFactor), :(b, f)),
    (
        :(LinearAlgebra.ldiv!),
        :(dest::AbstractVecOrMat, f::_SpectralSolveFactor, b::AbstractVecOrMat),
        :(dest, f, b),
    ),
    (
        :(LinearAlgebra.rdiv!),
        :(dest::AbstractMatrix, b::AbstractMatrix, f::_SpectralSolveFactor),
        :(dest, b, f),
    ),
    (:(LinearAlgebra.ldiv!), :(f::_SpectralSolveFactor, b::AbstractVecOrMat), :(f, b)),
    (:(LinearAlgebra.rdiv!), :(b::AbstractMatrix, f::_SpectralSolveFactor), :(b, f)),
)
    @eval function FaerScratch(::typeof($fn), $(signature.args...); kwargs...)
        req=scratch_requirements($fn, $(args.args...); kwargs...)
        return FaerScratch(req.bytes; alignment = req.alignment)
    end
    @eval function reserve!(
        work::FaerScratch,
        ::typeof($fn),
        $(signature.args...);
        kwargs...,
    )
        req=scratch_requirements($fn, $(args.args...); kwargs...)
        return reserve!(work, req.bytes; alignment = req.alignment)
    end
end
