_default_spectral_rtol(a, atol) =
    iszero(atol) ? min(size(a)...)*eps(_realtype(eltype(a))) : zero(_realtype(eltype(a)))
function _spectral_tolerances(atol, rtol)
    atol>=0 && rtol>=0 ||
        throw(ArgumentError("atol and rtol must be nonnegative and not NaN"))
    return Float64(atol), Float64(rtol)
end
function _pinv(a::FaerMatrix, mode, atol, rtol, team)
    at, rt=_spectral_tolerances(atol, rtol)
    lock(_matrix_lock) do
        T=eltype(a)
        dims=reverse(size(a))
        return T===Float64 ?
               _matrix_result(
            RustCrate.pseudoinverse_from,
            dims,
            a,
            Int32(mode),
            at,
            rt,
            promote_type(typeof(rtol), _realtype(T))===Float32,
            team,
        ) :
               _typed_result(
            T,
            RustCrate.pseudoinverse_from,
            dims,
            a,
            Int32(mode),
            at,
            rt,
            promote_type(typeof(rtol), _realtype(T))===Float32,
            team,
        )
    end
end
"""
    pinv(A::FaerMatrix; atol=0, rtol=min(size(A)...)*eps(real(eltype(A))), team=nothing)

Compute a Rust-owned Moore–Penrose pseudoinverse using faer's SVD and native
pseudoinverse reconstruction. Singular values at or below
`max(atol, rtol*maximum(svdvals(A)))` are discarded. Specifying nonzero `atol`
defaults `rtol` to zero. Hermitian inputs (and real Symmetric inputs) use faer's
self-adjoint eigendecomposition and honor the selected triangle. `team` lends a
caller-owned `FaerTeam` to the SVD/EVD and the reconstruction for this call; the
decomposition owns its factors, so no caller `scratch` form is provided.
"""
LinearAlgebra.pinv(
    a::FaerMatrix;
    atol::Real = 0,
    rtol::Real = _default_spectral_rtol(a, atol),
    team = nothing,
) = _pinv(a, 0, atol, rtol, team)
const _PinvSelfAdjoint=Union{_FaerSelfAdjoint,_ExtraSelfAdjoint}
LinearAlgebra.pinv(
    a::_PinvSelfAdjoint;
    atol::Real = 0,
    rtol::Real = _default_spectral_rtol(a, atol),
    team = nothing,
) = _pinv(parent(a), a.uplo=='L' ? 1 : 2, atol, rtol, team)
LinearAlgebra.pinv(a::FaerMatrixView; kwargs...) = LinearAlgebra.pinv(copy(a); kwargs...)
function _spectral_rank(a, values, atol, rtol)
    _spectral_tolerances(atol, rtol)
    isempty(values) && return 0
    tol=max(atol, rtol*maximum(values))
    return count(>(tol), values)
end
function LinearAlgebra.rank(
    a::FaerMatrix;
    atol::Real = 0,
    rtol::Real = _default_spectral_rtol(a, atol),
)
    _spectral_tolerances(atol, rtol)
    return _spectral_rank(a, LinearAlgebra.svdvals(a), atol, rtol)
end
function LinearAlgebra.rank(
    a::_PinvSelfAdjoint;
    atol::Real = 0,
    rtol::Real = _default_spectral_rtol(a, atol),
)
    _spectral_tolerances(atol, rtol)
    return _spectral_rank(a, abs.(LinearAlgebra.eigvals(a)), atol, rtol)
end
LinearAlgebra.rank(a::FaerMatrixView; kwargs...) = LinearAlgebra.rank(copy(a); kwargs...)
function LinearAlgebra.opnorm(a::FaerMatrix{T}, p::Real = 2) where {T<:_ExtraScalar}
    p in (1, 2, Inf) || throw(ArgumentError("operator norm supports p=1, 2 or Inf"))
    if p==2
        values=LinearAlgebra.svdvals(a)
        return isempty(values) ? zero(_realtype(T)) : maximum(values)
    end
    R=_realtype(T)
    out=Ref{R}()
    GC.@preserve out _native_call(
        RustCrate.operator_norm,
        a,
        Ptr{UInt8}(Base.unsafe_convert(Ptr{R}, out)),
        p==Inf,
    )
    return out[]
end
function LinearAlgebra.cond(a::FaerMatrix{T}, p::Real = 2) where {T}
    p in (1, 2, Inf) || throw(ArgumentError("condition number supports p=1, 2 or Inf"))
    lock(_matrix_lock) do
        _handle(a)
        R=_realtype(T)
        if isempty(a)
            LinearAlgebra.checksquare(a)
            return zero(R)
        elseif p==2
            values=LinearAlgebra.svdvals(a)
            largest=maximum(values)
            return iszero(largest) ? R(Inf) : largest/minimum(values)
        end
        LinearAlgebra.checksquare(a)
        try
            return LinearAlgebra.opnorm(a, p)*LinearAlgebra.opnorm(inv(a), p)
        catch e
            e isa LinearAlgebra.SingularException || rethrow()
            return R(Inf)
        end
    end
end
LinearAlgebra.cond(a::FaerMatrixView, p::Real = 2) = LinearAlgebra.cond(copy(a), p)
