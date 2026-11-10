const _SpectralTransformedFactor = Union{
    LinearAlgebra.TransposeFactorization{T,<:FaerSpectralFactor{T}},
    LinearAlgebra.AdjointFactorization{T,<:FaerSpectralFactor{T}},
} where {T<:_NativeScalar}
const _SpectralSolveFactor = Union{FaerSpectralFactor,_SpectralTransformedFactor}
_spectral_parent(f::FaerSpectralFactor) = f
_spectral_parent(f::_SpectralTransformedFactor) = parent(f)
_spectral_op(::FaerSpectralFactor) = 0
function _spectral_op(
    ::LinearAlgebra.TransposeFactorization{T,<:FaerSpectralFactor{T}}
) where {T}
    return 1
end
_spectral_op(::LinearAlgebra.AdjointFactorization{T,<:FaerSpectralFactor{T}}) where {T} = 2
function _spectral_rhs_open(b)
    if b isa _RotationArray
        _rotation_open(b)
    elseif _is_native_owner(b)
        _handle(b)
    elseif b isa Union{
        LinearAlgebra.Adjoint,
        LinearAlgebra.Transpose,
        LinearAlgebra.Hermitian,
        LinearAlgebra.Symmetric,
        LinearAlgebra.AbstractTriangular,
    }
        _spectral_rhs_open(parent(b))
    elseif b isa LinearAlgebra.Diagonal
        _spectral_rhs_open(b.diag)
    end
    return nothing
end
function _spectral_solve_rhs(
    f::_SpectralSolveFactor,
    b;
    atol=0,
    rtol=_default_spectral_rtol(_spectral_parent(f), atol),
    team=nothing,
    scratch=nothing,
    right=false,
)
    lock(_matrix_lock) do
        original = _spectral_parent(f)
        _handle(original)
        _spectral_rhs_open(b)
        _dense_rhs_validate(b)
        Base.require_one_based_indexing(b)
        isnothing(team) ||
            team isa FaerTeam ||
            throw(ArgumentError("team must be a FaerTeam"))
        !isnothing(team) && _handle(team)
        at, rt = _spectral_tolerances(atol, rtol)
        ndims(b) in (1, 2) ||
            throw(ArgumentError("spectral RHS must be a vector or matrix"))
        right && ndims(b) != 2 && throw(ArgumentError("right solve RHS must be a matrix"))
        (right ? size(b, 2) == size(f, 2) : size(b, 1) == size(f, 1)) ||
            throw(DimensionMismatch("spectral factor and RHS dimensions do not match"))
        P = promote_type(eltype(original), eltype(b))
        P <: _NativeScalar ||
            throw(ArgumentError("unsupported spectral solve scalar promotion"))
        extent = if right
            size(b, 1)
        elseif ndims(b) == 1
            1
        else
            size(b, 2)
        end
        req = _spectral_solve_workspace_query(
            original, P, _spectral_op(f), right, extent; team
        )
        temporary = isnothing(scratch)
        work = temporary ? FaerScratch(req.bytes; alignment=req.alignment) : scratch
        work isa FaerScratch || throw(ArgumentError("scratch must be a FaerScratch"))
        try
            cap = scratch_capacity(work)
            cap.bytes >= req.bytes && cap.alignment >= req.alignment || throw(
                ArgumentError(
                    "spectral solve scratch capacity/alignment is insufficient; call reserve!",
                ),
            )
            input = ndims(b) == 1 ? reshape(b, :, 1) : b
            rhs = b isa FaerMatrix ? _coerce(b, P) : FaerMatrix{P}(input)
            try
                dims = right ? (size(b, 1), size(f, 1)) : (size(f, 2), size(input, 2))
                args = (
                    original,
                    rhs,
                    UInt(size(original, 1)),
                    UInt(size(original, 2)),
                    UInt(dims[1]),
                    UInt(dims[2]),
                    _tag(eltype(original)),
                    _tag(P),
                    typeof(original).parameters[2] === :eigen,
                    Int32(_spectral_op(f)),
                    right,
                    at,
                    rt,
                    promote_type(typeof(rtol), _realtype(eltype(original))) === Float32,
                )
                fun = RustCrate.spectral_solve_workspace
                tail = (work, team)
                result = if P === Float64
                    _matrix_result(fun, dims, args..., tail...)
                else
                    _typed_result(P, fun, dims, args..., tail...)
                end
                if ndims(b) == 1
                    try
                        return vec(Matrix(result))
                    finally
                        close(result)
                    end
                end
                return result
            finally
                rhs === b || close(rhs)
            end
        finally
            temporary && close(work)
        end
    end
end
function Base.:\(f::_SpectralSolveFactor, b::AbstractVecOrMat{<:Number}; kwargs...)
    return _spectral_solve_rhs(f, b; kwargs...)
end
function Base.:/(b::AbstractMatrix{<:Number}, f::_SpectralSolveFactor; kwargs...)
    return _spectral_solve_rhs(f, b; right=true, kwargs...)
end
# Resolve the standard real-factor/complex-array reinterpretation methods.
for Wrapper in (LinearAlgebra.TransposeFactorization, LinearAlgebra.AdjointFactorization)
    @eval Base.:\(
        f::$Wrapper{T,<:FaerSpectralFactor{T}},
        b::Union{Vector{Complex{T}},Matrix{Complex{T}}};
        kwargs...,
    ) where {T<:Union{Float32,Float64}} = _spectral_solve_rhs(f, b; kwargs...)
    @eval Base.:/(
        b::Union{
            Matrix{Complex{T}},
            LinearAlgebra.Adjoint{Complex{T},Vector{Complex{T}}},
            LinearAlgebra.Transpose{Complex{T},Vector{Complex{T}}},
        },
        f::$Wrapper{T,<:FaerSpectralFactor{T}};
        kwargs...,
    ) where {T<:Union{Float32,Float64}} = _spectral_solve_rhs(f, b; right=true, kwargs...)
end
function Base.:\(
    f::FaerSpectralFactor{T}, b::Union{Vector{Complex{T}},Matrix{Complex{T}}}; kwargs...
) where {T<:Union{Float32,Float64}}
    return _spectral_solve_rhs(f, b; kwargs...)
end
function Base.:/(
    b::Union{
        Matrix{Complex{T}},
        LinearAlgebra.Adjoint{Complex{T},Vector{Complex{T}}},
        LinearAlgebra.Transpose{Complex{T},Vector{Complex{T}}},
    },
    f::FaerSpectralFactor{T};
    kwargs...,
) where {T<:Union{Float32,Float64}}
    return _spectral_solve_rhs(f, b; right=true, kwargs...)
end
for Wrapper in (LinearAlgebra.TransposeFactorization, LinearAlgebra.AdjointFactorization),
    BT in (
        :(Matrix{Complex{T}}),
        :(LinearAlgebra.Adjoint{Complex{T},Vector{Complex{T}}}),
        :(LinearAlgebra.Transpose{Complex{T},Vector{Complex{T}}}),
    )

    @eval Base.:/(
        b::$BT, f::$Wrapper{T,<:FaerSpectralFactor{T}}; kwargs...
    ) where {T<:Union{Float32,Float64}} = _spectral_solve_rhs(f, b; right=true, kwargs...)
end
function _spectral_ldiv!(dest, f, b; kwargs...)
    lock(_matrix_lock) do
        _spectral_rhs_open(dest)
        _dense_rhs_validate(dest)
        size(dest) == (ndims(b) == 1 ? (size(f, 2),) : (size(f, 2), size(b, 2))) ||
            throw(DimensionMismatch("spectral solve destination shape mismatch"))
        result = _spectral_solve_rhs(f, b; kwargs...)
        try
            return _operator_copy_result!(dest, result)
        finally
            result isa FaerMatrix && close(result)
        end
    end
end
"""
    ldiv!(dest, F::FaerSpectralFactor, B; atol=0, rtol, scratch=nothing, team=nothing)

Apply a retained SVD/self-adjoint EVD in a truncated minimum-norm solve. Stored
factors are reused, including mixed-precision/complex RHS, transpose and adjoint.
The cutoff follows the stored factor's precision. The solver projects/scales the
RHS through native factors without constructing a full pseudoinverse. Destination
conversion is staged before writeback; three-argument calls support rectangular
factors, while two-argument calls require square factors. `scratch=` includes
projection, weights and promoted coefficient buffers; result/Julia staging
allocates separately. Queries/reserve also cover backslash and right division.
"""
function LinearAlgebra.ldiv!(
    dest::AbstractMatrix, f::_SpectralSolveFactor, b::AbstractMatrix; kwargs...
)
    return _spectral_ldiv!(dest, f, b; kwargs...)
end
function LinearAlgebra.ldiv!(
    dest::AbstractVector, f::_SpectralSolveFactor, b::AbstractVector; kwargs...
)
    return _spectral_ldiv!(dest, f, b; kwargs...)
end
function LinearAlgebra.ldiv!(f::_SpectralSolveFactor, b::AbstractVecOrMat; kwargs...)
    size(f, 1) == size(f, 2) || throw(
        DimensionMismatch(
            "two-argument spectral solve requires square factors; use ldiv!(dest,F,B)"
        ),
    )
    return _spectral_ldiv!(b, f, b; kwargs...)
end
function LinearAlgebra.rdiv!(
    dest::AbstractMatrix, b::AbstractMatrix, f::_SpectralSolveFactor; kwargs...
)
    lock(_matrix_lock) do
        _spectral_rhs_open(dest)
        _dense_rhs_validate(dest)
        size(dest) == (size(b, 1), size(f, 1)) ||
            throw(DimensionMismatch("right spectral solve destination shape mismatch"))
        result = _spectral_solve_rhs(f, b; right=true, kwargs...)
        try
            return _operator_copy_result!(dest, result)
        finally
            close(result)
        end
    end
end
function LinearAlgebra.rdiv!(b::AbstractMatrix, f::_SpectralSolveFactor; kwargs...)
    size(f, 1) == size(f, 2) || throw(
        DimensionMismatch(
            "two-argument right solve requires square factors; use rdiv!(dest,B,F)"
        ),
    )
    return LinearAlgebra.rdiv!(b, b, f; kwargs...)
end
function Base.inv(
    f::_SpectralSolveFactor;
    atol::Real=0,
    rtol::Real=_default_spectral_rtol(_spectral_parent(f), atol),
    team=nothing,
    scratch=nothing,
)
    lock(_matrix_lock) do
        original = _spectral_parent(f)
        _handle(original)
        _spectral_solve_workspace_query(
            original, eltype(original), _spectral_op(f), false, 0; team
        )
        n = LinearAlgebra.checksquare(original)
        LinearAlgebra.rank(original; atol, rtol) == n ||
            throw(LinearAlgebra.SingularException(1))
        return _spectral_solve_rhs(
            f, Matrix{eltype(original)}(LinearAlgebra.I, n, n); atol, rtol, team, scratch
        )
    end
end

"""
    FaerLinearOperator(F::FaerSpectralFactor; atol=0, rtol=...)

Own an independent native inverse/pseudoinverse action using saved SVD or
self-adjoint EVD components. Small singular/eigenvalues are truncated using the
same tolerances as `pinv(F)`, chosen in the original factor precision. The
operator has shape reverse(size(F)), supports mixed native scalar promotion,
and applies with native scratch without forming a pseudoinverse matrix or
repeating decomposition. Iterative preconditioners require a suitable square
nonsingular action; rectangular/rank-truncated operators are general actions.
"""
function FaerLinearOperator(
    f::FaerSpectralFactor{T}; atol=0, rtol=_default_spectral_rtol(f, atol)
) where {T}
    lock(_matrix_lock) do
        _handle(f)
        at, rt = _spectral_tolerances(atol, rtol)
        h = RustCrate.OwnedLinearOperator()
        _native_call(
            RustCrate.operator_spectral_inverse,
            h,
            f,
            at,
            rt,
            promote_type(typeof(rtol), _realtype(T)) === Float32,
        )
        return FaerLinearOperator{T,typeof(h)}(
            h, reverse(size(f)), :factorization, false, false
        )
    end
end
