# LU/Cholesky numeric owners are independent of the source and symbolic plan.
const _SparseDirectParent =
    Union{FaerSparseFactorization{T,:lu},FaerSparseFactorization{T,:cholesky}} where {T}
const _SparseDirectTrans = Union{
    LinearAlgebra.TransposeFactorization{
        T,
        <:Union{FaerSparseFactorization{T,:lu},FaerSparseFactorization{T,:cholesky}},
    },
    LinearAlgebra.AdjointFactorization{
        T,
        <:Union{FaerSparseFactorization{T,:lu},FaerSparseFactorization{T,:cholesky}},
    },
} where {T}
const _SparseDirectSolve = Union{_SparseDirectParent,_SparseDirectTrans}
const _SparseDirectOf{T} = Union{
    FaerSparseFactorization{T,:lu},
    FaerSparseFactorization{T,:cholesky},
    LinearAlgebra.TransposeFactorization{
        T,
        <:Union{FaerSparseFactorization{T,:lu},FaerSparseFactorization{T,:cholesky}},
    },
    LinearAlgebra.AdjointFactorization{
        T,
        <:Union{FaerSparseFactorization{T,:lu},FaerSparseFactorization{T,:cholesky}},
    },
}
_sparse_direct_parent(f::_SparseDirectParent) = f
_sparse_direct_parent(f::_SparseDirectTrans) = parent(f)
_sparse_direct_op(::_SparseDirectParent) = 0
_sparse_direct_op(
    ::LinearAlgebra.TransposeFactorization{T,<:_SparseDirectParent},
) where {T} = 1
_sparse_direct_op(::LinearAlgebra.AdjointFactorization{T,<:_SparseDirectParent}) where {T} =
    2
_sparse_direct_kind(::FaerSparseFactorization{T,:lu}) where {T} = Int32(0)
_sparse_direct_kind(::FaerSparseFactorization{T,:cholesky}) where {T} = Int32(2)
function _sparse_direct_prepare(f, b, team, right)
    original=_sparse_direct_parent(f)
    _handle(original)
    _spectral_rhs_open(b)
    _dense_rhs_validate(b)
    Base.require_one_based_indexing(b)
    isnothing(team) || team isa FaerTeam || throw(ArgumentError("team must be a FaerTeam"))
    isnothing(team) || _handle(team)
    LinearAlgebra.issuccess(original) || (
        _sparse_direct_kind(original)==2 ?
        throw(LinearAlgebra.PosDefException(original.info)) :
        throw(LinearAlgebra.SingularException(original.info))
    )
    _native_call(
        RustCrate.sparse_direct_validate,
        original,
        UInt(size(original, 1)),
        UInt(size(original, 2)),
        _tag(eltype(original)),
        _sparse_direct_kind(original),
    )
    (right ? size(b, 2)==size(f, 2) : size(b, 1)==size(f, 1)) ||
        throw(DimensionMismatch("sparse factor and RHS dimensions must match"))
    op=_sparse_direct_op(f)
    right && (op=(2, 3, 0)[op+1])
    P=promote_type(eltype(original), _sparse_input_type(eltype(b)))
    return original, op, P
end
function _with_sparse_direct_operator(fun, f, P, op)
    base=FaerLinearOperator(f)
    promoted=nothing
    action=nothing
    try
        promoted=_operator_coerce(base, P)
        action=op==0 ? promoted :
               op==1 ? transpose(promoted) : op==2 ? adjoint(promoted) : conj(promoted)
        return fun(action)
    finally
        isnothing(action) || action===promoted || close(action)
        isnothing(promoted) || promoted===base || close(promoted)
        close(base)
    end
end
function _sparse_direct_native_query(f, columns, op; team = nothing)
    info=zeros(UInt, 2)
    args=(
        f,
        UInt(columns),
        Int32(op),
        UInt(size(f, 1)),
        UInt(size(f, 2)),
        _tag(eltype(f)),
        _sparse_direct_kind(f),
        pointer(info),
    )
    GC.@preserve info begin
        isnothing(team) ? _native_call(RustCrate.sparse_direct_requirements, args...) :
        _native_call(RustCrate.sparse_direct_requirements_team, args..., team)
    end
    return (bytes = Int(info[1]), alignment = Int(info[2]))
end
function _sparse_direct_requirements(f, b; team = nothing, right = false)
    lock(_matrix_lock) do
        original, op, P=_sparse_direct_prepare(f, b, team, right)
        columns=right ? size(b, 1) : ndims(b)==1 ? 1 : size(b, 2)
        P===eltype(original) &&
            return _sparse_direct_native_query(original, columns, op; team)
        return _with_sparse_direct_operator(original, P, op) do action
            _operator_workspace_query(action, columns; team)
        end
    end
end
function _sparse_direct_native_result(f, rhs, op; scratch = nothing, team = nothing)
    req=_sparse_direct_native_query(f, size(rhs, 2), op; team)
    temporary=isnothing(scratch)
    work=temporary ? FaerScratch(req.bytes; alignment = req.alignment) : scratch
    work isa FaerScratch || throw(ArgumentError("scratch must be a FaerScratch"))
    try
        cap=scratch_capacity(work)
        cap.bytes>=req.bytes && cap.alignment>=req.alignment ||
            throw(ArgumentError("insufficient sparse solve scratch; call reserve!"))
        args=(
            f,
            rhs,
            work,
            UInt(size(rhs, 2)),
            Int32(op),
            UInt(size(f, 1)),
            UInt(size(f, 2)),
            _tag(eltype(f)),
            _sparse_direct_kind(f),
        )
        fun=isnothing(team) ? RustCrate.sparse_direct_workspace :
            RustCrate.sparse_direct_workspace_team
        isnothing(team) || (args=(args..., team))
        return eltype(f)===Float64 ? _matrix_result(fun, size(rhs), args...) :
               _typed_result(eltype(f), fun, size(rhs), args...)
    finally
        temporary && close(work)
    end
end
function _sparse_direct_result(f, b; scratch = nothing, team = nothing, right = false)
    lock(_matrix_lock) do
        original, op, P=_sparse_direct_prepare(f, b, team, right)
        input=right ? adjoint(b) : ndims(b)==1 ? reshape(b, :, 1) : b
        rhs=FaerMatrix{P}(input)
        try
            result=if P===eltype(original)
                _sparse_direct_native_result(original, rhs, op; scratch, team)
            else
                _with_sparse_direct_operator(original, P, op) do action
                    _operator_workspace_result(action, rhs; scratch, team)
                end
            end
            right || return result
            try
                return FaerMatrix(copy(adjoint(Matrix(result))))
            finally
                close(result)
            end
        finally
            close(rhs)
        end
    end
end
function _sparse_direct_vector(f, b; kwargs...)
    result=_sparse_direct_result(f, b; kwargs...)
    try
        return b isa _BorrowVector ? _vector_from_source(result) : vec(Matrix(result))
    finally
        close(result)
    end
end
function _sparse_direct_write(
    dest,
    f,
    b;
    right = false,
    scratch = nothing,
    team = nothing,
    atol = nothing,
    rtol = nothing,
)
    lock(_matrix_lock) do
        isnothing(atol) && isnothing(rtol) ||
            throw(ArgumentError("rank tolerances require sparse QR"))
        _spectral_rhs_open(dest)
        _dense_rhs_validate(dest)
        Base.require_one_based_indexing(dest)
        expected=right ? (size(b, 1), size(f, 1)) :
                 ndims(b)==1 ? (size(f, 2),) : (size(f, 2), size(b, 2))
        size(dest)==expected ||
            throw(DimensionMismatch("sparse solve destination shape mismatch"))
        result=_sparse_direct_result(f, b; right, scratch, team)
        try
            buffer=ndims(dest)==1 ? vec(Matrix(result)) : Matrix(result)
            return _dense_copy_result!(dest, buffer)
        finally
            close(result)
        end
    end
end
Base.:\(f::_SparseDirectSolve, b::AbstractMatrix{<:Number}; kwargs...) =
    _sparse_direct_result(f, b; kwargs...)
Base.:\(f::_SparseDirectSolve, b::AbstractVector{<:Number}; kwargs...) =
    _sparse_direct_vector(f, b; kwargs...)
Base.:/(b::AbstractMatrix{<:Number}, f::_SparseDirectSolve; kwargs...) =
    _sparse_direct_result(f, b; right = true, kwargs...)
for B in (:AbstractMatrix, :AbstractVector)
    @eval LinearAlgebra.ldiv!(dest::$B, f::_SparseDirectSolve, b::$B; kwargs...) =
        _sparse_direct_write(dest, f, b; kwargs...)
end
LinearAlgebra.ldiv!(f::_SparseDirectSolve, b::AbstractVecOrMat; kwargs...) =
    _sparse_direct_write(b, f, b; kwargs...)
LinearAlgebra.rdiv!(
    dest::AbstractMatrix,
    b::AbstractMatrix,
    f::_SparseDirectSolve;
    kwargs...,
) = _sparse_direct_write(dest, f, b; right = true, kwargs...)
LinearAlgebra.rdiv!(b::AbstractMatrix, f::_SparseDirectSolve; kwargs...) =
    _sparse_direct_write(b, f, b; right = true, kwargs...)
for Wrapper in (LinearAlgebra.TransposeFactorization, LinearAlgebra.AdjointFactorization)
    @eval LinearAlgebra.rdiv!(
        b::AbstractMatrix,
        f::$Wrapper{
            T,
            <:Union{FaerSparseFactorization{T,:lu},FaerSparseFactorization{T,:cholesky}},
        };
        kwargs...,
    ) where {T} = _sparse_direct_write(b, f, b; right = true, kwargs...)
end
for B in (:Matrix, :Vector)
    helper=B===:Matrix ? :_sparse_direct_result : :_sparse_direct_vector
    @eval Base.:\(
        f::_SparseDirectOf{T},
        b::$B{Complex{T}};
        kwargs...,
    ) where {T<:Union{Float32,Float64}} = $helper(f, b; kwargs...)
end
Base.:/(
    b::Matrix{Complex{T}},
    f::_SparseDirectOf{T};
    kwargs...,
) where {T<:Union{Float32,Float64}} = _sparse_direct_result(f, b; right = true, kwargs...)
"""
    scratch_requirements(ldiv!, F, B; team=nothing)
    scratch_requirements(rdiv!, B, F; team=nothing)

Query retained sparse LU/Cholesky solve work for all algorithms. Transpose,
adjoint, vector/matrix RHS, right division and scalar promotion are supported.
Use `FaerScratch(fun,args...;team)` or `reserve!(work,fun,args...;team)` and pass
`scratch=work` to the solve. Same-type solves borrow the saved numeric factors.
Mixed types convert saved sparse factors into an inverse operator without
refactorizing the input; preparing that operator allocates outside solve scratch.
RHS/result storage and destination staging also allocate separately. Teams are
borrowed only for native execution. Closed/undersized resources fail before
writeback. QR retains its separate workspace interface.
"""
scratch_requirements(
    ::typeof(LinearAlgebra.ldiv!),
    f::_SparseDirectSolve,
    b::AbstractVecOrMat;
    team = nothing,
) = _sparse_direct_requirements(f, b; team)
scratch_requirements(
    ::typeof(LinearAlgebra.rdiv!),
    b::AbstractMatrix,
    f::_SparseDirectSolve;
    team = nothing,
) = _sparse_direct_requirements(f, b; team, right = true)
function scratch_requirements(
    ::typeof(LinearAlgebra.ldiv!),
    dest::AbstractVecOrMat,
    f::_SparseDirectSolve,
    b::AbstractVecOrMat;
    team = nothing,
)
    lock(_matrix_lock) do
        _spectral_rhs_open(dest)
        _dense_rhs_validate(dest)
        size(dest)==size(b) ||
            throw(DimensionMismatch("sparse solve destination shape mismatch"))
        return _sparse_direct_requirements(f, b; team)
    end
end
function scratch_requirements(
    ::typeof(LinearAlgebra.rdiv!),
    dest::AbstractMatrix,
    b::AbstractMatrix,
    f::_SparseDirectSolve;
    team = nothing,
)
    lock(_matrix_lock) do
        _spectral_rhs_open(dest)
        _dense_rhs_validate(dest)
        size(dest)==size(b) ||
            throw(DimensionMismatch("sparse solve destination shape mismatch"))
        return _sparse_direct_requirements(f, b; team, right = true)
    end
end
scratch_requirements(
    ::typeof(\),
    f::_SparseDirectSolve,
    b::AbstractVecOrMat;
    team = nothing,
) = _sparse_direct_requirements(f, b; team)
scratch_requirements(
    ::typeof(/),
    b::AbstractMatrix,
    f::_SparseDirectSolve;
    team = nothing,
) = _sparse_direct_requirements(f, b; team, right = true)
FaerScratch(
    fun::typeof(LinearAlgebra.ldiv!),
    f::_SparseDirectSolve,
    b::AbstractVecOrMat;
    team = nothing,
) = _new_operator_workspace(fun, f, b; team)
reserve!(
    s::FaerScratch,
    fun::typeof(LinearAlgebra.ldiv!),
    f::_SparseDirectSolve,
    b::AbstractVecOrMat;
    team = nothing,
) = _reserve_operator_workspace(s, fun, f, b; team)
FaerScratch(
    fun::typeof(LinearAlgebra.ldiv!),
    dest::AbstractVecOrMat,
    f::_SparseDirectSolve,
    b::AbstractVecOrMat;
    team = nothing,
) = _new_operator_workspace(fun, dest, f, b; team)
reserve!(
    s::FaerScratch,
    fun::typeof(LinearAlgebra.ldiv!),
    dest::AbstractVecOrMat,
    f::_SparseDirectSolve,
    b::AbstractVecOrMat;
    team = nothing,
) = _reserve_operator_workspace(s, fun, dest, f, b; team)
FaerScratch(
    fun::typeof(LinearAlgebra.rdiv!),
    b::AbstractMatrix,
    f::_SparseDirectSolve;
    team = nothing,
) = _new_operator_workspace(fun, b, f; team)
reserve!(
    s::FaerScratch,
    fun::typeof(LinearAlgebra.rdiv!),
    b::AbstractMatrix,
    f::_SparseDirectSolve;
    team = nothing,
) = _reserve_operator_workspace(s, fun, b, f; team)
FaerScratch(
    fun::typeof(LinearAlgebra.rdiv!),
    dest::AbstractMatrix,
    b::AbstractMatrix,
    f::_SparseDirectSolve;
    team = nothing,
) = _new_operator_workspace(fun, dest, b, f; team)
reserve!(
    s::FaerScratch,
    fun::typeof(LinearAlgebra.rdiv!),
    dest::AbstractMatrix,
    b::AbstractMatrix,
    f::_SparseDirectSolve;
    team = nothing,
) = _reserve_operator_workspace(s, fun, dest, b, f; team)
FaerScratch(fun::typeof(\), f::_SparseDirectSolve, b::AbstractVecOrMat; team = nothing) =
    _new_operator_workspace(fun, f, b; team)
reserve!(
    s::FaerScratch,
    fun::typeof(\),
    f::_SparseDirectSolve,
    b::AbstractVecOrMat;
    team = nothing,
) = _reserve_operator_workspace(s, fun, f, b; team)
FaerScratch(fun::typeof(/), b::AbstractMatrix, f::_SparseDirectSolve; team = nothing) =
    _new_operator_workspace(fun, b, f; team)
reserve!(
    s::FaerScratch,
    fun::typeof(/),
    b::AbstractMatrix,
    f::_SparseDirectSolve;
    team = nothing,
) = _reserve_operator_workspace(s, fun, b, f; team)

# LinearAlgebra defines additional real-factor/complex-row specializations.
for Wrapper in (LinearAlgebra.TransposeFactorization, LinearAlgebra.AdjointFactorization)
    @eval Base.:/(
        b::Union{
            Matrix{Complex{T}},
            LinearAlgebra.Adjoint{Complex{T},Vector{Complex{T}}},
            LinearAlgebra.Transpose{Complex{T},Vector{Complex{T}}},
        },
        f::$Wrapper{
            T,
            <:Union{FaerSparseFactorization{T,:lu},FaerSparseFactorization{T,:cholesky}},
        };
        kwargs...,
    ) where {T<:Union{Float32,Float64}} =
        _sparse_direct_result(f, b; right = true, kwargs...)
end
for Wrapper in (LinearAlgebra.TransposeFactorization, LinearAlgebra.AdjointFactorization),
    Row in (LinearAlgebra.Transpose, LinearAlgebra.Adjoint)

    @eval Base.:/(
        b::$Row{Complex{T},Vector{Complex{T}}},
        f::$Wrapper{
            T,
            <:Union{FaerSparseFactorization{T,:lu},FaerSparseFactorization{T,:cholesky}},
        };
        kwargs...,
    ) where {T<:Union{Float32,Float64}} =
        _sparse_direct_result(f, b; right = true, kwargs...)
end

for Wrapper in (LinearAlgebra.TransposeFactorization, LinearAlgebra.AdjointFactorization)
    @eval Base.:/(
        b::Matrix{Complex{T}},
        f::$Wrapper{
            T,
            <:Union{FaerSparseFactorization{T,:lu},FaerSparseFactorization{T,:cholesky}},
        };
        kwargs...,
    ) where {T<:Union{Float32,Float64}} =
        _sparse_direct_result(f, b; right = true, kwargs...)
end
