"""
    FaerLinearOperator(A)
    FaerLinearOperator(I, n; T=Float64)

Own an independent faer LinOp/BiLinOp snapshot of a dense matrix, native/Julia
CSC or native CSR sparse matrix, or diagonal matrix. Identity and diagonal
operators do not store a dense coefficient matrix. `*`, `mul!`, conjugation,
transpose and adjoint call native operator traits; Matrix(op) explicitly forms
its dense matrix. `.coefficients` returns an independent base representation,
with `.transposed/.conjugated` describing its orientation.
Arbitrary Julia-function callbacks and operator composition are not yet exposed.
"""
mutable struct FaerLinearOperator{T,H} <: AbstractMatrix{T}
    handle::Union{Nothing,H}
    dims::Tuple{Int,Int}
    storage::Symbol
    transposed::Bool
    conjugated::Bool
end
function _handle(op::FaerLinearOperator)
    h=getfield(op, :handle)
    isnothing(h) && throw(ArgumentError("linear operator has been closed"))
    return h
end
_native_argument(op::FaerLinearOperator) = _handle(op)
_is_native_owner(::FaerLinearOperator) = true
Base.size(op::FaerLinearOperator) = getfield(op, :dims)
Base.size(op::FaerLinearOperator, d::Integer) =
    d<1 ? throw(ArgumentError("dimension must be positive")) : d<=2 ? size(op)[d] : 1
Base.isopen(op::FaerLinearOperator) =
    lock(()->!isnothing(getfield(op, :handle)), _matrix_lock)
function Base.close(op::FaerLinearOperator)
    lock(_matrix_lock) do ;
        op.handle=nothing
    end
    return nothing
end
function _operator_create(a, kind, storage)
    lock(_matrix_lock) do
        h=RustCrate.OwnedLinearOperator()
        _native_call(RustCrate.operator_load, h, a, Int32(kind))
        return FaerLinearOperator{eltype(a),typeof(h)}(h, size(a), storage, false, false)
    end
end
function FaerLinearOperator(a::FaerMatrix)
    lock(_matrix_lock) do
        all(isfinite, Matrix(a)) ||
            throw(ArgumentError("operator coefficients must be finite"))
        return _operator_create(a, eltype(a)===Float64 ? 0 : 1, :dense)
    end
end
function FaerLinearOperator(a::FaerSparseMatrix)
    lock(_matrix_lock) do
        all(isfinite, Vector(SparseArrays.nonzeros(a))) ||
            throw(ArgumentError("operator stored coefficients must be finite"))
        return _operator_create(a, 2, a.format)
    end
end
function FaerLinearOperator(a::FaerDiagonal)
    lock(_matrix_lock) do
        all(isfinite, Vector(a)) || throw(ArgumentError("operator diagonal must be finite"))
        return _operator_create(a, 3, :diagonal)
    end
end
FaerLinearOperator(a::SparseArrays.AbstractSparseMatrixCSC) =
    FaerLinearOperator(FaerSparseMatrix(a))
FaerLinearOperator(a::LinearAlgebra.Diagonal) =
    FaerLinearOperator(FaerDiagonal{_reflector_type(eltype(a))}(a.diag))
function FaerLinearOperator(a::Union{LinearAlgebra.Hermitian,LinearAlgebra.Symmetric})
    source=parent(a)
    if source isa FaerSparseMatrix
        coefficients=SparseArrays.SparseMatrixCSC(source)
        wrapped=a isa LinearAlgebra.Hermitian ?
                LinearAlgebra.Hermitian(coefficients, a.uplo=='L' ? :L : :U) :
                LinearAlgebra.Symmetric(coefficients, a.uplo=='L' ? :L : :U)
        return FaerLinearOperator(
            FaerSparseMatrix(SparseArrays.sparse(wrapped); format = source.format),
        )
    elseif source isa SparseArrays.AbstractSparseMatrixCSC
        return FaerLinearOperator(SparseArrays.sparse(a))
    end
    return FaerLinearOperator(FaerMatrix(a))
end
FaerLinearOperator(a::AbstractMatrix) =
    FaerLinearOperator(FaerMatrix{_reflector_type(eltype(a))}(a))
FaerLinearOperator(a::FaerMatrixView) = FaerLinearOperator(copy(a))
function FaerLinearOperator(
    identity::LinearAlgebra.UniformScaling,
    n::Integer;
    T::Type = Float64,
)
    identity.λ==1 || throw(ArgumentError("only unit identity scaling is supported"))
    T<:_NativeScalar ||
        throw(ArgumentError("identity operator requires a native scalar type"))
    n=Int(n)
    n>=0 || throw(ArgumentError("identity dimension must be nonnegative"))
    lock(_matrix_lock) do
        h=RustCrate.OwnedLinearOperator()
        _native_call(RustCrate.operator_identity, h, UInt(n), _tag(T))
        return FaerLinearOperator{T,typeof(h)}(h, (n, n), :identity, false, false)
    end
end
function _operator_copy(
    op::FaerLinearOperator{T},
    transpose = false,
    conjugate = false,
) where {T}
    lock(_matrix_lock) do
        h=RustCrate.OwnedLinearOperator()
        _native_call(RustCrate.operator_clone, h, op, transpose, conjugate)
        dims=transpose ? reverse(size(op)) : size(op)
        return FaerLinearOperator{T,typeof(h)}(
            h,
            dims,
            op.storage,
            op.transposed⊻transpose,
            op.conjugated⊻conjugate,
        )
    end
end
Base.copy(op::FaerLinearOperator) = _operator_copy(op)
FaerLinearOperator(op::FaerLinearOperator) = copy(op)
Base.deepcopy_internal(op::FaerLinearOperator, d::IdDict) = get!(() -> copy(op), d, op)
Base.transpose(op::FaerLinearOperator) = _operator_copy(op, true, false)
Base.conj(op::FaerLinearOperator) = _operator_copy(op, false, true)
Base.adjoint(op::FaerLinearOperator) = _operator_copy(op, true, true)
function _operator_coefficients(op::FaerLinearOperator{T}) where {T}
    lock(_matrix_lock) do
        _handle(op)
        dims=op.transposed ? reverse(size(op)) : size(op)
        if op.storage===:dense
            return T===Float64 ? _matrix_result(RustCrate.operator_dense_from, dims, op) :
                   _typed_result(T, RustCrate.operator_dense_from, dims, op)
        elseif op.storage in (:csc, :csr)
            return _sparse_result(T, dims, op.storage, RustCrate.operator_sparse_from, op)
        elseif op.storage===:diagonal
            return _collection_result(
                FaerDiagonal,
                T,
                dims[1],
                RustCrate.operator_diagonal_from,
                op,
            )
        end
        return LinearAlgebra.I
    end
end
function Base.getproperty(op::FaerLinearOperator, name::Symbol)
    name===:coefficients && return _operator_coefficients(op)
    return getfield(op, name)
end
Base.propertynames(op::FaerLinearOperator, private::Bool = false) =
    private ? (fieldnames(typeof(op))..., :coefficients) :
    (:coefficients, :storage, :transposed, :conjugated)
function _operator_coerce(op::FaerLinearOperator{T}, ::Type{T}) where {T}
    _handle(op)
    return op
end
function _operator_coerce(op::FaerLinearOperator, ::Type{T}) where {T<:_NativeScalar}
    coefficients=_operator_coefficients(op)
    converted=op.storage===:identity ? FaerLinearOperator(LinearAlgebra.I, size(op, 1); T) :
              op.storage===:dense ? FaerLinearOperator(FaerMatrix{T}(coefficients)) :
              op.storage in (:csc, :csr) ?
              FaerLinearOperator(FaerSparseMatrix{T}(coefficients; format = op.storage)) :
              FaerLinearOperator(FaerDiagonal{T}(Vector(coefficients)))
    return op.transposed || op.conjugated ?
           _operator_copy(converted, op.transposed, op.conjugated) : converted
end
const _OperatorMatrix=_HouseholderMatrix
const _OperatorVector=_HouseholderVector
function _operator_product(
    op::FaerLinearOperator,
    a::Union{_OperatorMatrix,_OperatorVector},
)
    lock(_matrix_lock) do
        _handle(op)
        a isa _RotationArray && _rotation_open(a)
        size(op, 2)==size(a, 1) ||
            throw(DimensionMismatch("operator and RHS dimensions must match"))
        P=promote_type(eltype(op), _reflector_type(eltype(a)))
        rhs=_householder_matrix_source(a, P)
        dims=(size(op, 1), size(rhs, 2))
        args=(_operator_coerce(op, P), rhs)
        result=P===Float64 ? _matrix_result(RustCrate.operator_product, dims, args...) :
               _typed_result(P, RustCrate.operator_product, dims, args...)
        return a isa _BorrowVector ? _vector_from_source(result) :
               a isa AbstractVector ? vec(Matrix(result)) : result
    end
end
Base.:*(op::FaerLinearOperator, a::Union{_OperatorMatrix,_OperatorVector}) =
    _operator_product(op, a)
function Base.:*(a::_OperatorMatrix, op::FaerLinearOperator)
    lock(_matrix_lock) do
        work=_householder_matrix_source(
            a,
            promote_type(eltype(op), _reflector_type(eltype(a))),
        )
        return copy(adjoint(op'*copy(adjoint(work))))
    end
end
function FaerMatrix(op::FaerLinearOperator{T}) where {T}
    _handle(op)
    return op*FaerMatrix{T}(Matrix{T}(LinearAlgebra.I, size(op, 2), size(op, 2)))
end
Base.Matrix(op::FaerLinearOperator) = Matrix(FaerMatrix(op))
Base.Matrix{T}(op::FaerLinearOperator) where {T} = Matrix{T}(Matrix(op))
function Base.getindex(op::FaerLinearOperator{T}, i::Int, j::Int) where {T}
    checkbounds(op, i, j)
    unit=zeros(T, size(op, 2))
    unit[j]=1
    return (op*unit)[i]
end
function _operator_copy_result!(dest, result)
    T=eltype(dest)
    T<:_NativeScalar ||
        throw(ArgumentError("operator destinations require native scalar types"))
    buffer=ndims(dest)==1 ? Vector{T}(result) : Matrix{T}(result)
    copyto!(dest, buffer)
    return dest
end
function _operator_mul!(dest, op, a, alpha, beta, right)
    lock(_matrix_lock) do
        size(dest)==(
            ndims(a)==1 ? (size(op, 1),) :
            right ? (size(a, 1), size(op, 2)) : (size(op, 1), size(a, 2))
        ) || throw(DimensionMismatch("operator product destination shape mismatch"))
        dest isa _RotationArray && _rotation_open(dest)
        _handle(op)
        a isa _RotationArray && _rotation_open(a)
        (right ? size(a, 2)==size(op, 1) : size(a, 1)==size(op, 2)) ||
            throw(DimensionMismatch("operator/source product dimensions must match"))
        P=promote_type(
            eltype(op),
            _reflector_type(eltype(a)),
            _reflector_type(eltype(dest)),
            typeof(alpha),
            typeof(beta),
        )
        P<:_NativeScalar ||
            throw(ArgumentError("operator mul! requires native working scalar types"))
        scaled=if iszero(alpha)
            zeros(P, size(dest))
        else
            source=_householder_matrix_source(a, P)
            working=_operator_coerce(op, P)
            result=right ? source*working : working*source
            alpha .* (ndims(dest)==1 ? vec(Matrix(result)) : Matrix(result))
        end
        !iszero(beta) &&
            (scaled=scaled+beta .* (ndims(dest)==1 ? Vector(dest) : Matrix(dest)))
        return _operator_copy_result!(dest, scaled)
    end
end
LinearAlgebra.mul!(
    dest::_OperatorMatrix,
    op::FaerLinearOperator,
    a::_OperatorMatrix,
    alpha::Number,
    beta::Number,
) = _operator_mul!(dest, op, a, alpha, beta, false)
LinearAlgebra.mul!(
    dest::_OperatorVector,
    op::FaerLinearOperator,
    a::_OperatorVector,
    alpha::Number,
    beta::Number,
) = _operator_mul!(dest, op, a, alpha, beta, false)
LinearAlgebra.mul!(
    dest::_OperatorMatrix,
    a::_OperatorMatrix,
    op::FaerLinearOperator,
    alpha::Number,
    beta::Number,
) = _operator_mul!(dest, op, a, alpha, beta, true)
LinearAlgebra.mul!(dest::_OperatorMatrix, op::FaerLinearOperator, a::_OperatorMatrix) =
    _operator_mul!(dest, op, a, 1, 0, false)
LinearAlgebra.mul!(dest::_OperatorVector, op::FaerLinearOperator, a::_OperatorVector) =
    _operator_mul!(dest, op, a, 1, 0, false)
LinearAlgebra.mul!(dest::_OperatorMatrix, a::_OperatorMatrix, op::FaerLinearOperator) =
    _operator_mul!(dest, op, a, 1, 0, true)
function LinearAlgebra.lmul!(
    op::FaerLinearOperator,
    a::Union{FaerMatrix,FaerVector,FaerRow},
)
    lock(_matrix_lock) do
        _handle(op)
        _rotation_open(a)
        size(op, 1)==size(op, 2)==size(a, 1) ||
            throw(DimensionMismatch("preconditioner dimensions must match"))
        _native_call(RustCrate.operator_precondition, a, _operator_coerce(op, eltype(a)))
        return a
    end
end
function LinearAlgebra.lmul!(
    op::FaerLinearOperator,
    a::Union{FaerView,StridedVector,StridedMatrix},
)
    lock(_matrix_lock) do
        a isa FaerView && _rotation_open(a)
        size(op, 1)==size(op, 2) ||
            throw(DimensionMismatch("in-place preconditioner must be square"))
        return _operator_copy_result!(a, op*a)
    end
end
LinearAlgebra.rmul!(a::_OperatorMatrix, op::FaerLinearOperator) =
    _operator_copy_result!(a, a*op)
"""Create an implicit native identity preconditioner (an action approximating inv(A), not A)."""
identity_preconditioner(n::Integer; T::Type = Float64) =
    FaerLinearOperator(LinearAlgebra.I, n; T)
"""
    diagonal_preconditioner(d)

Own the inverse-diagonal multiplication action for Jacobi preconditioning.
Entries d must be finite and nonzero, and their reciprocals must be finite.
CG requires the resulting preconditioner action to be Hermitian positive definite.
"""
function diagonal_preconditioner(d::AbstractVector)
    T=_reflector_type(eltype(d))
    values=Vector{T}(d)
    all(x->isfinite(x)&&!iszero(x), values) ||
        throw(ArgumentError("preconditioner diagonal entries must be finite and nonzero"))
    inverse=one(T) ./ values
    all(isfinite, inverse) ||
        throw(ArgumentError("preconditioner inverse entries must be finite"))
    return FaerLinearOperator(FaerDiagonal(inverse))
end
"""CG solution plus native termination, completed updates, true and recursively estimated Frobenius residuals."""
struct FaerCgResult{X,R}
    x::X
    converged::Bool
    iterations::Int
    abs_residual::R
    rel_residual::R
    estimated_abs_residual::R
    estimated_rel_residual::R
    reason::Symbol
end
"""CG failed; `.result` retains its last iterate and diagnostics."""
struct FaerCgError{R} <: Exception
    result::R
end
Base.showerror(io::IO, e::FaerCgError) = print(
    io,
    "CG stopped: ",
    e.result.reason,
    " after ",
    e.result.iterations,
    " updates (true residual ",
    e.result.abs_residual,
    ")",
)
LinearAlgebra.issuccess(result::FaerCgResult) = result.converged
function _cg_matrix(a, ::Type{T}) where {T}
    a isa _RotationArray && _rotation_open(a)
    result=_householder_matrix_source(a, T)
    all(isfinite, Matrix(result)) ||
        throw(ArgumentError("CG RHS/initial guess must be finite"))
    return result
end
"""
    cg(A, b; M=nothing, x0=nothing, atol=0, rtol=nothing,
       maxiter=max(1,size(A,1)), check=true)

Call faer's preconditioned block conjugate-gradient solver. A must be Hermitian
positive definite, and M (if provided) must apply a Hermitian positive definite
approximate inverse. Dense/sparse/diagonal/identity native operators are accepted;
matrices are copied to an operator snapshot. Julia callbacks are not yet supported.
Returns FaerCgResult; x is a native vector for native vector RHS, a Julia vector
for Julia vector RHS, or a native matrix for matrix RHS. Multiple RHS use faer's
block algorithm. Inputs are unchanged. All native scalar types/promotions apply.
Default rtol is 128eps of the working real type. Residuals are Frobenius norms.
A true residual is recomputed independently; reported convergence is rejected if
it exceeds the requested tolerance. `check=true` throws FaerCgError with the
last result for nonconvergence or detected nonpositive operators/preconditioners.
These diagnostics do not certify that A or M is positive definite.
"""
function cg(
    A,
    b::Union{AbstractVector{<:Number},AbstractMatrix{<:Number}};
    M = nothing,
    x0 = nothing,
    atol::Real = 0,
    rtol = nothing,
    maxiter::Integer = max(1, size(A, 1)),
    check::Bool = true,
)
    lock(_matrix_lock) do
        op=A isa FaerLinearOperator ? A : FaerLinearOperator(A)
        _handle(op)
        size(op, 1)==size(op, 2) || throw(DimensionMismatch("CG operator must be square"))
        size(b, 1)==size(op, 1) ||
            throw(DimensionMismatch("CG RHS rows must match operator"))
        precond=isnothing(M) ? nothing :
                M isa FaerLinearOperator ? M : FaerLinearOperator(M)
        !isnothing(precond) && (
            _handle(precond);
            size(precond)==size(op) ||
                throw(DimensionMismatch("CG preconditioner shape must match operator"))
        )
        !isnothing(x0) &&
            size(x0)!=size(b) &&
            throw(DimensionMismatch("CG initial guess shape must match RHS"))
        P=promote_type(
            eltype(op),
            _reflector_type(eltype(b)),
            isnothing(x0) ? eltype(op) : _reflector_type(eltype(x0)),
            isnothing(precond) ? eltype(op) : eltype(precond),
        )
        R=_realtype(P)
        abs_tol=R(atol)
        rel_tol=isnothing(rtol) ? R(128)*eps(R) : R(rtol)
        isfinite(abs_tol)&&isfinite(rel_tol)&&abs_tol>=0&&rel_tol>=0 ||
            throw(ArgumentError("CG tolerances must be finite and nonnegative"))
        limit=Int(maxiter)
        limit>=0 || throw(ArgumentError("CG maxiter must be nonnegative"))
        rhs=_cg_matrix(b, P)
        initial=isnothing(x0) ? _zeros(P, size(rhs)) : _cg_matrix(x0, P)
        operator=_operator_coerce(op, P)
        preconditioner=isnothing(precond) ? Ptr{UInt8}(C_NULL) :
                       _operator_coerce(precond, P)
        stats=Vector{P}(undef, 5)
        iterations=Ref{UInt}(0)
        args=(
            operator,
            preconditioner,
            rhs,
            initial,
            Float64(abs_tol),
            Float64(rel_tol),
            UInt(limit),
            Ptr{UInt8}(pointer(stats)),
            Base.unsafe_convert(Ptr{UInt}, iterations),
        )
        solution=GC.@preserve stats iterations (
            P===Float64 ? _matrix_result(RustCrate.operator_cg, size(rhs), args...) :
            _typed_result(P, RustCrate.operator_cg, size(rhs), args...)
        )
        status=Int(real(stats[1]))
        x=b isa _BorrowVector ? _vector_from_source(solution) :
          b isa AbstractVector ? vec(Matrix(solution)) : solution
        reason=(
            :converged,
            :no_convergence,
            :nonpositive_operator,
            :nonpositive_preconditioner,
            :residual_mismatch,
        )[status+1]
        result=FaerCgResult(
            x,
            status==0,
            Int(iterations[]),
            R(real(stats[2])),
            R(real(stats[3])),
            R(real(stats[4])),
            R(real(stats[5])),
            reason,
        )
        check && !result.converged && throw(FaerCgError(result))
        return result
    end
end
"""
    cg!(x, A, b; kwargs...)

Use x as the initial guess and replace it after a successful solve. With
check=false, copy the last iterate even on nonconvergence. The destination
scalar type must equal the promoted solver type, and the full result is staged
before mutation. Returns FaerCgResult whose x is the destination itself.
"""
function cg!(x::_HouseholderArray, A, b; check::Bool = true, kwargs...)
    lock(_matrix_lock) do
        size(x)==size(b) || throw(DimensionMismatch("CG destination shape must match RHS"))
        x isa _RotationArray && _rotation_open(x)
        result=cg(A, b; x0 = x, check, kwargs...)
        eltype(x)===eltype(result.x) || throw(
            ArgumentError("cg! destination must match the promoted solver scalar type"),
        )
        _operator_copy_result!(x, result.x)
        return FaerCgResult(
            x,
            result.converged,
            result.iterations,
            result.abs_residual,
            result.rel_residual,
            result.estimated_abs_residual,
            result.estimated_rel_residual,
            result.reason,
        )
    end
end
