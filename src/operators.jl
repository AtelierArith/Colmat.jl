"""
    FaerLinearOperator(A)
    FaerLinearOperator(I, n; T=Float64)
    FaerLinearOperator(apply!, adjoint_apply!, m, n; T=Float64)

Own an independent faer LinOp/BiLinOp snapshot of a dense matrix, native/Julia
CSC or native CSR sparse matrix, or diagonal matrix. Identity and diagonal
operators do not store a dense coefficient matrix. `*`, `mul!`, conjugation,
transpose and adjoint call native operator traits; Matrix(op) explicitly forms
its dense matrix. `.coefficients` returns an independent base representation,
with `.transposed/.conjugated` describing its orientation.
Native composition is available with operator * operator. Application supports
`mul!(X, op, B[, alpha,beta]; scratch,team)` and explicit `*(op,B; scratch,team)`
with scratch queries/reserve, including prepared owned/wide sparse-QR inverse
actions. Julia callback operators retain function state by reference and marshal
independent Julia arrays around native trait applications.
"""
mutable struct FaerLinearOperator{T,H} <: AbstractMatrix{T}
    handle::Union{Nothing,H}
    dims::Tuple{Int,Int}
    storage::Symbol
    transposed::Bool
    conjugated::Bool
    callbacks::Union{Nothing,Dict{UInt,Any}}
end
FaerLinearOperator{T,H}(h, dims, storage, tr, cj) where {T,H} =
    FaerLinearOperator{T,H}(h, dims, storage, tr, cj, nothing)
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
        op.callbacks=nothing
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
            op.callbacks,
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
        op.storage in (:composition, :factorization, :callback) && throw(
            ArgumentError("operator has no single coefficient storage; use Matrix(op)"),
        )
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
    _handle(op)
    eltype(op)===T && return op
    h=RustCrate.OwnedLinearOperator()
    _native_call(
        RustCrate.operator_promote,
        h,
        op,
        _tag(T),
        UInt(size(op, 1)),
        UInt(size(op, 2)),
        _tag(eltype(op)),
    )
    return FaerLinearOperator{T,typeof(h)}(
        h,
        size(op),
        op.storage,
        op.transposed,
        op.conjugated,
        op.callbacks,
    )
end
const _OperatorMatrix=_HouseholderMatrix
const _OperatorVector=_HouseholderVector
function _operator_product(
    op::FaerLinearOperator,
    a::Union{_OperatorMatrix,_OperatorVector};
    scratch = nothing,
    team = nothing,
)
    lock(_matrix_lock) do
        _handle(op)
        a isa _RotationArray && _rotation_open(a)
        size(op, 2)==size(a, 1) ||
            throw(DimensionMismatch("operator and RHS dimensions must match"))
        P=promote_type(eltype(op), _reflector_type(eltype(a)))
        rhs=_householder_matrix_source(a, P)
        dims=(size(op, 1), size(rhs, 2))
        working=_operator_coerce(op, P)
        args=(
            working,
            rhs,
            UInt(size(op, 1)),
            UInt(size(op, 2)),
            UInt(size(rhs, 2)),
            _tag(P),
        )
        result=if isnothing(scratch) && isnothing(team)
            P===Float64 ? _matrix_result(RustCrate.operator_product, dims, args...) :
            _typed_result(P, RustCrate.operator_product, dims, args...)
        else
            _operator_workspace_result(working, rhs; scratch, team)
        end
        return a isa _BorrowVector ? _vector_from_source(result) :
               a isa AbstractVector ? vec(Matrix(result)) : result
    end
end
Base.:*(
    op::FaerLinearOperator,
    a::Union{_OperatorMatrix,_OperatorVector};
    scratch = nothing,
    team = nothing,
) = _operator_product(op, a; scratch, team)
function Base.:*(
    a::_OperatorMatrix,
    op::FaerLinearOperator;
    scratch = nothing,
    team = nothing,
)
    lock(_matrix_lock) do
        work=_householder_matrix_source(
            a,
            promote_type(eltype(op), _reflector_type(eltype(a))),
        )
        return copy(adjoint(_operator_product(op', copy(adjoint(work)); scratch, team)))
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
    # Native results are assigned natively, also straight into Julia memory.
    if result isa Union{_BorrowMatrix,_BorrowVector} &&
       eltype(result)===T &&
       (
           dest isa Union{_BorrowMatrix,_BorrowVector} ||
           dest isa Union{FaerBorrowedArray,StridedArray} && size(result)==size(dest)
       )
        return copyto!(dest, result)
    end
    buffer=ndims(dest)==1 ? Vector{T}(result) : Matrix{T}(result)
    copyto!(dest, buffer)
    return dest
end
function _operator_mul_prepare(dest, op, a, alpha, beta, right)
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
    return P
end
function _operator_mul!(dest, op, a, alpha, beta, right; scratch = nothing, team = nothing)
    lock(_matrix_lock) do
        P=_operator_mul_prepare(dest, op, a, alpha, beta, right)
        if iszero(alpha) && (!isnothing(scratch) || !isnothing(team))
            working=_operator_coerce(op, P)
            effective=right ? adjoint(working) : working
            columns=right ? size(a, 1) : ndims(a)==1 ? 1 : size(a, 2)
            _with_operator_workspace(effective, columns; scratch, team) do _
                nothing
            end
        end
        scaled=if iszero(alpha)
            zeros(P, size(dest))
        else
            source=_householder_matrix_source(a, P)
            working=_operator_coerce(op, P)
            result=right ? *(source, working; scratch, team) :
                   _operator_product(working, source; scratch, team)
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
    beta::Number;
    scratch = nothing,
    team = nothing,
) = _operator_mul!(dest, op, a, alpha, beta, false; scratch, team)
LinearAlgebra.mul!(
    dest::_OperatorVector,
    op::FaerLinearOperator,
    a::_OperatorVector,
    alpha::Number,
    beta::Number;
    scratch = nothing,
    team = nothing,
) = _operator_mul!(dest, op, a, alpha, beta, false; scratch, team)
LinearAlgebra.mul!(
    dest::_OperatorMatrix,
    a::_OperatorMatrix,
    op::FaerLinearOperator,
    alpha::Number,
    beta::Number;
    scratch = nothing,
    team = nothing,
) = _operator_mul!(dest, op, a, alpha, beta, true; scratch, team)
LinearAlgebra.mul!(
    dest::_OperatorMatrix,
    op::FaerLinearOperator,
    a::_OperatorMatrix;
    scratch = nothing,
    team = nothing,
) = _operator_mul!(dest, op, a, 1, 0, false; scratch, team)
LinearAlgebra.mul!(
    dest::_OperatorVector,
    op::FaerLinearOperator,
    a::_OperatorVector;
    scratch = nothing,
    team = nothing,
) = _operator_mul!(dest, op, a, 1, 0, false; scratch, team)
LinearAlgebra.mul!(
    dest::_OperatorMatrix,
    a::_OperatorMatrix,
    op::FaerLinearOperator;
    scratch = nothing,
    team = nothing,
) = _operator_mul!(dest, op, a, 1, 0, true; scratch, team)
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
    history::Vector{_IterationEntry{R}}
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
# Native inputs are copied once into the native solver operand and checked
# natively; Julia arrays are checked in place. No Julia staging copy is made.
function _cg_matrix(a, ::Type{T}) where {T}
    _dense_rhs_validate(a)
    a isa _RotationArray && _rotation_open(a)
    native=a isa Union{_BorrowMatrix,_BorrowVector}
    native ||
        all(isfinite, a) ||
        throw(ArgumentError("iterative solver RHS/initial guess must be finite"))
    m=_householder_matrix_source(a, T)
    if native && !all(isfinite, m)
        close(m)
        throw(ArgumentError("iterative solver RHS/initial guess must be finite"))
    end
    return m
end
"""
    cg(A, b; M=nothing, x0=nothing, atol=0, rtol=nothing,
       maxiter=max(1,size(A,1)), check=true, callback=nothing, history=false,
       scratch=nothing, team=nothing)

Call faer's preconditioned block conjugate-gradient solver. A must be Hermitian
positive definite, and M (if provided) must apply a Hermitian positive definite
approximate inverse. Dense/sparse/diagonal/identity native operators are accepted;
matrices are copied to an operator snapshot. Julia callback operators and
preconditioners are accepted; iteration callbacks and histories are supported
as described below.
Returns FaerCgResult; x is a native vector for native vector RHS, a Julia vector
for Julia vector RHS, or a native matrix for matrix RHS. Multiple RHS use faer's
block algorithm. Inputs are unchanged. All native scalar types/promotions apply.
Default rtol is 128eps of the working real type. Residuals are Frobenius norms.
A true residual is recomputed independently; reported convergence is rejected if
it exceeds the requested tolerance. `check=true` throws FaerCgError with the
last result for nonconvergence or detected nonpositive operators/preconditioners.
These diagnostics do not certify that A or M is positive definite.

`scratch` reuses caller-owned native iteration and final residual workspace;
`team` supplies the scoped native parallel team. Query with
`scratch_requirements(cg, A, b; M, x0, team)` and grow with `reserve!`.
Native RHS, initial guesses and in-place destinations are not staged in Julia:
they are copied once into the native solver operands and checked natively, and
`cg!` writes native destinations natively. Julia RHS are copied straight into
native storage; a Julia vector RHS returns a Julia vector. Scalar promotion and
Julia callback/history buffers allocate separately.
Queries do not apply operators or invoke callbacks. BiCGSTAB and LSMR follow
the same resource contract.

`history=true` retains a residual-only record after each native callback update.
`callback(state)` receives `iteration`, an independent Julia array `x`, true
`abs_residual`/`rel_residual`, and `normal_abs_residual`/`normal_rel_residual`
(`nothing` for CG/BiCGSTAB; preconditioned normal residuals for LSMR). Return
`nothing` or `true` to continue, `false` to stop. A requested stop returns
`reason=:stopped`, `converged=false` even with `check=true`; in-place solvers
copy that last iterate. Callback exceptions propagate after native return and
leave in-place destinations unchanged. Initial convergence invokes no callback.

Observation copies each iterate and recomputes residuals with extra native
operator applications; it allocates. History stores the diagnostic fields,
excluding `x`. With both options disabled, no iterate snapshot or extra operator
application is made. Callbacks run synchronously on the calling Julia task;
they may retain/mutate their copy of `x` and make nested Colmat calls.

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
    callback = nothing,
    history::Bool = false,
    scratch = nothing,
    team = nothing,
)
    lock(_matrix_lock) do
        _iterative_resource_check(scratch, team)
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
        resources=_iterative_resource_prepare(
            0,
            operator,
            preconditioner,
            nothing,
            size(rhs, 2),
            scratch,
            team,
        )
        stats=Vector{P}(undef, 5)
        iterations=Ref{UInt}(0)
        log, observing, observer, normalop=_iteration_observer(
            P,
            operator,
            rhs,
            callback,
            history;
            vector = b isa AbstractVector,
        )
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
            observing,
            observer,
        )
        solution=try
            GC.@preserve stats iterations _iterative_solution(
                resources,
                0,
                P,
                RustCrate.operator_cg,
                size(rhs),
                args,
                scratch,
                team,
            )
        finally
            isnothing(normalop) || close(normalop)
        end
        status=Int(real(stats[1]))
        x=b isa _BorrowVector ? _vector_from_source(solution) :
          b isa AbstractVector ? vec(Matrix(solution)) : solution
        reason=(
            :converged,
            :no_convergence,
            :nonpositive_operator,
            :nonpositive_preconditioner,
            :residual_mismatch,
            :stopped,
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
            log,
        )
        check && !result.converged && result.reason!==:stopped && throw(FaerCgError(result))
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
            result.history,
        )
    end
end

"""BiCGSTAB solution and independently verified residual diagnostics."""
struct FaerBicgstabResult{X,R}
    x::X
    converged::Bool
    iterations::Int
    abs_residual::R
    rel_residual::R
    estimated_abs_residual::R
    estimated_rel_residual::R
    reason::Symbol
    history::Vector{_IterationEntry{R}}
end
"""BiCGSTAB failed; `.result` retains its last iterate and diagnostics."""
struct FaerBicgstabError{R} <: Exception
    result::R
end
Base.showerror(io::IO, e::FaerBicgstabError) = print(
    io,
    "BiCGSTAB stopped: ",
    e.result.reason,
    " after ",
    e.result.iterations,
    " updates (true residual ",
    e.result.abs_residual,
    ")",
)
LinearAlgebra.issuccess(result::FaerBicgstabResult) = result.converged

"""
    bicgstab(A, b; M=nothing, N=nothing, x0=nothing, atol=0,
             rtol=nothing, maxiter=max(1,size(A,1)), check=true, callback=nothing, history=false,
       scratch=nothing, team=nothing)

Run faer's block BiCGSTAB for a square, possibly nonsymmetric operator.
M and N apply left and right approximate inverses, respectively. Native
operators preserve sparse/diagonal/identity storage. Scalar promotion, output
ownership, tolerances and true-residual verification follow [`cg`](@ref).
Returns `FaerBicgstabResult`; checked failures throw `FaerBicgstabError` retaining
the last iterate. No Hermitian or positive definite assumption is required.
"""
function bicgstab(
    A,
    b::Union{AbstractVector{<:Number},AbstractMatrix{<:Number}};
    M = nothing,
    N = nothing,
    x0 = nothing,
    atol::Real = 0,
    rtol = nothing,
    maxiter::Integer = max(1, size(A, 1)),
    check::Bool = true,
    callback = nothing,
    history::Bool = false,
    scratch = nothing,
    team = nothing,
)
    lock(_matrix_lock) do
        _iterative_resource_check(scratch, team)
        op=A isa FaerLinearOperator ? A : FaerLinearOperator(A)
        _handle(op)
        size(op, 1)==size(op, 2) ||
            throw(DimensionMismatch("BiCGSTAB operator must be square"))
        size(b, 1)==size(op, 1) ||
            throw(DimensionMismatch("BiCGSTAB RHS rows must match operator"))
        precond=isnothing(M) ? nothing :
                M isa FaerLinearOperator ? M : FaerLinearOperator(M)
        !isnothing(precond) && (
            _handle(precond);
            size(precond)==size(op) || throw(
                DimensionMismatch("BiCGSTAB preconditioner shape must match operator"),
            )
        )
        right=isnothing(N) ? nothing : N isa FaerLinearOperator ? N : FaerLinearOperator(N)
        !isnothing(right) && (
            _handle(right);
            size(right)==size(op) || throw(
                DimensionMismatch(
                    "BiCGSTAB right preconditioner shape must match operator",
                ),
            )
        )
        !isnothing(x0) &&
            size(x0)!=size(b) &&
            throw(DimensionMismatch("BiCGSTAB initial guess shape must match RHS"))
        P=promote_type(
            eltype(op),
            _reflector_type(eltype(b)),
            isnothing(x0) ? eltype(op) : _reflector_type(eltype(x0)),
            isnothing(precond) ? eltype(op) : eltype(precond),
            isnothing(right) ? eltype(op) : eltype(right),
        )
        R=_realtype(P)
        abs_tol=R(atol)
        rel_tol=isnothing(rtol) ? R(128)*eps(R) : R(rtol)
        isfinite(abs_tol)&&isfinite(rel_tol)&&abs_tol>=0&&rel_tol>=0 ||
            throw(ArgumentError("BiCGSTAB tolerances must be finite and nonnegative"))
        limit=Int(maxiter)
        limit>=0 || throw(ArgumentError("BiCGSTAB maxiter must be nonnegative"))
        rhs=_cg_matrix(b, P)
        initial=isnothing(x0) ? _zeros(P, size(rhs)) : _cg_matrix(x0, P)
        operator=_operator_coerce(op, P)
        preconditioner=isnothing(precond) ? Ptr{UInt8}(C_NULL) :
                       _operator_coerce(precond, P)
        right_preconditioner=isnothing(right) ? Ptr{UInt8}(C_NULL) :
                             _operator_coerce(right, P)
        resources=_iterative_resource_prepare(
            1,
            operator,
            preconditioner,
            right_preconditioner,
            size(rhs, 2),
            scratch,
            team,
        )
        stats=Vector{P}(undef, 5)
        iterations=Ref{UInt}(0)
        log, observing, observer, normalop=_iteration_observer(
            P,
            operator,
            rhs,
            callback,
            history;
            vector = b isa AbstractVector,
        )
        args=(
            operator,
            preconditioner,
            right_preconditioner,
            rhs,
            initial,
            Float64(abs_tol),
            Float64(rel_tol),
            UInt(limit),
            Ptr{UInt8}(pointer(stats)),
            Base.unsafe_convert(Ptr{UInt}, iterations),
            observing,
            observer,
        )
        solution=try
            GC.@preserve stats iterations _iterative_solution(
                resources,
                1,
                P,
                RustCrate.operator_bicgstab,
                size(rhs),
                args,
                scratch,
                team,
            )
        finally
            isnothing(normalop) || close(normalop)
        end
        status=Int(real(stats[1]))
        x=b isa _BorrowVector ? _vector_from_source(solution) :
          b isa AbstractVector ? vec(Matrix(solution)) : solution
        reason=(
            :converged,
            :no_convergence,
            :nonpositive_operator,
            :nonpositive_preconditioner,
            :residual_mismatch,
            :stopped,
        )[status+1]
        result=FaerBicgstabResult(
            x,
            status==0,
            Int(iterations[]),
            R(real(stats[2])),
            R(real(stats[3])),
            R(real(stats[4])),
            R(real(stats[5])),
            reason,
            log,
        )
        check &&
            !result.converged &&
            result.reason!==:stopped &&
            throw(FaerBicgstabError(result))
        return result
    end
end

"""Use x as the initial guess for BiCGSTAB; checked failures leave x unchanged. See [`cg!`](@ref)."""
function bicgstab!(x::_HouseholderArray, A, b; check::Bool = true, kwargs...)
    lock(_matrix_lock) do
        size(x)==size(b) ||
            throw(DimensionMismatch("BiCGSTAB destination shape must match RHS"))
        x isa _RotationArray && _rotation_open(x)
        result=bicgstab(A, b; x0 = x, check, kwargs...)
        eltype(x)===eltype(result.x) || throw(
            ArgumentError(
                "bicgstab! destination must match the promoted solver scalar type",
            ),
        )
        _operator_copy_result!(x, result.x)
        return FaerBicgstabResult(
            x,
            result.converged,
            result.iterations,
            result.abs_residual,
            result.rel_residual,
            result.estimated_abs_residual,
            result.estimated_rel_residual,
            result.reason,
            result.history,
        )
    end
end

"""LSMR solution, primal residuals and independently verified preconditioned normal residuals."""
struct FaerLsmrResult{X,R}
    x::X
    converged::Bool
    iterations::Int
    abs_residual::R
    rel_residual::R
    normal_abs_residual::R
    normal_rel_residual::R
    estimated_abs_residual::R
    estimated_rel_residual::R
    reason::Symbol
    history::Vector{_IterationEntry{R}}
end
"""LSMR failed; `.result` retains its last iterate and diagnostics."""
struct FaerLsmrError{R} <: Exception
    result::R
end
Base.showerror(io::IO, e::FaerLsmrError) = print(
    io,
    "LSMR stopped: ",
    e.result.reason,
    " after ",
    e.result.iterations,
    " updates (true residual ",
    e.result.abs_residual,
    ")",
)
LinearAlgebra.issuccess(result::FaerLsmrResult) = result.converged

"""
    lsmr(A, b; M=nothing, x0=nothing, atol=0, rtol=nothing,
         maxiter=max(1,4min(size(A)...)), check=true, callback=nothing, history=false,
       scratch=nothing, team=nothing)

Run faer's block LSMR for rectangular least squares. M applies a square right
preconditioner of size size(A,2). It must be nonsingular and support its
adjoint action to preserve the original least squares problem.
Convergence uses the Frobenius norm of `M' * A' * (b-A*x)`, with threshold
`max(atol, rtol * norm(M' * A' * b))`; this is independently recomputed.
`abs_residual`/`rel_residual` describe the primal residual, while
`normal_abs_residual`/`normal_rel_residual` describe the preconditioned normal
residual. Native estimates are normal residual estimates. Default rtol is
128eps of the promoted real type. Absolute and relative tolerances are passed
independently to the native solver.

Returns `FaerLsmrResult`, with solution rows equal to size(A,2). Checked failures
throw `FaerLsmrError` retaining the last iterate. Zero initial guesses give the
minimum-norm solution when the unpreconditioned method converges; arbitrary
initial guesses may retain nullspace components. Right preconditioning changes
the minimum-norm metric. Storage, scalar promotion and ownership follow [`cg`](@ref).
"""
function lsmr(
    A,
    b::Union{AbstractVector{<:Number},AbstractMatrix{<:Number}};
    M = nothing,
    x0 = nothing,
    atol::Real = 0,
    rtol = nothing,
    maxiter::Integer = max(1, 4 * min(size(A)...)),
    check::Bool = true,
    callback = nothing,
    history::Bool = false,
    scratch = nothing,
    team = nothing,
)
    lock(_matrix_lock) do
        _iterative_resource_check(scratch, team)
        op=A isa FaerLinearOperator ? A : FaerLinearOperator(A)
        _handle(op)
        size(b, 1)==size(op, 1) ||
            throw(DimensionMismatch("LSMR RHS rows must match operator"))
        precond=isnothing(M) ? nothing :
                M isa FaerLinearOperator ? M : FaerLinearOperator(M)
        !isnothing(precond) && (
            _handle(precond);
            size(precond)==(size(op, 2), size(op, 2)) || throw(
                DimensionMismatch("LSMR preconditioner shape must match operator"),
            )
        )
        !isnothing(x0) &&
            size(x0)!=(b isa AbstractVector ? (size(op, 2),) : (size(op, 2), size(b, 2))) &&
            throw(DimensionMismatch("LSMR initial guess shape must match RHS"))
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
            throw(ArgumentError("LSMR tolerances must be finite and nonnegative"))
        limit=Int(maxiter)
        limit>=0 || throw(ArgumentError("LSMR maxiter must be nonnegative"))
        rhs=_cg_matrix(b, P)
        dims=(size(op, 2), size(rhs, 2))
        initial=isnothing(x0) ? _zeros(P, dims) : _cg_matrix(x0, P)
        operator=_operator_coerce(op, P)
        preconditioner=isnothing(precond) ? Ptr{UInt8}(C_NULL) :
                       _operator_coerce(precond, P)
        resources=_iterative_resource_prepare(
            2,
            operator,
            preconditioner,
            nothing,
            size(rhs, 2),
            scratch,
            team,
        )
        stats=Vector{P}(undef, 7)
        iterations=Ref{UInt}(0)
        log, observing, observer, normalop=_iteration_observer(
            P,
            operator,
            rhs,
            callback,
            history;
            vector = b isa AbstractVector,
            normal = true,
            preconditioner = isnothing(precond) ? nothing : preconditioner,
        )
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
            observing,
            observer,
        )
        solution=try
            GC.@preserve stats iterations _iterative_solution(
                resources,
                2,
                P,
                RustCrate.operator_lsmr,
                dims,
                args,
                scratch,
                team,
            )
        finally
            isnothing(normalop) || close(normalop)
        end
        status=Int(real(stats[1]))
        x=b isa _BorrowVector ? _vector_from_source(solution) :
          b isa AbstractVector ? vec(Matrix(solution)) : solution
        reason=(
            :converged,
            :no_convergence,
            :nonpositive_operator,
            :nonpositive_preconditioner,
            :residual_mismatch,
            :stopped,
        )[status+1]
        result=FaerLsmrResult(
            x,
            status==0,
            Int(iterations[]),
            R(real(stats[2])),
            R(real(stats[3])),
            R(real(stats[4])),
            R(real(stats[5])),
            R(real(stats[6])),
            R(real(stats[7])),
            reason,
            log,
        )
        check &&
            !result.converged &&
            result.reason!==:stopped &&
            throw(FaerLsmrError(result))
        return result
    end
end

"""Use x as LSMR initial guess/output; checked failure leaves x unchanged. See [`lsmr`](@ref)."""
function lsmr!(x::_HouseholderArray, A, b; check::Bool = true, kwargs...)
    lock(_matrix_lock) do
        size(x)==(b isa AbstractVector ? (size(A, 2),) : (size(A, 2), size(b, 2))) ||
            throw(DimensionMismatch("LSMR destination shape must match solution"))
        x isa _RotationArray && _rotation_open(x)
        result=lsmr(A, b; x0 = x, check, kwargs...)
        eltype(x)===eltype(result.x) || throw(
            ArgumentError("lsmr! destination must match the promoted solver scalar type"),
        )
        _operator_copy_result!(x, result.x)
        return FaerLsmrResult(
            x,
            result.converged,
            result.iterations,
            result.abs_residual,
            result.rel_residual,
            result.normal_abs_residual,
            result.normal_rel_residual,
            result.estimated_abs_residual,
            result.estimated_rel_residual,
            result.reason,
            result.history,
        )
    end
end

"""
    A::FaerLinearOperator * B::FaerLinearOperator

Compose independent native operators without forming their coefficient product.
The composition owns copies of both operands and supports native application,
transpose/adjoint and iterative solvers. Application uses an intermediate
RHS matrix in native scratch; coefficients are available through explicit `Matrix(op)` conversion.
"""
function Base.:*(a::FaerLinearOperator, b::FaerLinearOperator)
    lock(_matrix_lock) do
        _handle(a)
        _handle(b)
        size(a, 2)==size(b, 1) ||
            throw(DimensionMismatch("operator composition dimensions must match"))
        T=promote_type(eltype(a), eltype(b))
        left, right=_operator_coerce(a, T), _operator_coerce(b, T)
        h=RustCrate.OwnedLinearOperator()
        _native_call(RustCrate.operator_compose, h, left, right)
        FaerLinearOperator{T,typeof(h)}(
            h,
            (size(a, 1), size(b, 2)),
            :composition,
            false,
            false,
            _merge_operator_callbacks(left, right),
        )
    end
end

"""
    FaerLinearOperator(F::FaerFactorization{T,:lu})

Own an independent inverse-action operator from a square nonsingular native LU, QR (optionally pivoted), Cholesky or LDLT (optionally block pivoted) factorization.
Apply it as a preconditioner with `cg(A,b; M=FaerLinearOperator(F))` when that
inverse action is positive definite, or with BiCGSTAB for general systems. No
inverse matrix or new factorization is formed; each apply solves using saved
triangular factors and any pivots with native scratch. Mixed native scalar types promote saved factors without refactorization.
"""
FaerLinearOperator(f::FaerFactorization) = _dense_factor_operator(f)
function _dense_factor_operator(f::FaerFactorization{T,K}) where {T,K}
    K in (:lu, :full_lu, :qr, :pivoted_qr, :cholesky, :ldlt, :bunchkaufman) || throw(
        ArgumentError(
            "factor preconditioner requires LU, QR (optionally pivoted), Cholesky or LDLT (optionally block pivoted)",
        ),
    )
    lock(_matrix_lock) do
        _handle(f)
        size(f, 1)==size(f, 2) ||
            throw(DimensionMismatch("factor preconditioner must be square"))
        LinearAlgebra.issuccess(f) || (
            K===:cholesky ? throw(LinearAlgebra.PosDefException(f.info)) :
            throw(LinearAlgebra.SingularException(f.info))
        )
        _dense_solve_validate(f)
        h=RustCrate.OwnedLinearOperator()
        fun=T===Float64 ? RustCrate.operator_factor_preconditioner :
            RustCrate.operator_typed_factor_preconditioner
        _native_call(fun, h, f)
        FaerLinearOperator{T,typeof(h)}(h, size(f), :factorization, false, false)
    end
end

const _DenseQrFactor =
    Union{FaerFactorization{T,:qr},FaerFactorization{T,:pivoted_qr}} where {T}
"""
    FaerLinearOperator(F::FaerFactorization{T,:qr}; atol=0, rtol=nothing)

Build a saved-factor inverse/pseudoinverse action from QR, with or without
column pivoting. Rectangular QR uses a
minimum-norm action with cutoff `max(atol,rtol*maximum(svdvals(R)))` and a default
rtol of `min(size(F)...)*eps(real(T))` when atol is zero (otherwise zero).
Nonzero atol or explicit rtol also select this rank-truncated action for square QR.
Square QR without tolerances keeps its existing nonsingular inverse contract.

Construction computes the SVD of saved R in caller-owned `scratch` (queried by
`scratch_requirements(FaerLinearOperator, F; atol, rtol, team)`; a temporary
workspace when omitted) on the optional `team`, and retains only the singular
columns above the cutoff plus compact Householder factors and column
permutation. No dense Q, pseudoinverse of the original matrix or source
refactorization is formed. Applications use native scratch/team; the retained
factors and result staging allocate as operator output.
Promotion converts saved data and preserves the original cutoff decisions.
"""
function FaerLinearOperator(
    f::_DenseQrFactor;
    atol::Real = 0,
    rtol = nothing,
    scratch = nothing,
    team = nothing,
)
    lock(_matrix_lock) do
        _handle(f)
        at, rt=_spectral_tolerances(
            atol,
            isnothing(rtol) ? _default_spectral_rtol(f, atol) : rtol,
        )
        _construction_options(scratch, team)
        if size(f, 1)==size(f, 2) && iszero(atol) && isnothing(rtol)
            return _dense_factor_operator(f)
        end
        _dense_solve_validate(f)
        LinearAlgebra.issuccess(f) || throw(LinearAlgebra.SingularException(f.info))
        relative=isnothing(rtol) ? _default_spectral_rtol(f, atol) : rtol
        T=eltype(f)
        h=RustCrate.OwnedLinearOperator()
        fun=T===Float64 ? RustCrate.operator_qr_pseudoinverse :
            RustCrate.operator_qr_pseudoinverse_typed
        req=_qr_pseudoinverse_requirement(f, team)
        _with_construction_scratch(req, scratch) do work
            _native_call(
                fun,
                h,
                f,
                work,
                at,
                rt,
                promote_type(typeof(relative), _realtype(T))===Float32,
                UInt(size(f, 1)),
                UInt(size(f, 2)),
                _tag(T),
                team,
            )
        end
        return FaerLinearOperator{T,typeof(h)}(
            h,
            reverse(size(f)),
            :factorization,
            false,
            false,
        )
    end
end

"""
    FaerLinearOperator(F::FaerSparseFactorization)

Own a sparse inverse-action snapshot of LU/QR/Cholesky factors for every algorithm. Native
CSC L/U and row/column permutations are retained independently; mixed scalar
promotion converts these factors without refactorizing the stored input.
Application uses saved factors and sparse triangular solves. Prepared
auto/supernodal and wide QR actions use declared native scratch and caller teams.
Wide QR construction stores a dense pseudoinverse of R, computed from an SVD in
caller-owned `scratch` (`scratch_requirements(FaerLinearOperator, F; team)`) on
the optional `team`. Other kinds only copy factors and need zero bytes.
"""
function FaerLinearOperator(
    f::FaerSparseFactorization{T,K};
    scratch = nothing,
    team = nothing,
) where {T,K}
    K in (:lu, :cholesky, :qr) ||
        throw(ArgumentError("sparse inverse operator requires LU/QR or Cholesky"))
    lock(_matrix_lock) do
        _handle(f)
        _construction_options(scratch, team)
        LinearAlgebra.issuccess(f) || (
            K===:cholesky ? throw(LinearAlgebra.PosDefException(f.info)) :
            throw(LinearAlgebra.SingularException(f.info))
        )
        if K===:qr && _native_call(RustCrate.sparse_qr_needs_spectral_solve, f)
            inverse=LinearAlgebra.pinv(f)
            try
                return FaerLinearOperator(inverse)
            finally
                close(inverse)
            end
        end
        if K===:qr
            h=RustCrate.OwnedLinearOperator()
            req=_sparse_qr_inverse_requirement(f, team)
            _with_construction_scratch(req, scratch) do work
                _native_call(RustCrate.operator_sparse_qr_inverse, h, f, work, team)
            end
            return FaerLinearOperator{T,typeof(h)}(
                h,
                reverse(size(f)),
                :factorization,
                false,
                false,
            )
        end
        l, u, rows=f.L, f.U, f.p
        cols=K===:lu ? f.q : copy(rows)
        try
            h=RustCrate.OwnedLinearOperator()
            _native_call(
                RustCrate.operator_sparse_factor_inverse,
                h,
                l,
                u,
                rows,
                cols,
                K===:lu,
            )
            FaerLinearOperator{T,typeof(h)}(h, size(f), :factorization, false, false)
        finally
            close(l)
            close(u)
            close(rows)
            close(cols)
        end
    end
end
