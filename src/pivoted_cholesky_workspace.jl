const _PivotedBorrow = Union{FaerMatrix,FaerMatrixView}
const _PivotedInput = Union{
    _PivotedBorrow,
    LinearAlgebra.Hermitian{<:Any,<:_PivotedBorrow},
    LinearAlgebra.Symmetric{<:Any,<:_PivotedBorrow},
}
const _PivotedConstruction = Union{
    typeof(LinearAlgebra.cholesky),typeof(LinearAlgebra.cholesky!)
}
function _pivoted_input_plan(a, tol, block_size, team)
    _iterative_resource_check(nothing, team)
    source = a isa _PivotedBorrow ? a : parent(a)
    _dense_rhs_validate(source)
    n = LinearAlgebra.checksquare(a)
    a isa LinearAlgebra.Symmetric &&
        eltype(a) <: Complex &&
        throw(ArgumentError("complex symmetric input must be Hermitian"))
    isfinite(tol) || throw(ArgumentError("tol must be finite"))
    ftol = Float64(tol)
    ftol isa Float64 || throw(ArgumentError("tol conversion must produce Float64"))
    isfinite(ftol) || throw(ArgumentError("tol must be representable as a finite Float64"))
    1 <= block_size <= typemax(UInt) ||
        throw(ArgumentError("block_size must be between one and typemax(UInt)"))
    block = UInt(block_size)
    _dense_rhs_validate(source)
    size(a) == (n, n) ||
        throw(DimensionMismatch("pivoted Cholesky input changed shape during conversion"))
    return source, n, ftol, block
end
function _pivoted_construction_requirement(n, T, ftol, block, team)
    info = zeros(UInt, 2)
    _native_call(
        RustCrate.pivoted_llt_requirements, UInt(n), _tag(T), ftol, block, team, info
    )
    return (bytes=Int(info[1]), alignment=Int(info[2]))
end
"""
    scratch_requirements(cholesky, A, RowMaximum(); tol=0.0, block_size=128, team=nothing)
    scratch_requirements(cholesky!, A, RowMaximum(); tol=0.0, block_size=128, team=nothing)

Query pivoted Cholesky's declared temporary bytes/alignment. Use the same
options for computation and `FaerScratch`/`reserve!`. Automatic tolerance uses
faer's kernel requirement; explicit tolerance includes a Hermitian working
matrix in caller scratch. Returned factors and view copies allocate separately.
"""
function scratch_requirements(
    ::_PivotedConstruction,
    a::_PivotedInput,
    ::LinearAlgebra.RowMaximum;
    tol::Real=0.0,
    block_size::Integer=128,
    team=nothing,
    check::Bool=true,
)
    lock(_matrix_lock) do
        source, n, ftol, block = _pivoted_input_plan(a, tol, block_size, team)
        return _pivoted_construction_requirement(n, eltype(source), ftol, block, team)
    end
end
function FaerScratch(
    f::_PivotedConstruction, a::_PivotedInput, p::LinearAlgebra.RowMaximum; kwargs...
)
    req = scratch_requirements(f, a, p; kwargs...)
    return FaerScratch(req.bytes; alignment=req.alignment)
end
function reserve!(
    s::FaerScratch,
    f::_PivotedConstruction,
    a::_PivotedInput,
    p::LinearAlgebra.RowMaximum;
    kwargs...,
)
    req = scratch_requirements(f, a, p; kwargs...)
    return reserve!(s, req.bytes; alignment=req.alignment)
end
function _pivoted_with_scratch(f, req, scratch)
    temp = isnothing(scratch)
    work = temp ? FaerScratch(req.bytes; alignment=req.alignment) : scratch
    work isa FaerScratch || throw(ArgumentError("scratch must be a FaerScratch"))
    try
        cap = scratch_capacity(work)
        cap.bytes >= req.bytes && cap.alignment >= req.alignment || throw(
            ArgumentError(
                "pivoted Cholesky scratch capacity/alignment is insufficient; call reserve! with the same options",
            ),
        )
        return f(work)
    finally
        temp && close(work)
    end
end
_pivoted_empty_owner(::Type{Float64}) = RustCrate.OwnedMatrix()
_pivoted_empty_owner(::Type{T}) where {T} = RustCrate.TypedMatrix()
function _pivoted_cholesky_workspace(
    a;
    tol::Real=0.0,
    check::Bool=true,
    block_size::Integer=128,
    scratch=nothing,
    team=nothing,
)
    lock(_matrix_lock) do
        _iterative_resource_check(scratch, team)
        source, n, ftol, block = _pivoted_input_plan(a, tol, block_size, team)
        req = _pivoted_construction_requirement(n, eltype(source), ftol, block, team)
        _pivoted_with_scratch(req, scratch) do work
            actual = source isa FaerMatrixView ? copy(source) : source
            try
                a isa _PivotedBorrow &&
                    !LinearAlgebra.ishermitian(actual) &&
                    throw(ArgumentError("matrix must be Hermitian"))
                T = eltype(a)
                h = _pivoted_empty_owner(T)
                p = zeros(UInt, n)
                lower = a isa _PivotedBorrow || a.uplo == 'L'
                rank = Int(
                    _native_call(
                        RustCrate.pivoted_llt_workspace,
                        actual,
                        h,
                        work,
                        p,
                        lower,
                        UInt(n),
                        _tag(T),
                        ftol,
                        block,
                        team,
                    ),
                )
                factors = FaerMatrix{T,typeof(h)}(h, (n, n))
                if check && rank < n
                    close(factors)
                    throw(LinearAlgebra.RankDeficientException(rank))
                end
                return LinearAlgebra.CholeskyPivoted(
                    factors, 'L', Int.(p) .+ 1, rank, tol, rank == n ? 0 : 1
                )
            finally
                actual === source || close(actual)
            end
        end
    end
end
function _pivoted_factor_plan(F, op, b=nothing)
    _dense_rhs_validate(F.factors)
    n = LinearAlgebra.checksquare(F.factors)
    F.uplo in ('L', 'U') || throw(ArgumentError("pivoted Cholesky triangle must be L or U"))
    0 <= F.rank <= n || throw(ArgumentError("pivoted Cholesky rank is out of bounds"))
    op != 2 && F.rank != n && throw(LinearAlgebra.RankDeficientException(Int(F.rank)))
    length(F.p) == n ||
        throw(DimensionMismatch("pivot permutation length must match factors"))
    p = zeros(UInt, n)
    seen = falses(n)
    for (i, j) in enumerate(F.p)
        1 <= j <= n && !seen[j] ||
            throw(ArgumentError("pivot permutation must contain each index once"))
        seen[j] = true
        p[i] = UInt(j - 1)
    end
    if op in (0, 3)
        b isa AbstractVecOrMat || throw(ArgumentError("RHS must be a vector or matrix"))
        _dense_rhs_validate(b)
        Base.require_one_based_indexing(b)
        op == 3 &&
            b isa AbstractVector &&
            throw(ArgumentError("right solve requires a matrix"))
        size(b, op == 3 ? 2 : 1) == n ||
            throw(DimensionMismatch("pivoted Cholesky RHS size must match"))
        P = promote_type(eltype(F), eltype(b))
        P <: _NativeScalar ||
            throw(ArgumentError("unsupported pivoted Cholesky promoted scalar type"))
        return n,
        if op == 3
            size(b, 1)
        elseif b isa AbstractVector
            1
        else
            size(b, 2)
        end,
        P,
        p
    end
    return n, 0, eltype(F), p
end
function _pivoted_apply_requirement(n, k, P, op, team)
    info = zeros(UInt, 2)
    _native_call(
        RustCrate.pivoted_llt_apply_requirements,
        UInt(n),
        UInt(k),
        _tag(P),
        Int32(op),
        team,
        info,
    )
    return (bytes=Int(info[1]), alignment=Int(info[2]))
end
"""
    scratch_requirements(ldiv!, F::CholeskyPivoted, B; team=nothing)
    scratch_requirements(rdiv!, F::CholeskyPivoted, B; team=nothing)
    scratch_requirements(inv, F::CholeskyPivoted; team=nothing)
    scratch_requirements(reconstruct, F::CholeskyPivoted; team=nothing)

For native factor storage, query kernel and inverse-permutation temporaries
using current factor/RHS metadata and the promoted scalar type. Right solve
width is `size(B, 1)`. Native RHS `\` and `/` have matching queries. Promotion,
upper-to-lower conversion, result owners and Julia conversion allocate separately.
"""
function scratch_requirements(
    ::typeof(LinearAlgebra.ldiv!),
    F::_PivotedNativeCholesky,
    b::AbstractVecOrMat;
    team=nothing,
)
    lock(_matrix_lock) do
        _iterative_resource_check(nothing, team)
        n, k, P, _ = _pivoted_factor_plan(F, 0, b)
        return _pivoted_apply_requirement(n, k, P, 0, team)
    end
end
function scratch_requirements(
    ::typeof(\), F::_PivotedNativeCholesky, b::FaerMatrix; kwargs...
)
    return scratch_requirements(LinearAlgebra.ldiv!, F, b; kwargs...)
end
function scratch_requirements(
    f::Union{typeof(inv),typeof(reconstruct)}, F::_PivotedNativeCholesky; team=nothing
)
    lock(_matrix_lock) do
        _iterative_resource_check(nothing, team)
        op = f === inv ? 1 : 2
        n, k, P, _ = _pivoted_factor_plan(F, op)
        return _pivoted_apply_requirement(n, k, P, op, team)
    end
end
const _PivotedApply = Union{
    typeof(LinearAlgebra.rdiv!),
    typeof(/),
    typeof(LinearAlgebra.ldiv!),
    typeof(\),
    typeof(inv),
    typeof(reconstruct),
}
function FaerScratch(f::_PivotedApply, F::_PivotedNativeCholesky, args...; kwargs...)
    req = scratch_requirements(f, F, args...; kwargs...)
    return FaerScratch(req.bytes; alignment=req.alignment)
end
function reserve!(
    s::FaerScratch, f::_PivotedApply, F::_PivotedNativeCholesky, args...; kwargs...
)
    req = scratch_requirements(f, F, args...; kwargs...)
    return reserve!(s, req.bytes; alignment=req.alignment)
end
# Copy verified dense storage directly; do not invoke a user conversion twice.
function _pivoted_convert(a, ::Type{P}, dims; name="RHS") where {P}
    if a isa AbstractVector
        vector = Vector{P}(a)
        vector isa Vector{P} ||
            throw(ArgumentError("pivoted Cholesky $name conversion changed scalar storage"))
        (length(vector), 1) == dims ||
            throw(DimensionMismatch("pivoted Cholesky $name conversion changed shape"))
        data = reshape(vector, :, 1)
    else
        data = Matrix{P}(a)
        data isa Matrix{P} ||
            throw(ArgumentError("pivoted Cholesky $name conversion changed scalar storage"))
        size(data) == dims ||
            throw(DimensionMismatch("pivoted Cholesky $name conversion changed shape"))
    end
    return _matrix_from_dense(data)
end
function _pivoted_native_apply(F, b, op; scratch=nothing, team=nothing)
    lock(_matrix_lock) do
        _iterative_resource_check(scratch, team)
        n, k, P, p = _pivoted_factor_plan(F, op, b)
        req = _pivoted_apply_requirement(n, k, P, op, team)
        _pivoted_with_scratch(req, scratch) do work
            factors = nothing
            rhs = nothing
            structure = (F.uplo, F.rank)
            try
                factors = if P === eltype(F)
                    F.factors
                else
                    _pivoted_convert(
                        if F.uplo == 'L'
                            LinearAlgebra.LowerTriangular(F.factors)
                        else
                            LinearAlgebra.UpperTriangular(F.factors)
                        end,
                        P,
                        (n, n);
                        name="factor",
                    )
                end
                if op in (0, 3)
                    dims = op == 3 ? (k, n) : (n, k)
                    rhs = b isa FaerMatrix{P} ? b : _pivoted_convert(b, P, dims)
                    size(rhs) == dims || throw(
                        DimensionMismatch("pivoted Cholesky RHS conversion changed shape"),
                    )
                end
                current_n, current_k, current_P, current_p = _pivoted_factor_plan(F, op, b)
                (
                    current_n == n &&
                    current_k == k &&
                    current_P === P &&
                    current_p == p &&
                    (F.uplo, F.rank) == structure
                ) || throw(
                    ArgumentError("pivoted Cholesky metadata changed during conversion")
                )
                _dense_rhs_validate(factors)
                h = _pivoted_empty_owner(P)
                _native_call(
                    RustCrate.pivoted_llt_apply_workspace,
                    factors,
                    h,
                    rhs,
                    work,
                    p,
                    F.uplo == 'L',
                    UInt(n),
                    UInt(k),
                    _tag(P),
                    Int32(op),
                    team,
                )
                return FaerMatrix{P,typeof(h)}(h, op == 3 ? (k, n) : (n, op == 0 ? k : n))
            finally
                isnothing(rhs) || rhs === b || close(rhs)
                isnothing(factors) || factors === F.factors || close(factors)
            end
        end
    end
end
_pivoted_storage_owner(::Any) = nothing
_pivoted_storage_owner(a::FaerMatrix) = a
_pivoted_storage_owner(a::FaerView) = _view_owner(a)
function _pivoted_storage_owner(a::Union{LinearAlgebra.Transpose,LinearAlgebra.Adjoint})
    return _pivoted_storage_owner(parent(a))
end
function _pivoted_destination_check(F, b)
    lock(_matrix_lock) do
        _dense_rhs_validate(F.factors)
        _dense_rhs_validate(b)
        owner = _pivoted_storage_owner(b)
        return !isnothing(owner) &&
               _handle(owner) === _handle(F.factors) &&
               throw(
                   ArgumentError(
                       "pivoted Cholesky destination must not alias factor storage"
                   ),
               )
    end
end

function scratch_requirements(
    ::typeof(LinearAlgebra.rdiv!),
    F::_PivotedNativeCholesky,
    b::AbstractMatrix;
    team=nothing,
)
    lock(_matrix_lock) do
        _iterative_resource_check(nothing, team)
        n, k, P, _ = _pivoted_factor_plan(F, 3, b)
        return _pivoted_apply_requirement(n, k, P, 3, team)
    end
end
function scratch_requirements(
    ::typeof(/), F::_PivotedNativeCholesky, b::FaerMatrix; kwargs...
)
    return scratch_requirements(LinearAlgebra.rdiv!, F, b; kwargs...)
end

function _pivoted_storage_owner(
    a::Union{SubArray,Base.ReshapedArray,Base.PermutedDimsArray}
)
    return _pivoted_storage_owner(parent(a))
end
function LinearAlgebra.cholesky(a::FaerMatrixView, ::LinearAlgebra.RowMaximum; kwargs...)
    return _pivoted_cholesky(a; kwargs...)
end
function LinearAlgebra.cholesky!(a::FaerMatrixView, ::LinearAlgebra.RowMaximum; kwargs...)
    return _pivoted_writeback!(a; kwargs...)
end
for W in (:Hermitian, :Symmetric)
    bound = W === :Symmetric ? Real : Any
    @eval LinearAlgebra.cholesky(
        a::LinearAlgebra.$W{T,<:FaerMatrixView}, ::LinearAlgebra.RowMaximum; kwargs...
    ) where {T} = _pivoted_cholesky(a; kwargs...)
    @eval LinearAlgebra.cholesky!(
        a::LinearAlgebra.$W{T,<:FaerMatrixView}, ::LinearAlgebra.RowMaximum; kwargs...
    ) where {T<:$bound} = _pivoted_writeback!(a; kwargs...)
end
