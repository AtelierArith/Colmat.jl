# Sparse QR solve resources preserve source-precision rank decisions.
const _SparseQrResourceParent = FaerSparseFactorization{T,:qr} where {T}
const _SparseQrResourceTrans = Union{
    LinearAlgebra.TransposeFactorization{T,<:FaerSparseFactorization{T,:qr}},
    LinearAlgebra.AdjointFactorization{T,<:FaerSparseFactorization{T,:qr}},
} where {T}
const _SparseQrResourceSolve = Union{_SparseQrResourceParent,_SparseQrResourceTrans}
const _SparseQrResourceOf{T} = Union{
    FaerSparseFactorization{T,:qr},
    LinearAlgebra.TransposeFactorization{T,<:FaerSparseFactorization{T,:qr}},
    LinearAlgebra.AdjointFactorization{T,<:FaerSparseFactorization{T,:qr}},
}
_sparse_qr_resource_parent(f::_SparseQrResourceParent) = f
_sparse_qr_resource_parent(f::_SparseQrResourceTrans) = parent(f)
_sparse_qr_resource_op(::_SparseQrResourceParent) = 0
_sparse_qr_resource_op(
    ::LinearAlgebra.TransposeFactorization{T,<:_SparseQrResourceParent},
) where {T} = 1
_sparse_qr_resource_op(
    ::LinearAlgebra.AdjointFactorization{T,<:_SparseQrResourceParent},
) where {T} = 2

function _sparse_qr_resource_prepare(f, b, team, right, atol, rtol)
    original=_sparse_qr_resource_parent(f)
    _handle(original)
    _spectral_rhs_open(b)
    _dense_rhs_validate(b)
    Base.require_one_based_indexing(b)
    isnothing(team)||team isa FaerTeam||throw(ArgumentError("team must be a FaerTeam"))
    isnothing(team)||_handle(team)
    LinearAlgebra.issuccess(original)||throw(LinearAlgebra.SingularException(original.info))
    (right ? size(b, 2)==size(f, 2) : size(b, 1)==size(f, 1))||throw(
        DimensionMismatch("QR factor and RHS dimensions must match"),
    )
    op=_sparse_qr_resource_op(f)
    right&&(op=(2, 3, 0)[op+1])
    at=isnothing(atol) ? 0 : atol
    rt=isnothing(rtol) ? _default_spectral_rtol(original, at) : rtol
    av, rv=_spectral_tolerances(at, rt)
    P=promote_type(eltype(original), _sparse_input_type(eltype(b)))
    relative_f32=promote_type(typeof(rt), _realtype(eltype(original)))===Float32
    fast=isnothing(atol)&&isnothing(rtol)&&op==0&&P===eltype(original)&&size(original, 1)>=size(
        original,
        2,
    )&&!_native_call(RustCrate.sparse_qr_needs_spectral_solve, original)
    return original, op, P, av, rv, relative_f32, fast
end
function _sparse_qr_resource_native_query(
    f,
    P,
    columns,
    op,
    at,
    rt;
    team = nothing,
    fast = false,
)
    info=zeros(UInt, 2)
    args=(
        f,
        UInt(size(f, 1)),
        UInt(size(f, 2)),
        UInt(columns),
        _tag(eltype(f)),
        _tag(P),
        Int32(op),
        at,
        rt,
        fast,
        pointer(info),
    )
    GC.@preserve info begin
        isnothing(team) ? _native_call(RustCrate.sparse_qr_rank_requirements, args...) :
        _native_call(RustCrate.sparse_qr_rank_requirements_team, args..., team)
    end
    return (bytes = Int(info[1]), alignment = Int(info[2]))
end
function _sparse_qr_resource_requirements(
    f,
    b;
    team = nothing,
    right = false,
    atol = nothing,
    rtol = nothing,
)
    lock(_matrix_lock) do
        original, op, P, at, rt, _, fast=_sparse_qr_resource_prepare(
            f,
            b,
            team,
            right,
            atol,
            rtol,
        )
        return _sparse_qr_resource_native_query(
            original,
            P,
            right ? size(b, 1) : ndims(b)==1 ? 1 : size(b, 2),
            op,
            at,
            rt;
            team,
            fast,
        )
    end
end
function _sparse_qr_resource_result(
    f,
    b;
    scratch = nothing,
    team = nothing,
    right = false,
    atol = nothing,
    rtol = nothing,
)
    lock(_matrix_lock) do
        original, op, P, at, rt, relative_f32, fast=_sparse_qr_resource_prepare(
            f,
            b,
            team,
            right,
            atol,
            rtol,
        )
        columns=right ? size(b, 1) : ndims(b)==1 ? 1 : size(b, 2)
        req=_sparse_qr_resource_native_query(original, P, columns, op, at, rt; team, fast)
        # Keep the existing fast full-rank default; deficient/tolerance fallbacks below
        # re-enter with explicit tolerances and therefore use the workspace kernel.
        if !right &&
           size(original, 1)>=size(original, 2) &&
           isnothing(scratch)&&isnothing(team)&&isnothing(atol)&&isnothing(rtol)
            input=ndims(b)==1 ? reshape(b, :, 1) : b
            return _sparse_solve(f, input)
        end
        temporary=isnothing(scratch)
        work=temporary ? FaerScratch(req.bytes; alignment = req.alignment) : scratch
        work isa FaerScratch||throw(ArgumentError("scratch must be a FaerScratch"))
        rhs=nothing
        try
            cap=scratch_capacity(work)
            cap.bytes>=req.bytes&&cap.alignment>=req.alignment||throw(
                ArgumentError("insufficient QR solve scratch; call reserve!"),
            )
            input=right ? reshape(adjoint(b), size(b, 2), size(b, 1)) :
                  ndims(b)==1 ? reshape(b, :, 1) : b
            rhs=FaerMatrix{P}(input)
            args=(
                original,
                rhs,
                work,
                UInt(size(original, 1)),
                UInt(size(original, 2)),
                UInt(columns),
                _tag(eltype(original)),
                _tag(P),
                Int32(op),
                at,
                rt,
                relative_f32,
                fast,
            )
            fun=isnothing(team) ? RustCrate.sparse_qr_rank_workspace :
                RustCrate.sparse_qr_rank_workspace_team
            isnothing(team)||(args=(args..., team))
            dims=(size(original, op in (1, 2) ? 1 : 2), columns)
            result=P===Float64 ? _matrix_result(fun, dims, args...) :
                   _typed_result(P, fun, dims, args...)
            right||return result
            try
                return FaerMatrix(copy(adjoint(Matrix(result))))
            finally
                close(result)
            end
        finally
            isnothing(rhs)||close(rhs)
            temporary&&close(work)
        end
    end
end
function _sparse_qr_resource_vector(f, b; kwargs...)
    result=_sparse_qr_resource_result(f, b; kwargs...)
    try
        return b isa _BorrowVector ? _vector_from_source(result) : vec(Matrix(result))
    finally
        close(result)
    end
end
function _sparse_qr_resource_write(
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

        _spectral_rhs_open(dest)
        _dense_rhs_validate(dest)
        Base.require_one_based_indexing(dest)
        expected=right ? (size(b, 1), size(f, 1)) :
                 ndims(b)==1 ? (size(f, 2),) : (size(f, 2), size(b, 2))
        size(dest)==expected ||
            throw(DimensionMismatch("sparse solve destination shape mismatch"))
        result=_sparse_qr_resource_result(f, b; right, scratch, team, atol, rtol)
        try
            buffer=ndims(dest)==1 ? vec(Matrix(result)) : Matrix(result)
            return _dense_copy_result!(dest, buffer)
        finally
            close(result)
        end
    end
end
Base.:\(f::_SparseQrResourceSolve, b::AbstractMatrix{<:Number}; kwargs...) =
    _sparse_qr_resource_result(f, b; kwargs...)
Base.:\(f::_SparseQrResourceSolve, b::AbstractVector{<:Number}; kwargs...) =
    _sparse_qr_resource_vector(f, b; kwargs...)
Base.:/(b::AbstractMatrix{<:Number}, f::_SparseQrResourceSolve; kwargs...) =
    _sparse_qr_resource_result(f, b; right = true, kwargs...)
for B in (:AbstractMatrix, :AbstractVector)
    @eval LinearAlgebra.ldiv!(dest::$B, f::_SparseQrResourceSolve, b::$B; kwargs...) =
        _sparse_qr_resource_write(dest, f, b; kwargs...)
end
LinearAlgebra.ldiv!(f::_SparseQrResourceSolve, b::AbstractVecOrMat; kwargs...) =
    _sparse_qr_resource_write(b, f, b; kwargs...)
LinearAlgebra.rdiv!(
    dest::AbstractMatrix,
    b::AbstractMatrix,
    f::_SparseQrResourceSolve;
    kwargs...,
) = _sparse_qr_resource_write(dest, f, b; right = true, kwargs...)
LinearAlgebra.rdiv!(b::AbstractMatrix, f::_SparseQrResourceSolve; kwargs...) =
    _sparse_qr_resource_write(b, f, b; right = true, kwargs...)
for Wrapper in (LinearAlgebra.TransposeFactorization, LinearAlgebra.AdjointFactorization)
    @eval LinearAlgebra.rdiv!(
        b::AbstractMatrix,
        f::$Wrapper{T,<:FaerSparseFactorization{T,:qr}};
        kwargs...,
    ) where {T} = _sparse_qr_resource_write(b, f, b; right = true, kwargs...)
end
for B in (:Matrix, :Vector)
    helper=B===:Matrix ? :_sparse_qr_resource_result : :_sparse_qr_resource_vector
    @eval Base.:\(
        f::_SparseQrResourceOf{T},
        b::$B{Complex{T}};
        kwargs...,
    ) where {T<:Union{Float32,Float64}} = $helper(f, b; kwargs...)
end
Base.:/(
    b::Matrix{Complex{T}},
    f::_SparseQrResourceOf{T};
    kwargs...,
) where {T<:Union{Float32,Float64}} =
    _sparse_qr_resource_result(f, b; right = true, kwargs...)

"""
    scratch_requirements(ldiv!, F::FaerSparseFactorization{T,:qr}, B; team,atol,rtol)
    scratch_requirements(rdiv!, B, F::FaerSparseFactorization{T,:qr}; team,atol,rtol)

Query native QR solve scratch for all algorithms, shapes, scalar promotions and
transforms. Includes source-precision R SVD, promoted coefficients and projection.
Rank decisions retain the source precision. Saved R and column permutations are
borrowed; Q is applied directly to the R singular vectors using caller scratch.
No independent Q/R/column-permutation copy is prepared for a solve. The rank
certificate is prepared once per numeric factor revision. RHS/result ownership
and destination staging allocate separately. Indexed simplicial reflector
traversal is sequential; blocked Householder calls use the team. No input
reconstruction or explicit dense pseudoinverse is used. Use `reserve!` explicitly.
"""
scratch_requirements(
    ::typeof(LinearAlgebra.ldiv!),
    f::_SparseQrResourceSolve,
    b::AbstractVecOrMat;
    team = nothing,
    atol = nothing,
    rtol = nothing,
) = _sparse_qr_resource_requirements(f, b; team, atol, rtol)
scratch_requirements(
    ::typeof(LinearAlgebra.rdiv!),
    b::AbstractMatrix,
    f::_SparseQrResourceSolve;
    team = nothing,
    atol = nothing,
    rtol = nothing,
) = _sparse_qr_resource_requirements(f, b; team, right = true, atol, rtol)
function scratch_requirements(
    ::typeof(LinearAlgebra.ldiv!),
    dest::AbstractVecOrMat,
    f::_SparseQrResourceSolve,
    b::AbstractVecOrMat;
    team = nothing,
    atol = nothing,
    rtol = nothing,
)
    lock(_matrix_lock) do
        _spectral_rhs_open(dest)
        _dense_rhs_validate(dest)
        size(dest)==(ndims(b)==1 ? (size(f, 2),) : (size(f, 2), size(b, 2))) ||
            throw(DimensionMismatch("sparse solve destination shape mismatch"))
        return _sparse_qr_resource_requirements(f, b; team, atol, rtol)
    end
end
function scratch_requirements(
    ::typeof(LinearAlgebra.rdiv!),
    dest::AbstractMatrix,
    b::AbstractMatrix,
    f::_SparseQrResourceSolve;
    team = nothing,
    atol = nothing,
    rtol = nothing,
)
    lock(_matrix_lock) do
        _spectral_rhs_open(dest)
        _dense_rhs_validate(dest)
        size(dest)==(size(b, 1), size(f, 1)) ||
            throw(DimensionMismatch("sparse solve destination shape mismatch"))
        return _sparse_qr_resource_requirements(f, b; team, right = true, atol, rtol)
    end
end
scratch_requirements(
    ::typeof(\),
    f::_SparseQrResourceSolve,
    b::AbstractVecOrMat;
    team = nothing,
    atol = nothing,
    rtol = nothing,
) = _sparse_qr_resource_requirements(f, b; team, atol, rtol)
scratch_requirements(
    ::typeof(/),
    b::AbstractMatrix,
    f::_SparseQrResourceSolve;
    team = nothing,
    atol = nothing,
    rtol = nothing,
) = _sparse_qr_resource_requirements(f, b; team, right = true, atol, rtol)
FaerScratch(
    fun::typeof(LinearAlgebra.ldiv!),
    f::_SparseQrResourceSolve,
    b::AbstractVecOrMat;
    team = nothing,
    atol = nothing,
    rtol = nothing,
) = _new_sparse_qr_solve_workspace(fun, f, b; team, atol, rtol)
reserve!(
    s::FaerScratch,
    fun::typeof(LinearAlgebra.ldiv!),
    f::_SparseQrResourceSolve,
    b::AbstractVecOrMat;
    team = nothing,
    atol = nothing,
    rtol = nothing,
) = _reserve_sparse_qr_solve_workspace(s, fun, f, b; team, atol, rtol)
FaerScratch(
    fun::typeof(LinearAlgebra.ldiv!),
    dest::AbstractVecOrMat,
    f::_SparseQrResourceSolve,
    b::AbstractVecOrMat;
    team = nothing,
    atol = nothing,
    rtol = nothing,
) = _new_sparse_qr_solve_workspace(fun, dest, f, b; team, atol, rtol)
reserve!(
    s::FaerScratch,
    fun::typeof(LinearAlgebra.ldiv!),
    dest::AbstractVecOrMat,
    f::_SparseQrResourceSolve,
    b::AbstractVecOrMat;
    team = nothing,
    atol = nothing,
    rtol = nothing,
) = _reserve_sparse_qr_solve_workspace(s, fun, dest, f, b; team, atol, rtol)
FaerScratch(
    fun::typeof(LinearAlgebra.rdiv!),
    b::AbstractMatrix,
    f::_SparseQrResourceSolve;
    team = nothing,
    atol = nothing,
    rtol = nothing,
) = _new_sparse_qr_solve_workspace(fun, b, f; team, atol, rtol)
reserve!(
    s::FaerScratch,
    fun::typeof(LinearAlgebra.rdiv!),
    b::AbstractMatrix,
    f::_SparseQrResourceSolve;
    team = nothing,
    atol = nothing,
    rtol = nothing,
) = _reserve_sparse_qr_solve_workspace(s, fun, b, f; team, atol, rtol)
FaerScratch(
    fun::typeof(LinearAlgebra.rdiv!),
    dest::AbstractMatrix,
    b::AbstractMatrix,
    f::_SparseQrResourceSolve;
    team = nothing,
    atol = nothing,
    rtol = nothing,
) = _new_sparse_qr_solve_workspace(fun, dest, b, f; team, atol, rtol)
reserve!(
    s::FaerScratch,
    fun::typeof(LinearAlgebra.rdiv!),
    dest::AbstractMatrix,
    b::AbstractMatrix,
    f::_SparseQrResourceSolve;
    team = nothing,
    atol = nothing,
    rtol = nothing,
) = _reserve_sparse_qr_solve_workspace(s, fun, dest, b, f; team, atol, rtol)
FaerScratch(
    fun::typeof(\),
    f::_SparseQrResourceSolve,
    b::AbstractVecOrMat;
    team = nothing,
    atol = nothing,
    rtol = nothing,
) = _new_sparse_qr_solve_workspace(fun, f, b; team, atol, rtol)
reserve!(
    s::FaerScratch,
    fun::typeof(\),
    f::_SparseQrResourceSolve,
    b::AbstractVecOrMat;
    team = nothing,
    atol = nothing,
    rtol = nothing,
) = _reserve_sparse_qr_solve_workspace(s, fun, f, b; team, atol, rtol)
FaerScratch(
    fun::typeof(/),
    b::AbstractMatrix,
    f::_SparseQrResourceSolve;
    team = nothing,
    atol = nothing,
    rtol = nothing,
) = _new_sparse_qr_solve_workspace(fun, b, f; team, atol, rtol)
reserve!(
    s::FaerScratch,
    fun::typeof(/),
    b::AbstractMatrix,
    f::_SparseQrResourceSolve;
    team = nothing,
    atol = nothing,
    rtol = nothing,
) = _reserve_sparse_qr_solve_workspace(s, fun, b, f; team, atol, rtol)

# LinearAlgebra defines additional real-factor/complex-row specializations.
for Wrapper in (LinearAlgebra.TransposeFactorization, LinearAlgebra.AdjointFactorization)
    @eval Base.:/(
        b::Union{
            Matrix{Complex{T}},
            LinearAlgebra.Adjoint{Complex{T},Vector{Complex{T}}},
            LinearAlgebra.Transpose{Complex{T},Vector{Complex{T}}},
        },
        f::$Wrapper{T,<:FaerSparseFactorization{T,:qr}};
        kwargs...,
    ) where {T<:Union{Float32,Float64}} =
        _sparse_qr_resource_result(f, b; right = true, kwargs...)
end
for Wrapper in (LinearAlgebra.TransposeFactorization, LinearAlgebra.AdjointFactorization),
    Row in (LinearAlgebra.Transpose, LinearAlgebra.Adjoint)

    @eval Base.:/(
        b::$Row{Complex{T},Vector{Complex{T}}},
        f::$Wrapper{T,<:FaerSparseFactorization{T,:qr}};
        kwargs...,
    ) where {T<:Union{Float32,Float64}} =
        _sparse_qr_resource_result(f, b; right = true, kwargs...)
end

for Wrapper in (LinearAlgebra.TransposeFactorization, LinearAlgebra.AdjointFactorization)
    @eval Base.:/(
        b::Matrix{Complex{T}},
        f::$Wrapper{T,<:FaerSparseFactorization{T,:qr}};
        kwargs...,
    ) where {T<:Union{Float32,Float64}} =
        _sparse_qr_resource_result(f, b; right = true, kwargs...)
end

function _new_sparse_qr_solve_workspace(fun, args...; kwargs...)
    req=scratch_requirements(fun, args...; kwargs...)
    return FaerScratch(req.bytes; alignment = req.alignment)
end
function _reserve_sparse_qr_solve_workspace(work, fun, args...; kwargs...)
    req=scratch_requirements(fun, args...; kwargs...)
    return reserve!(work, req.bytes; alignment = req.alignment)
end
