const _QzConstructionFunction = Union{
    typeof(LinearAlgebra.schur),
    typeof(LinearAlgebra.schur!),
    typeof(LinearAlgebra.hessenberg),
    typeof(LinearAlgebra.hessenberg!),
}
_qz_validate_operand(a) = _dense_rhs_validate(a)
function _qz_validate_operand(a::FaerMatrix)
    return _native_call(
        RustCrate.qz_input_validate, a, UInt(size(a, 1)), UInt(size(a, 2)), _tag(eltype(a))
    )
end
function _qz_validate_operand(a::FaerView)
    owner = _view_owner(a)
    eltype(a) === eltype(owner) || throw(ArgumentError("QZ view scalar metadata mismatch"))
    return _qz_validate_operand(owner)
end
function _qz_validate_operand(
    a::Union{
        LinearAlgebra.Hermitian,
        LinearAlgebra.Symmetric,
        LinearAlgebra.Transpose,
        LinearAlgebra.Adjoint,
        LinearAlgebra.AbstractTriangular,
    },
)
    eltype(a) === eltype(parent(a)) ||
        throw(ArgumentError("QZ wrapper scalar metadata mismatch"))
    return _qz_validate_operand(parent(a))
end
function _qz_validate_operand(a::LinearAlgebra.Diagonal)
    return _qz_validate_operand(a.diag)
end
function _qz_validate_operand(
    a::Union{LinearAlgebra.Bidiagonal,LinearAlgebra.SymTridiagonal}
)
    _qz_validate_operand(a.dv)
    return _qz_validate_operand(a.ev)
end
function _qz_validate_operand(a::LinearAlgebra.Tridiagonal)
    _qz_validate_operand(a.dl)
    _qz_validate_operand(a.d)
    return _qz_validate_operand(a.du)
end
function _qz_validate_operand(a::FaerSparseMatrix{T}) where {T}
    return _native_call(
        RustCrate.sparse_factor_matrix_validate,
        a,
        UInt(size(a, 1)),
        UInt(size(a, 2)),
        _tag(T),
        _sparse_format(a.format),
        UInt(a.stored),
    )
end
function _qz_validate_operand(a::FaerSparsity)
    return _native_call(
        RustCrate.qz_pattern_validate,
        a,
        UInt(size(a, 1)),
        UInt(size(a, 2)),
        _sparse_format(a.format),
        UInt(a.stored),
    )
end
function _qz_plan(a, b, initial, team)
    isnothing(team) || team isa FaerTeam || throw(ArgumentError("team must be a FaerTeam"))
    isnothing(team) || _handle(team)
    _qz_validate_operand(a)
    _qz_validate_operand(b)
    n = LinearAlgebra.checksquare(a)
    size(b) == (n, n) ||
        throw(DimensionMismatch("QZ inputs must have matching square shapes"))
    if !isnothing(initial)
        _check_pair_factor(initial)
        for m in _pair_matrices(initial)
            _qz_validate_operand(m)
        end
    end
    P = promote_type(eltype(a), eltype(b))
    P <: _NativeScalar ||
        throw(ArgumentError("QZ requires a supported native promoted scalar type"))
    return n, P
end
function _qz_native_requirement(n, P, stage, team)
    info = zeros(UInt, 2)
    GC.@preserve info begin
        args = (UInt(n), _tag(P), Int32(stage), info)
        _native_call(RustCrate.qz_construction_requirements, args..., team)
    end
    return (bytes=Int(info[1]), alignment=Int(info[2]))
end
"""
    scratch_requirements(schur, A, B; team=nothing)
    scratch_requirements(hessenberg, A, B; team=nothing)
    scratch_requirements(schur, F::FaerGeneralizedHessenberg; team=nothing)

Query declared native temporary storage for generalized Hessenberg/QZ construction.
The corresponding bang methods have the same requirement. One matrix input must
be a FaerMatrix or FaerMatrixView. Metadata is checked before promotion; saved H/T/Q/Z are
validated on both query and execution. Includes QR coefficients, projection and
homogeneous-value temporary columns. Owned outputs, staging, promotion and Julia
alpha/beta arrays allocate separately. The selected faer QR may also allocate its
internal deep Householder partial buffer (faer #267 remains unmerged).
"""
function scratch_requirements(
    fun::_QzConstructionFunction,
    a::AbstractMatrix{<:Number},
    b::AbstractMatrix{<:Number};
    team=nothing,
)
    a isa _BorrowMatrix ||
        b isa _BorrowMatrix ||
        throw(ArgumentError("QZ workspace requires a native matrix or view input"))
    lock(_matrix_lock) do
        n, P = _qz_plan(a, b, nothing, team)
        return _qz_native_requirement(
            n, P, fun in (LinearAlgebra.hessenberg, LinearAlgebra.hessenberg!) ? 1 : 0, team
        )
    end
end
function scratch_requirements(
    fun::Union{typeof(LinearAlgebra.schur),typeof(LinearAlgebra.schur!)},
    f::FaerGeneralizedHessenberg;
    team=nothing,
)
    lock(_matrix_lock) do
        n, P = _qz_plan(f.H, f.T, f, team)
        return _qz_native_requirement(n, P, 2, team)
    end
end
function FaerScratch(fun::_QzConstructionFunction, args...; team=nothing)
    req = scratch_requirements(fun, args...; team)
    return FaerScratch(req.bytes; alignment=req.alignment)
end
function reserve!(s::FaerScratch, fun::_QzConstructionFunction, args...; team=nothing)
    req = scratch_requirements(fun, args...; team)
    return reserve!(s, req.bytes; alignment=req.alignment)
end
function _qz_with_workspace(f, a, b, initial, only_hessenberg, scratch, team)
    isnothing(scratch) ||
        scratch isa FaerScratch ||
        throw(ArgumentError("scratch must be a FaerScratch"))
    n, P = _qz_plan(a, b, initial, team)
    stage = if only_hessenberg
        1
    elseif isnothing(initial)
        0
    else
        2
    end
    req = _qz_native_requirement(n, P, stage, team)
    work = isnothing(scratch) ? FaerScratch(req.bytes; alignment=req.alignment) : scratch
    try
        cap = scratch_capacity(work)
        cap.bytes >= req.bytes && cap.alignment >= req.alignment || throw(
            ArgumentError(
                "QZ construction scratch capacity/alignment is insufficient; call reserve! before computing",
            ),
        )
        f(n, P, work)
    finally
        isnothing(scratch) && close(work)
    end
end
function _qz_materialize(a, P)
    if a isa FaerMatrix
        return _coerce(a, P)
    elseif a isa FaerMatrixView
        x = copy(a)
        eltype(x) === P && return x
        try
            return _coerce(x, P)
        finally
            close(x)
        end
    end
    return FaerMatrix{P}(a)
end
function _with_qz_inputs(f, a, b, P)
    aa = _qz_materialize(a, P)
    try
        bb = _qz_materialize(b, P)
        try
            f(aa, bb)
        finally
            bb === b || close(bb)
        end
    finally
        aa === a || close(aa)
    end
end
