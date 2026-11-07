# Caller workspace for sparse triangular solves (src/sparse_triangular.jl).
function _sparse_triangular_requirements(a, b; team = nothing)
    lock(_matrix_lock) do
        source, op, P=_sparse_triangular_prepare(a, b, team)
        return _sparse_triangular_query(source, a, op, P)
    end
end
"""
    scratch_requirements(ldiv!, A::SparseTriangular, B; team=nothing)
    scratch_requirements(ldiv!, Y, A::SparseTriangular, B; team=nothing)
    scratch_requirements(\\, A::SparseTriangular, B; team=nothing)

Bytes and alignment of the caller workspace for a sparse triangular solve. It is
zero when the stored owner already has faer's triangle layout and the scalar is
not promoted; otherwise it covers the rebuilt triangle (at most `nnz + n`
entries) or the promoted values. It depends on the sparsity pattern, not on the
stored values or the RHS width.
"""
scratch_requirements(
    ::typeof(LinearAlgebra.ldiv!),
    a::_SparseTriangular,
    b::AbstractVecOrMat;
    team = nothing,
) = _sparse_triangular_requirements(a, b; team)
function scratch_requirements(
    ::typeof(LinearAlgebra.ldiv!),
    dest::AbstractVecOrMat,
    a::_SparseTriangular,
    b::AbstractVecOrMat;
    team = nothing,
)
    size(dest)==size(b) ||
        throw(DimensionMismatch("triangular output size must match RHS"))
    return _sparse_triangular_requirements(a, b; team)
end
scratch_requirements(
    ::typeof(\),
    a::_SparseTriangular,
    b::AbstractVecOrMat;
    team = nothing,
) = _sparse_triangular_requirements(a, b; team)
for F in (:(typeof(LinearAlgebra.ldiv!)), :(typeof(\)))
    @eval begin
        FaerScratch(fun::$F, a::_SparseTriangular, b::AbstractVecOrMat; team = nothing) =
            _new_operator_workspace(fun, a, b; team)
        reserve!(
            s::FaerScratch,
            fun::$F,
            a::_SparseTriangular,
            b::AbstractVecOrMat;
            team = nothing,
        ) = _reserve_operator_workspace(s, fun, a, b; team)
    end
end
FaerScratch(
    fun::typeof(LinearAlgebra.ldiv!),
    dest::AbstractVecOrMat,
    a::_SparseTriangular,
    b::AbstractVecOrMat;
    team = nothing,
) = _new_operator_workspace(fun, dest, a, b; team)
reserve!(
    s::FaerScratch,
    fun::typeof(LinearAlgebra.ldiv!),
    dest::AbstractVecOrMat,
    a::_SparseTriangular,
    b::AbstractVecOrMat;
    team = nothing,
) = _reserve_operator_workspace(s, fun, dest, a, b; team)
