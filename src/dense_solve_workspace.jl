_dense_rhs_validate(::Any) = nothing
_dense_rhs_layout(::FaerMatrix) = 3
_dense_rhs_layout(::FaerVector) = 0
_dense_rhs_layout(::FaerRow) = 1
_dense_rhs_layout(::FaerDiagonal) = 2
function _dense_rhs_validate(a::_ViewOwner)
    _native_call(
        RustCrate.factor_rhs_validate,
        a,
        UInt(size(a, 1)),
        UInt(size(a, 2)),
        _tag(eltype(a)),
        Int32(_dense_rhs_layout(a)),
    )
end
function _dense_rhs_validate(a::FaerView)
    owner=_view_owner(a)
    eltype(a)===eltype(owner) ||
        throw(ArgumentError("dense solve view scalar metadata mismatch"))
    return _dense_rhs_validate(owner)
end
function _dense_rhs_validate(a::Union{LinearAlgebra.Transpose,LinearAlgebra.Adjoint})
    eltype(a)===eltype(parent(a)) ||
        throw(ArgumentError("dense solve array scalar metadata mismatch"))
    return _dense_rhs_validate(parent(a))
end
function _dense_workspace_promote(f::FaerFactorization{T,K}, P; team = nothing) where {T,K}
    return _factor(
        _promoted_factor_matrix(f, P; team),
        Val(K),
        _dense_kind(f);
        uplo = f.uplo,
        team,
    )
end
function _dense_native_requirements(
    f,
    columns,
    op;
    team = nothing,
    precision = Float64(eps(_realtype(eltype(f)))),
)
    info=zeros(UInt, 2)
    args=(
        f,
        UInt(columns),
        Int32(op),
        UInt(size(f, 1)),
        UInt(size(f, 2)),
        _tag(eltype(f)),
        _dense_kind(f),
        Float64(precision),
        pointer(info),
    )
    GC.@preserve info begin
        # `nothing` crosses as RustCall 0.10's checked optional owner.
        _native_call(RustCrate.factor_solve_requirements, args..., team)
    end
    return (bytes = Int(info[1]), alignment = Int(info[2]))
end
function _dense_workspace_native_result(
    f,
    b,
    op,
    scratch,
    team;
    precision = Float64(eps(_realtype(eltype(f)))),
)
    req=_dense_native_requirements(f, size(b, 2), op; team, precision)
    temporary=isnothing(scratch)
    work=temporary ? FaerScratch(req.bytes; alignment = req.alignment) : scratch
    work isa FaerScratch || throw(ArgumentError("scratch must be a FaerScratch"))
    try
        cap=scratch_capacity(work)
        cap.bytes>=req.bytes && cap.alignment>=req.alignment || throw(
            ArgumentError(
                "dense solve scratch capacity/alignment is insufficient; call reserve!",
            ),
        )
        dims=(size(f, op in (1, 2) ? 1 : 2), size(b, 2))
        args=(
            f,
            b,
            work,
            Int32(op),
            UInt(size(f, 1)),
            UInt(size(f, 2)),
            UInt(size(b, 2)),
            _tag(eltype(f)),
            _dense_kind(f),
            Float64(precision),
        )
        fun=isnothing(team) ? RustCrate.factor_solve_workspace :
            RustCrate.factor_solve_workspace_team
        isnothing(team) || (args=(args..., team))
        return eltype(f)===Float64 ? _matrix_result(fun, dims, args...) :
               _typed_result(eltype(f), fun, dims, args...)
    finally
        temporary && close(work)
    end
end

function _dense_solve_requirements(f, b; team = nothing, right = false)
    lock(_matrix_lock) do
        original=_dense_parent(f)
        _handle(original)
        _spectral_rhs_open(b)
        _dense_rhs_validate(b)
        Base.require_one_based_indexing(b)
        isnothing(team) ||
            team isa FaerTeam ||
            throw(ArgumentError("team must be a FaerTeam"))
        isnothing(team) || _handle(team)
        LinearAlgebra.issuccess(original) ||
            throw(LinearAlgebra.SingularException(original.info))
        _dense_solve_validate(original)
        (right ? size(b, 2)==size(f, 2) : size(b, 1)==size(f, 1)) ||
            throw(DimensionMismatch("factor and RHS dimensions do not match"))
        op=_dense_op(f)
        right && (op=(2, 3, 0)[op+1])
        columns=right ? size(b, 1) : ndims(b)==1 ? 1 : size(b, 2)
        T=eltype(original)
        P=promote_type(T, _reflector_type(eltype(b)))
        # Float64 factors solve complex RHS by two real solves with the same buffer.
        (T===Float64 || P===T) &&
            return _dense_native_requirements(original, columns, op; team)
        promoted=_dense_workspace_promote(original, P; team)
        try
            return _dense_native_requirements(
                promoted,
                columns,
                op;
                team,
                precision = Float64(eps(_realtype(T))),
            )
        finally
            close(promoted)
        end
    end
end
"""
    scratch_requirements(ldiv!, X, F::FaerFactorization, B; team=nothing)
    scratch_requirements(ldiv!, F::FaerFactorization, B; team=nothing)
    scratch_requirements(rdiv!, X, B, F::FaerFactorization; team=nothing)

Query caller-owned workspace for retained dense factor solves. Two-argument
forms require square factors; transpose/adjoint, rectangular/deficient QR,
scalar promotion, vector/matrix RHS, and right solves are supported.
QR fallback uses saved Householder/R and an R SVD inside scratch; rectangular
LU/full LU reconstructs from saved components and computes SVD inside scratch.
Retained SVD
uses its existing U/S/V. The native result and Julia staging allocate separately.
QR retains the source-precision rank cutoff through mixed-scalar promotion.
Mixed-scalar queries can reconstruct/refactor an effective factor, following the
existing solve policy; that construction work remains outside solve scratch.
Float64 factors split complex RHS into two real solves reusing one buffer.
Use `FaerScratch(fun,args...;team)` or `reserve!(work,fun,args...;team)` and pass
`scratch=work` to `ldiv!`/`rdiv!`. Insufficient or closed resources fail before
writeback; factor/RHS owners are never retained by the scratch or team.
"""
function scratch_requirements(
    ::typeof(LinearAlgebra.ldiv!),
    dest::AbstractVecOrMat,
    f::_DenseSolveFactor,
    b::AbstractVecOrMat;
    team = nothing,
)
    lock(_matrix_lock) do
        _spectral_rhs_open(dest)
        _dense_rhs_validate(dest)
        size(dest)==(ndims(b)==1 ? (size(f, 2),) : (size(f, 2), size(b, 2))) ||
            throw(DimensionMismatch("factor solve destination shape mismatch"))
        return _dense_solve_requirements(f, b; team)
    end
end
function scratch_requirements(
    ::typeof(LinearAlgebra.rdiv!),
    dest::AbstractMatrix,
    b::AbstractMatrix,
    f::_DenseSolveFactor;
    team = nothing,
)
    lock(_matrix_lock) do
        _spectral_rhs_open(dest)
        _dense_rhs_validate(dest)
        size(dest)==(size(b, 1), size(f, 1)) ||
            throw(DimensionMismatch("right factor solve destination shape mismatch"))
        return _dense_solve_requirements(f, b; team, right = true)
    end
end
function scratch_requirements(
    ::typeof(LinearAlgebra.ldiv!),
    f::_DenseSolveFactor,
    b::AbstractVecOrMat;
    team = nothing,
)
    size(f, 1)==size(f, 2) ||
        throw(DimensionMismatch("two-argument ldiv! requires square factors"))
    return scratch_requirements(LinearAlgebra.ldiv!, b, f, b; team)
end
function scratch_requirements(
    ::typeof(LinearAlgebra.rdiv!),
    b::AbstractMatrix,
    f::_DenseSolveFactor;
    team = nothing,
)
    size(f, 1)==size(f, 2) ||
        throw(DimensionMismatch("two-argument rdiv! requires square factors"))
    return scratch_requirements(LinearAlgebra.rdiv!, b, b, f; team)
end
for (fn, signature, args) in (
    (
        :(LinearAlgebra.ldiv!),
        :(dest::AbstractVecOrMat, f::_DenseSolveFactor, b::AbstractVecOrMat),
        :(dest, f, b),
    ),
    (:(LinearAlgebra.ldiv!), :(f::_DenseSolveFactor, b::AbstractVecOrMat), :(f, b)),
    (
        :(LinearAlgebra.rdiv!),
        :(dest::AbstractMatrix, b::AbstractMatrix, f::_DenseSolveFactor),
        :(dest, b, f),
    ),
    (:(LinearAlgebra.rdiv!), :(b::AbstractMatrix, f::_DenseSolveFactor), :(b, f)),
)
    @eval function FaerScratch(::typeof($fn), $(signature.args...); team = nothing)
        req=scratch_requirements($fn, $(args.args...); team)
        return FaerScratch(req.bytes; alignment = req.alignment)
    end
    @eval function reserve!(
        work::FaerScratch,
        ::typeof($fn),
        $(signature.args...);
        team = nothing,
    )
        req=scratch_requirements($fn, $(args.args...); team)
        return reserve!(work, req.bytes; alignment = req.alignment)
    end
end
