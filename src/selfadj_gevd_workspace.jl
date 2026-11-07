"""
    scratch_requirements(eigen, A::Hermitian, B::Hermitian; team=nothing)

Workspace for definite self-adjoint generalized EVD, including LLT reduction
copies and eigenvalue storage. Both selected triangles are respected independently.
B must be positive definite. Output arrays and sorting allocate separately.
The same query supports eigvals and their bang variants.
"""
function scratch_requirements(
    f::_EvdFunction,
    a::_AnyFaerSelfAdjoint,
    b::_AnyFaerSelfAdjoint;
    team = nothing,
)
    lock(_matrix_lock) do
        source=parent(a)
        _handle(source)
        _handle(parent(b))
        size(a)==size(b) ||
            throw(DimensionMismatch("generalized eigen input sizes must match"))
        n=LinearAlgebra.checksquare(source)
        isnothing(team) ||
            team isa FaerTeam ||
            throw(ArgumentError("team must be a FaerTeam"))
        !isnothing(team) && _handle(team)
        info=zeros(UInt, 2)
        args=(
            UInt(n),
            _tag(promote_type(eltype(a), eltype(b))),
            f in (LinearAlgebra.eigvals, LinearAlgebra.eigvals!),
        )
        GC.@preserve info begin
            _native_call(RustCrate.selfadj_gevd_requirements, args..., info, team)
        end
        return (bytes = Int(info[1]), alignment = Int(info[2]))
    end
end
function FaerScratch(
    f::_EvdFunction,
    a::_AnyFaerSelfAdjoint,
    b::_AnyFaerSelfAdjoint;
    kwargs...,
)
    req=scratch_requirements(f, a, b; kwargs...)
    return FaerScratch(req.bytes; alignment = req.alignment)
end
function reserve!(
    s::FaerScratch,
    f::_EvdFunction,
    a::_AnyFaerSelfAdjoint,
    b::_AnyFaerSelfAdjoint;
    kwargs...,
)
    req=scratch_requirements(f, a, b; kwargs...)
    return reserve!(s, req.bytes; alignment = req.alignment)
end
# Translate native failures into LinearAlgebra's conventions: a B that is not
# positive definite throws PosDefException with the one-based failing pivot, and
# non-finite selected triangles throw ArgumentError. Inputs are never modified.
function _selfadj_gevd_rethrow(e)
    e isa RustCall.RustError || rethrow()
    msg=sprint(showerror, e)
    m=match(r"generalized EVD B is not positive definite at pivot (\d+)", msg)
    isnothing(m) || throw(LinearAlgebra.PosDefException(parse(Int, m.captures[1])))
    occursin("must be finite", msg) &&
        throw(ArgumentError("generalized eigenproblem matrices contain Infs or NaNs"))
    rethrow()
end
function _selfadj_gevd_workspace_result(
    a::_AnyFaerSelfAdjoint,
    b::_AnyFaerSelfAdjoint;
    vectors::Bool = true,
    sortby = _eigensort,
    scratch = nothing,
    team = nothing,
)
    lock(_matrix_lock) do
        source=parent(a)
        _handle(source)
        _handle(parent(b))
        size(a)==size(b) ||
            throw(DimensionMismatch("generalized eigen input sizes must match"))
        T=promote_type(eltype(a), eltype(b))
        source=_coerce(source, T)
        other=_coerce(parent(b), T)
        n=LinearAlgebra.checksquare(source)
        f=vectors ? LinearAlgebra.eigen : LinearAlgebra.eigvals
        req=scratch_requirements(f, a, b; team)
        temp=isnothing(scratch)
        work=temp ? FaerScratch(req.bytes; alignment = req.alignment) : scratch
        work isa FaerScratch || throw(ArgumentError("scratch must be a FaerScratch"))
        try
            cap=scratch_capacity(work)
            cap.bytes>=req.bytes && cap.alignment>=req.alignment || throw(
                ArgumentError(
                    "EVD scratch capacity/alignment is insufficient; reserve for the selected mode/team",
                ),
            )
            values=zeros(_realtype(T), n)
            eigenvectors=zeros(T, n, vectors ? n : 0)
            GC.@preserve values eigenvectors begin
                args=(
                    Ptr{UInt8}(pointer(eigenvectors)),
                    Ptr{UInt8}(pointer(values)),
                    !vectors,
                    _uplo(a)===:L,
                    _uplo(b)===:L,
                    UInt(n),
                    _tag(T),
                )
                try
                    _native_call(RustCrate.selfadj_gevd_workspace, source, other, work, team, args...)
                catch e
                    _selfadj_gevd_rethrow(e)
                end
            end
            if !isnothing(sortby)
                order=sortperm(values; by = sortby)
                values=values[order]
                vectors && (eigenvectors=eigenvectors[:, order])
            end
            return vectors ? LinearAlgebra.GeneralizedEigen(values, eigenvectors) : values
        finally
            temp && close(work)
        end
    end
end

# Diagonal right-hand sides follow `eigen(::AbstractMatrix, ::Diagonal)`: a zero
# entry is singular; a positive diagonal is a Hermitian positive definite B and
# uses the native self-adjoint driver; other diagonals use LinearAlgebra's
# general path on `D \ A`.
const _SelfAdjointDiagonal=Union{FaerDiagonal,LinearAlgebra.Diagonal}
function _selfadj_diagonal_gevd(a, d, vectors; kwargs...)
    LinearAlgebra.checksquare(a)==size(d, 1) ||
        throw(DimensionMismatch("generalized eigen input sizes must match"))
    v=d isa FaerDiagonal ? Vector(d) : Vector(d.diag)
    any(iszero, v) && throw(ArgumentError("right-hand side diagonal matrix is singular"))
    if all(x->isreal(x) && real(x)>0, v)
        T=promote_type(eltype(a), eltype(v))
        b=LinearAlgebra.Hermitian(FaerMatrix{T}(Matrix(LinearAlgebra.Diagonal(v))))
        return vectors ? LinearAlgebra.eigen(a, b; kwargs...) :
               LinearAlgebra.eigvals(a, b; kwargs...)
    end
    dense=Matrix(a)
    D=LinearAlgebra.Diagonal(v)
    return vectors ? LinearAlgebra.eigen(dense, D) : LinearAlgebra.eigvals(dense, Matrix(D))
end
"""
    eigen(A::Hermitian{<:Any,<:FaerMatrix}, D::Union{FaerDiagonal,Diagonal}; sortby, scratch, team)

Generalized self-adjoint eigenproblem with a diagonal `B`. A zero diagonal
entry throws `ArgumentError`; a positive diagonal uses the native self-adjoint
driver (accepting `sortby`, `scratch` and `team`); other diagonals follow
LinearAlgebra's general path and return Julia arrays.
"""
LinearAlgebra.eigen(a::_AnyFaerSelfAdjoint, d::FaerDiagonal; kwargs...) =
    _selfadj_diagonal_gevd(a, d, true; kwargs...)
LinearAlgebra.eigen(a::_AnyFaerSelfAdjoint, d::LinearAlgebra.Diagonal; kwargs...) =
    _selfadj_diagonal_gevd(a, d, true; kwargs...)
LinearAlgebra.eigvals(a::_AnyFaerSelfAdjoint, d::FaerDiagonal; kwargs...) =
    _selfadj_diagonal_gevd(a, d, false; kwargs...)
LinearAlgebra.eigvals(a::_AnyFaerSelfAdjoint, d::LinearAlgebra.Diagonal; kwargs...) =
    _selfadj_diagonal_gevd(a, d, false; kwargs...)
