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
            isnothing(team) ?
            _native_call(RustCrate.selfadj_gevd_requirements, args..., pointer(info)) :
            _native_call(
                RustCrate.selfadj_gevd_team_requirements,
                team,
                args...,
                pointer(info),
            )
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
                isnothing(team) ?
                _native_call(
                    RustCrate.selfadj_gevd_workspace,
                    source,
                    other,
                    work,
                    args...,
                ) :
                _native_call(
                    RustCrate.selfadj_gevd_workspace_team,
                    source,
                    other,
                    work,
                    team,
                    args...,
                )
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
