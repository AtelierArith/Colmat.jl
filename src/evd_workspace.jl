const _EvdFunction=Union{
    typeof(LinearAlgebra.eigen),
    typeof(LinearAlgebra.eigvals),
    typeof(LinearAlgebra.eigen!),
    typeof(LinearAlgebra.eigvals!),
}
"""
    scratch_requirements(eigen, A::Hermitian; team=nothing)
    scratch_requirements(eigvals, A::Hermitian; team=nothing)

Query self-adjoint EVD workspace (also for eigen!/eigvals!) on native Hermitian
or real Symmetric wrappers. Requirement includes a canonical selected-triangle
copy and temporary eigenvalues. With a team, actual execution conditions are
queried. Eigenvector output arrays and sorting allocate separately.
"""
function scratch_requirements(f::_EvdFunction, a::_AnyFaerSelfAdjoint; team = nothing)
    lock(_matrix_lock) do
        source=parent(a)
        _handle(source)
        n=LinearAlgebra.checksquare(source)
        isnothing(team) ||
            team isa FaerTeam ||
            throw(ArgumentError("team must be a FaerTeam"))
        !isnothing(team) && _handle(team)
        info=zeros(UInt, 2)
        args=(
            UInt(n),
            _tag(eltype(a)),
            f in (LinearAlgebra.eigvals, LinearAlgebra.eigvals!),
        )
        GC.@preserve info begin
            isnothing(team) ?
            _native_call(RustCrate.evd_requirements, args..., pointer(info)) :
            _native_call(RustCrate.evd_team_requirements, team, args..., pointer(info))
        end
        return (bytes = Int(info[1]), alignment = Int(info[2]))
    end
end
function FaerScratch(f::_EvdFunction, a::_AnyFaerSelfAdjoint; kwargs...)
    req=scratch_requirements(f, a; kwargs...)
    return FaerScratch(req.bytes; alignment = req.alignment)
end
function reserve!(s::FaerScratch, f::_EvdFunction, a::_AnyFaerSelfAdjoint; kwargs...)
    req=scratch_requirements(f, a; kwargs...)
    return reserve!(s, req.bytes; alignment = req.alignment)
end
function _evd_workspace_result(
    a::_AnyFaerSelfAdjoint;
    vectors::Bool = true,
    sortby = _eigensort,
    scratch = nothing,
    team = nothing,
)
    lock(_matrix_lock) do
        source=parent(a)
        _handle(source)
        T=eltype(a)
        n=LinearAlgebra.checksquare(source)
        f=vectors ? LinearAlgebra.eigen : LinearAlgebra.eigvals
        req=scratch_requirements(f, a; team)
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
                    UInt(n),
                    _tag(T),
                )
                isnothing(team) ?
                _native_call(RustCrate.evd_workspace, source, work, args...) :
                _native_call(RustCrate.evd_workspace_team, source, work, team, args...)
            end
            if !isnothing(sortby)
                order=sortperm(values; by = sortby)
                values=values[order]
                vectors && (eigenvectors=eigenvectors[:, order])
            end
            return vectors ? LinearAlgebra.Eigen(values, eigenvectors) : values
        finally
            temp && close(work)
        end
    end
end
