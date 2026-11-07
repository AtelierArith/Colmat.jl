"""
    scratch_requirements(eigen, A::FaerMatrix, B::FaerMatrix; team=nothing, side=:right)

Generalized EVD workspace including writable A/B copies and packed real
outputs where required. Use `side=:both` for simultaneous left/right output;
real input requires an additional packed matrix. Bang queries describe the
eigen pass; the separate
QZ writeback uses its own sequential workspace. Output and sorting allocate.
"""
function scratch_requirements(
    f::_EvdFunction,
    a::FaerMatrix,
    b::FaerMatrix;
    team = nothing,
    side::Symbol = :right,
)
    lock(_matrix_lock) do
        side in (:left, :right, :both) ||
            throw(ArgumentError("side must be :left, :right or :both"))
        _handle(a)
        _handle(b)
        size(a)==size(b) ||
            throw(DimensionMismatch("generalized eigen input sizes must match"))
        n=LinearAlgebra.checksquare(a)
        isnothing(team) ||
            team isa FaerTeam ||
            throw(ArgumentError("team must be a FaerTeam"))
        !isnothing(team) && _handle(team)
        info=zeros(UInt, 2)
        args=(
            UInt(n),
            _tag(promote_type(eltype(a), eltype(b))),
            f in (LinearAlgebra.eigvals, LinearAlgebra.eigvals!),
            side===:both,
        )
        GC.@preserve info begin
            isnothing(team) ?
            _native_call(RustCrate.general_gevd_requirements, args..., pointer(info)) :
            _native_call(
                RustCrate.general_gevd_team_requirements,
                team,
                args...,
                pointer(info),
            )
        end
        return (bytes = Int(info[1]), alignment = Int(info[2]))
    end
end
function FaerScratch(f::_EvdFunction, a::FaerMatrix, b::FaerMatrix; kwargs...)
    req=scratch_requirements(f, a, b; kwargs...)
    return FaerScratch(req.bytes; alignment = req.alignment)
end
function reserve!(s::FaerScratch, f::_EvdFunction, a::FaerMatrix, b::FaerMatrix; kwargs...)
    req=scratch_requirements(f, a, b; kwargs...)
    return reserve!(s, req.bytes; alignment = req.alignment)
end
function _general_gevd_workspace_result(
    a::FaerMatrix,
    b::FaerMatrix;
    vectors::Bool = true,
    homogeneous::Bool = false,
    side::Symbol = :right,
    sortby = _eigensort,
    scratch = nothing,
    team = nothing,
)
    lock(_matrix_lock) do
        side in (:left, :right, :both) ||
            throw(ArgumentError("side must be :left, :right or :both"))
        _handle(a)
        _handle(b)
        size(a)==size(b) ||
            throw(DimensionMismatch("generalized eigen input sizes must match"))
        T=promote_type(eltype(a), eltype(b))
        a, b=_coerce(a, T), _coerce(b, T)
        n=LinearAlgebra.checksquare(a)
        f=vectors ? LinearAlgebra.eigen : LinearAlgebra.eigvals
        req=scratch_requirements(f, a, b; team, side)
        temp=isnothing(scratch)
        work=temp ? FaerScratch(req.bytes; alignment = req.alignment) : scratch
        work isa FaerScratch || throw(ArgumentError("scratch must be a FaerScratch"))
        try
            cap=scratch_capacity(work)
            cap.bytes>=req.bytes && cap.alignment>=req.alignment || throw(
                ArgumentError("general EVD scratch capacity/alignment is insufficient"),
            )
            C=Complex{_realtype(T)}
            alpha=zeros(C, n)
            beta=zeros(T, n)
            eigenvectors=zeros(C, n, vectors ? n : 0)
            rightvectors=zeros(C, n, vectors && side===:both ? n : 0)
            GC.@preserve alpha beta eigenvectors rightvectors begin
                args=(
                    Ptr{UInt8}(pointer(eigenvectors)),
                    Ptr{UInt8}(pointer(rightvectors)),
                    Ptr{UInt8}(pointer(alpha)),
                    Ptr{UInt8}(pointer(beta)),
                    !vectors,
                    side===:left,
                    side===:both,
                    UInt(n),
                    _tag(T),
                )
                isnothing(team) ?
                _native_call(RustCrate.general_gevd_workspace, a, b, work, args...) :
                _native_call(
                    RustCrate.general_gevd_workspace_team,
                    a,
                    b,
                    work,
                    team,
                    args...,
                )
            end
            if vectors
                for matrix in
                    (side===:both ? (eigenvectors, rightvectors) : (eigenvectors,)),
                    j in axes(matrix, 2)

                    v=view(matrix, :, j)
                    scale=LinearAlgebra.norm(v)
                    iszero(scale) || (v ./= scale)
                end
            end
            values=homogeneous && isnothing(sortby) ? nothing : alpha ./ beta
            if !isnothing(sortby)
                order=sortperm(values; by = sortby)
                values=values[order]
                alpha, beta=alpha[order], beta[order]
                vectors && (eigenvectors=eigenvectors[:, order])
                vectors && side===:both && (rightvectors=rightvectors[:, order])
            end
            if vectors && side===:both
                return homogeneous ?
                       (α = alpha, β = beta, left = eigenvectors, right = rightvectors) :
                       (values = values, left = eigenvectors, right = rightvectors)
            end
            if homogeneous
                return vectors ? (α = alpha, β = beta, vectors = eigenvectors) :
                       (α = alpha, β = beta)
            end
            vectors && side===:left && return (values = values, vectors = eigenvectors)
            return vectors ? LinearAlgebra.GeneralizedEigen(values, eigenvectors) : values
        finally
            temp && close(work)
        end
    end
end
