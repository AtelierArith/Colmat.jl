"""
    scratch_requirements(eigen, A::FaerMatrix, B::FaerMatrix; team=nothing)

Generalized EVD workspace including writable A/B copies and packed real
outputs where required. Bang queries describe the eigen pass; the separate
QZ writeback uses its own sequential workspace. Output and sorting allocate.
Exceptional repeated-eigenvalue vector recovery uses native sequential SVD
with additional internal workspace outside this requirement.
"""
function scratch_requirements(f::_EvdFunction, a::FaerMatrix, b::FaerMatrix; team = nothing)
    lock(_matrix_lock) do
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
    sortby = _eigensort,
    scratch = nothing,
    team = nothing,
)
    lock(_matrix_lock) do
        _handle(a)
        _handle(b)
        size(a)==size(b) ||
            throw(DimensionMismatch("generalized eigen input sizes must match"))
        T=promote_type(eltype(a), eltype(b))
        a, b=_coerce(a, T), _coerce(b, T)
        n=LinearAlgebra.checksquare(a)
        f=vectors ? LinearAlgebra.eigen : LinearAlgebra.eigvals
        req=scratch_requirements(f, a, b; team)
        temp=isnothing(scratch)
        work=temp ? FaerScratch(req.bytes; alignment = req.alignment) : scratch
        work isa FaerScratch || throw(ArgumentError("scratch must be a FaerScratch"))
        try
            cap=scratch_capacity(work)
            cap.bytes>=req.bytes && cap.alignment>=req.alignment || throw(
                ArgumentError("general EVD scratch capacity/alignment is insufficient"),
            )
            C=Complex{_realtype(T)}
            values=zeros(C, n)
            eigenvectors=zeros(C, n, vectors ? n : 0)
            GC.@preserve values eigenvectors begin
                args=(
                    Ptr{UInt8}(pointer(eigenvectors)),
                    Ptr{UInt8}(pointer(values)),
                    !vectors,
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
            vectors && _repair_generalized_vectors!(eigenvectors, values, a, b)
            if vectors
                for j in axes(eigenvectors, 2)
                    v=view(eigenvectors, :, j)
                    scale=LinearAlgebra.norm(v)
                    iszero(scale) || (v ./= scale)
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

# faer solve_shifted_1x1 can form 0 * inv(0) for repeated eigenvalues.
# Recover only invalid finite-eigenvalue columns using native SVD nullspaces.
function _repair_generalized_vectors!(vectors, values, a, b)
    bad=findall(j->!all(isfinite, view(vectors, :, j)), axes(vectors, 2))
    isempty(bad) && return vectors
    R=_realtype(eltype(vectors))
    threshold=100eps(R)*max(size(a, 1), 1)
    A, B=Matrix(a), Matrix(b)
    pending=Set(bad)
    while !isempty(pending)
        j=first(pending)
        λ=values[j]
        isfinite(λ) || throw(
            ArgumentError(
                "native generalized eigenvectors are nonfinite for a nonfinite eigenvalue",
            ),
        )
        group=findall(x->isapprox(x, λ; rtol = threshold, atol = threshold), values)
        matrix=FaerMatrix(A-λ*B)
        F=try
            LinearAlgebra.svd(matrix)
        finally
            close(matrix)
        end
        k=length(group)
        scale=max(LinearAlgebra.norm(A), abs(λ)*LinearAlgebra.norm(B), one(R))
        count(x->x<=threshold*scale, F.S)>=k || throw(
            ArgumentError(
                "native generalized eigenvector recovery found insufficient nullity",
            ),
        )
        vectors[:, group]=F.V[:, (end-k+1):end]
        for i in group
            delete!(pending, i)
        end
    end
    return vectors
end
