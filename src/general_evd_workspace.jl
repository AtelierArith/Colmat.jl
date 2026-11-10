"""
    scratch_requirements(eigen, A::FaerMatrix; team=nothing)
    scratch_requirements(eigvals, A::FaerMatrix; team=nothing)

General EVD workspace query; bang aliases use the same computational mode.
Real input requires scratch for packed real/imaginary eigenvectors and values.
The actual team conditions are queried. Output arrays and sorting allocate;
bang Schur-form writeback additionally computes an independent Schur factor.
"""
function scratch_requirements(
    f::_EvdFunction, a::FaerMatrix; team=nothing, side::Symbol=:right
)
    lock(_matrix_lock) do
        side in (:left, :right, :both) ||
            throw(ArgumentError("side must be :left, :right or :both"))
        _handle(a)
        n = LinearAlgebra.checksquare(a)
        isnothing(team) ||
            team isa FaerTeam ||
            throw(ArgumentError("team must be a FaerTeam"))
        !isnothing(team) && _handle(team)
        info = zeros(UInt, 2)
        args = (
            UInt(n),
            _tag(eltype(a)),
            f in (LinearAlgebra.eigvals, LinearAlgebra.eigvals!),
            Int32(
                if side === :right
                    0
                elseif side === :left
                    1
                else
                    2
                end,
            ),
        )
        GC.@preserve info begin
            _native_call(RustCrate.general_evd_requirements, args..., info, team)
        end
        return (bytes=Int(info[1]), alignment=Int(info[2]))
    end
end
function FaerScratch(f::_EvdFunction, a::FaerMatrix; kwargs...)
    req = scratch_requirements(f, a; kwargs...)
    return FaerScratch(req.bytes; alignment=req.alignment)
end
function reserve!(s::FaerScratch, f::_EvdFunction, a::FaerMatrix; kwargs...)
    req = scratch_requirements(f, a; kwargs...)
    return reserve!(s, req.bytes; alignment=req.alignment)
end
function _general_evd_workspace_result(
    a::FaerMatrix{T};
    vectors::Bool=true,
    side::Symbol=:right,
    sortby=_eigensort,
    scratch=nothing,
    team=nothing,
) where {T}
    lock(_matrix_lock) do
        side in (:left, :right, :both) ||
            throw(ArgumentError("side must be :left, :right or :both"))
        _handle(a)
        n = LinearAlgebra.checksquare(a)
        f = vectors ? LinearAlgebra.eigen : LinearAlgebra.eigvals
        req = scratch_requirements(f, a; team, side)
        temp = isnothing(scratch)
        work = temp ? FaerScratch(req.bytes; alignment=req.alignment) : scratch
        work isa FaerScratch || throw(ArgumentError("scratch must be a FaerScratch"))
        try
            cap = scratch_capacity(work)
            cap.bytes >= req.bytes && cap.alignment >= req.alignment || throw(
                ArgumentError("general EVD scratch capacity/alignment is insufficient")
            )
            C = Complex{_realtype(T)}
            values = zeros(C, n)
            eigenvectors = zeros(C, n, vectors ? n : 0)
            rightvectors = zeros(C, n, vectors && side === :both ? n : 0)
            GC.@preserve values eigenvectors rightvectors begin
                args = (
                    Ptr{UInt8}(pointer(eigenvectors)),
                    Ptr{UInt8}(pointer(rightvectors)),
                    Ptr{UInt8}(pointer(values)),
                    !vectors,
                    Int32(
                        if side === :right
                            0
                        elseif side === :left
                            1
                        else
                            2
                        end,
                    ),
                    UInt(n),
                    _tag(T),
                )
                _native_call(RustCrate.general_evd_workspace, a, work, team, args...)
            end
            if vectors
                for matrix in
                    (side === :both ? (eigenvectors, rightvectors) : (eigenvectors,)),
                    j in axes(matrix, 2)

                    v = view(matrix, :, j)
                    scale = LinearAlgebra.norm(v)
                    iszero(scale) || (v ./= scale)
                end
            end
            if !isnothing(sortby)
                order = sortperm(values; by=sortby)
                values = values[order]
                vectors && (eigenvectors = eigenvectors[:, order])
                vectors && side === :both && (rightvectors = rightvectors[:, order])
            end
            vectors &&
                side === :both &&
                return (values=values, left=eigenvectors, right=rightvectors)
            vectors && side === :left && return (values=values, vectors=eigenvectors)
            return vectors ? LinearAlgebra.Eigen(values, eigenvectors) : values
        finally
            temp && close(work)
        end
    end
end
