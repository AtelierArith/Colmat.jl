const _SvdFunction=Union{
    typeof(LinearAlgebra.svd),
    typeof(LinearAlgebra.svdvals),
    typeof(LinearAlgebra.svd!),
    typeof(LinearAlgebra.svdvals!),
}
"""
    scratch_requirements(svd, A; full=false, team=nothing)
    scratch_requirements(svdvals, A; team=nothing)

Query byte count/alignment for the chosen SVD mode and team width. Use identical
options for the computation. The requirement includes temporary singular-value
storage for complex inputs. Factor output arrays are separate Julia allocations.
"""
function scratch_requirements(
    f::_SvdFunction,
    a::FaerMatrix{T};
    full::Bool = false,
    team = nothing,
) where {T}
    lock(_matrix_lock) do
        _handle(a)
        isnothing(team) ||
            team isa FaerTeam ||
            throw(ArgumentError("team must be a FaerTeam"))
        !isnothing(team) && _handle(team)
        width=isnothing(team) ? 1 : team.width
        info=zeros(UInt, 2)
        GC.@preserve info begin
            args=(
                UInt(size(a, 1)),
                UInt(size(a, 2)),
                _tag(T),
                full,
                f in (LinearAlgebra.svdvals, LinearAlgebra.svdvals!),
            )
            isnothing(team) ?
            _native_call(RustCrate.svd_requirements, args..., UInt(width), info) :
            _native_call(RustCrate.svd_team_requirements, team, args..., info)
        end
        return (bytes = Int(info[1]), alignment = Int(info[2]))
    end
end
function FaerScratch(f::_SvdFunction, a::FaerMatrix; kwargs...)
    req=scratch_requirements(f, a; kwargs...)
    return FaerScratch(req.bytes; alignment = req.alignment)
end
function reserve!(s::FaerScratch, f::_SvdFunction, a::FaerMatrix; kwargs...)
    req=scratch_requirements(f, a; kwargs...)
    return reserve!(s, req.bytes; alignment = req.alignment)
end
function _svd_workspace_result(
    a::FaerMatrix{T};
    full::Bool = false,
    values_only::Bool = false,
    scratch = nothing,
    team = nothing,
) where {T}
    lock(_matrix_lock) do
        _handle(a)
        isnothing(team) ||
            team isa FaerTeam ||
            throw(ArgumentError("team must be a FaerTeam"))
        f=values_only ? LinearAlgebra.svdvals : LinearAlgebra.svd
        req=scratch_requirements(f, a; full, team)
        temp=isnothing(scratch)
        work=temp ? FaerScratch(req.bytes; alignment = req.alignment) : scratch
        work isa FaerScratch || throw(ArgumentError("scratch must be a FaerScratch"))
        try
            cap=scratch_capacity(work)
            cap.bytes>=req.bytes && cap.alignment>=req.alignment || throw(
                ArgumentError(
                    "SVD scratch capacity/alignment is insufficient; reserve for the selected mode/team",
                ),
            )
            m, n=size(a)
            k=min(m, n)
            u=zeros(T, m, values_only ? 0 : full ? m : k)
            v=zeros(T, n, values_only ? 0 : full ? n : k)
            singular=zeros(_realtype(T), k)
            GC.@preserve u v singular begin
                args=(
                    Ptr{UInt8}(pointer(u)),
                    Ptr{UInt8}(pointer(singular)),
                    Ptr{UInt8}(pointer(v)),
                    full,
                    values_only,
                    UInt(m),
                    UInt(n),
                    _tag(T),
                )
                _native_call(RustCrate.svd_workspace, a, work, team, args...)
            end
            return values_only ? singular : SVD(u, singular, copy(adjoint(v)))
        finally
            temp && close(work)
        end
    end
end
"""
    svd!(A::FaerMatrix; full=false, scratch=nothing, team=nothing)
    svdvals!(A::FaerMatrix; scratch=nothing, team=nothing)

Compute through faer's low-level SVD and replace A by rectangular Sigma only
on success. Return independent Julia SVD factors or real singular values.
Existing same-shape views see Sigma; returned factors survive changes/close of A.
This explicitly defined writeback is not LAPACK's unspecified work matrix.
Input/output staging and Julia bookkeeping allocate; scratch may be reused.
"""
function LinearAlgebra.svd!(
    a::FaerMatrix;
    full::Bool = false,
    scratch = nothing,
    team = nothing,
    alg = nothing,
)
    alg===nothing || return _svd_alg(()->LinearAlgebra.svd!(a; full, scratch, team), alg)
    lock(_matrix_lock) do
        F=_svd_workspace_result(a; full, scratch, team)
        _svd_writeback!(a, F.S)
        return F
    end
end
function _svd_writeback!(a::FaerMatrix{T}, s) where {T}
    sigma=zeros(T, size(a))
    for i in eachindex(s)
        sigma[i, i]=s[i]
    end
    copyto!(a, FaerMatrix(sigma))
    return a
end
function LinearAlgebra.svdvals!(
    a::FaerMatrix;
    scratch = nothing,
    team = nothing,
    alg = nothing,
)
    alg===nothing || return _svd_alg(()->LinearAlgebra.svdvals!(a; scratch, team), alg)
    lock(_matrix_lock) do
        s=_svd_workspace_result(a; values_only = true, scratch, team)
        _svd_writeback!(a, s)
        return s
    end
end
