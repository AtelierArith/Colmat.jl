function _schur_order_plan(f::_NativeSchur, team)
    isnothing(team)||team isa FaerTeam || throw(ArgumentError("team must be a FaerTeam"))
    isnothing(team)||_handle(team)
    # Validate erased native owners before _schur_selection reads any coefficients.
    _schur_validate(f.T)
    _schur_validate(f.Z)
    _check_schur(f)
end
function _schur_order_requirement(n, T, team)
    info=zeros(UInt, 2)
    GC.@preserve info begin
        args=(UInt(n), _tag(T), pointer(info))
        isnothing(team) ? _native_call(RustCrate.schur_reorder_requirements, args...) :
        _native_call(RustCrate.schur_reorder_team_requirements, team, args...)
    end
    (bytes = Int(info[1]), alignment = Int(info[2]))
end
"""
    scratch_requirements(ordschur, F::Schur{T,<:FaerMatrix}; team=nothing)
    scratch_requirements(ordschur!, F::Schur{T,<:FaerMatrix}; team=nothing)

Query declared saved Schur reorder workspace, including native selection flags
and block-swap scratch. Both functions have the same requirement. Use this query
with `FaerScratch` or `reserve!`. Independent work factors and Julia selection,
eigenvalue and conversion arrays allocate separately. A supplied team is checked
and lends its token; selected faer's reorder kernels currently run sequentially
and ignore that token, so requirements do not depend on width.
"""
function scratch_requirements(::_QzOrderFunction, f::_NativeSchur; team = nothing)
    lock(_matrix_lock) do
        n=_schur_order_plan(f, team)
        _schur_order_requirement(n, eltype(f), team)
    end
end
function FaerScratch(fun::_QzOrderFunction, f::_NativeSchur; team = nothing)
    req=scratch_requirements(fun, f; team)
    FaerScratch(req.bytes; alignment = req.alignment)
end
function reserve!(s::FaerScratch, fun::_QzOrderFunction, f::_NativeSchur; team = nothing)
    req=scratch_requirements(fun, f; team)
    reserve!(s, req.bytes; alignment = req.alignment)
end
function _schur_order_result(f::_NativeSchur, data, n, work, team)
    T=eltype(f)
    z=_zeros(T, (n, n))
    values=Vector{Complex{_realtype(T)}}(undef, n)
    fn=isnothing(team) ? RustCrate.schur_reorder_workspace :
       RustCrate.schur_reorder_workspace_team
    try
        h=GC.@preserve data values begin
            args=(
                f.T,
                f.Z,
                z,
                work,
                pointer(data),
                UInt(n),
                Ptr{UInt8}(pointer(values)),
                _tag(T),
            )
            args=isnothing(team) ? args : (args..., team)
            T===Float64 ? _matrix_result(fn, (n, n), args...) :
            _typed_result(T, fn, (n, n), args...)
        end
        LinearAlgebra.Schur(h, z, values)
    catch
        close(z)
        rethrow()
    end
end
function _ordschur(f::_NativeSchur, select; scratch = nothing, team = nothing)
    lock(_matrix_lock) do
        isnothing(scratch)||scratch isa FaerScratch ||
            throw(ArgumentError("scratch must be a FaerScratch"))
        n=_schur_order_plan(f, team)
        req=_schur_order_requirement(n, eltype(f), team)
        work=isnothing(scratch) ? FaerScratch(req.bytes; alignment = req.alignment) :
             scratch
        try
            cap=scratch_capacity(work)
            cap.bytes>=req.bytes&&cap.alignment>=req.alignment || throw(
                ArgumentError(
                    "Schur reorder scratch capacity/alignment is insufficient; call reserve! before computing",
                ),
            )
            data=_schur_selection(f, select)
            _schur_order_result(f, data, n, work, team)
        finally
            isnothing(scratch) && close(work)
        end
    end
end
