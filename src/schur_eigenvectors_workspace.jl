const _SchurEigenFunction=Union{typeof(LinearAlgebra.eigen),typeof(LinearAlgebra.eigvecs)}
function _schur_eigen_requirement(n, T, team)
    info=zeros(UInt, 2)
    GC.@preserve info begin
        args=(UInt(n), _tag(T), pointer(info))
        isnothing(team) ? _native_call(RustCrate.schur_eigenvectors_requirements, args...) :
        _native_call(RustCrate.schur_eigenvectors_team_requirements, team, args...)
    end
    (bytes = Int(info[1]), alignment = Int(info[2]))
end
"""
    scratch_requirements(eigvecs, F::Schur{T,<:FaerMatrix}; team=nothing)
    scratch_requirements(eigen, F::Schur{T,<:FaerMatrix}; team=nothing)

Return declared native workspace for right eigenvectors of a saved Schur factor.
Includes real packed output plus faer's eigenvector kernel requirement. Current
native metadata and resources are validated before output setup. Requirements
are width-independent; execution passes the actual token to back substitution
and back transformation. Owned output and Julia eigenvalue copies allocate
separately. Selected faer's internal extreme-scale Schur copy can still allocate.
"""
function scratch_requirements(::_SchurEigenFunction, f::_NativeSchur; team = nothing)
    lock(_matrix_lock) do
        n=_schur_order_plan(f, team)
        _schur_eigen_requirement(n, eltype(f), team)
    end
end
function FaerScratch(fun::_SchurEigenFunction, f::_NativeSchur; team = nothing)
    req=scratch_requirements(fun, f; team)
    FaerScratch(req.bytes; alignment = req.alignment)
end
function reserve!(s::FaerScratch, fun::_SchurEigenFunction, f::_NativeSchur; team = nothing)
    req=scratch_requirements(fun, f; team)
    reserve!(s, req.bytes; alignment = req.alignment)
end
function _saved_schur_eigenvectors(f::_NativeSchur; scratch = nothing, team = nothing)
    lock(_matrix_lock) do
        isnothing(scratch)||scratch isa FaerScratch ||
            throw(ArgumentError("scratch must be a FaerScratch"))
        n=_schur_order_plan(f, team)
        T=eltype(f)
        req=_schur_eigen_requirement(n, T, team)
        work=isnothing(scratch) ? FaerScratch(req.bytes; alignment = req.alignment) :
             scratch
        try
            cap=scratch_capacity(work)
            cap.bytes>=req.bytes&&cap.alignment>=req.alignment || throw(
                ArgumentError(
                    "Saved Schur eigenvectors scratch capacity/alignment is insufficient; call reserve! before computing",
                ),
            )
            fn=if T===Float64
                isnothing(team) ? RustCrate.schur_eigenvectors_owned_workspace :
                RustCrate.schur_eigenvectors_owned_workspace_team
            else
                isnothing(team) ? RustCrate.schur_eigenvectors_typed_workspace :
                RustCrate.schur_eigenvectors_typed_workspace_team
            end
            args=(f.T, f.Z, work, UInt(n), _tag(T))
            args=isnothing(team) ? args : (args..., team)
            _typed_result(Complex{_realtype(T)}, fn, (n, n), args...)
        finally
            isnothing(scratch) && close(work)
        end
    end
end
