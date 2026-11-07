const _TridiagonalEigenFunction=Union{typeof(tridiagonal_eigen),typeof(tridiagonal_eigvals)}
function _tridiagonal_evd_requirement(n, T, vectors, threshold, team)
    info=zeros(UInt, 2)
    args=(UInt(n), _tag(T), vectors, UInt(threshold), info)
    GC.@preserve info begin
        _native_call(RustCrate.tridiagonal_evd_requirements, args..., team)
    end
    return (bytes = Int(info[1]), alignment = Int(info[2]))
end
"""
    scratch_requirements(tridiagonal_eigen, d, e; recursion_threshold=128, team=nothing)
    scratch_requirements(tridiagonal_eigvals, d, e; recursion_threshold=128, team=nothing)

Query dedicated Hermitian tridiagonal EVD workspace. Includes the scaled band,
temporary eigenvalues and the kernel's declared requirement at the actual team
shape. Coefficient conversion, output and sorting arrays allocate separately.
"""
function scratch_requirements(
    f::_TridiagonalEigenFunction,
    d::AbstractVector{<:Number},
    e::AbstractVector{<:Number};
    recursion_threshold::Integer = 128,
    team = nothing,
)
    lock(_matrix_lock) do
        _iterative_resource_check(nothing, team)
        _tridiagonal_open(d)
        _tridiagonal_open(e)
        _dense_rhs_validate(d)
        _dense_rhs_validate(e)
        n=length(d)
        length(e)==max(0, n-1) ||
            throw(DimensionMismatch("subdiagonal length must be max(0,length(diag)-1)"))
        T=_reflector_type(promote_type(eltype(d), eltype(e)))
        threshold=Int(recursion_threshold)
        threshold>=2 || throw(ArgumentError("recursion_threshold must be at least two"))
        return _tridiagonal_evd_requirement(n, T, f===tridiagonal_eigen, threshold, team)
    end
end
function FaerScratch(
    f::_TridiagonalEigenFunction,
    d::AbstractVector{<:Number},
    e::AbstractVector{<:Number};
    kwargs...,
)
    req=scratch_requirements(f, d, e; kwargs...)
    return FaerScratch(req.bytes; alignment = req.alignment)
end
function reserve!(
    s::FaerScratch,
    f::_TridiagonalEigenFunction,
    d::AbstractVector{<:Number},
    e::AbstractVector{<:Number};
    kwargs...,
)
    req=scratch_requirements(f, d, e; kwargs...)
    return reserve!(s, req.bytes; alignment = req.alignment)
end
const _BandEigenFunction=Union{typeof(LinearAlgebra.eigen),typeof(LinearAlgebra.eigvals)}
function scratch_requirements(
    f::_BandEigenFunction,
    a::Union{FaerTridiagonalization,_NativeSymTridiagonal};
    kwargs...,
)
    lock(_matrix_lock) do
        if a isa FaerTridiagonalization
            _dense_rhs_validate(a.Q)
            size(a.Q)==size(a.H) ||
                throw(DimensionMismatch("saved tridiagonal Q/H shapes do not match"))
        end
        h=a isa FaerTridiagonalization ? a.H : a
        fun=f===LinearAlgebra.eigen ? tridiagonal_eigen : tridiagonal_eigvals
        req=scratch_requirements(fun, h.dv, h.ev; kwargs...)
        if f===LinearAlgebra.eigen &&
           a isa FaerTridiagonalization &&
           a.Q isa FaerTridiagonalQ
            applyreq=_householder_workspace_query(
                a.Q,
                size(a.Q, 1),
                false;
                team = get(kwargs, :team, nothing),
            )
            return (
                bytes = max(req.bytes, applyreq.bytes),
                alignment = max(req.alignment, applyreq.alignment),
            )
        end
        return req
    end
end
function FaerScratch(
    f::_BandEigenFunction,
    a::Union{FaerTridiagonalization,_NativeSymTridiagonal};
    kwargs...,
)
    req=scratch_requirements(f, a; kwargs...)
    return FaerScratch(req.bytes; alignment = req.alignment)
end
function reserve!(
    s::FaerScratch,
    f::_BandEigenFunction,
    a::Union{FaerTridiagonalization,_NativeSymTridiagonal};
    kwargs...,
)
    req=scratch_requirements(f, a; kwargs...)
    return reserve!(s, req.bytes; alignment = req.alignment)
end
