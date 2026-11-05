function _bidiagonal_svd_plan(d, e, uplo, recursion_threshold)
    _tridiagonal_open(d)
    _tridiagonal_open(e)
    _dense_rhs_validate(d)
    _dense_rhs_validate(e)
    n=length(d)
    length(e)==max(0, n-1) ||
        throw(DimensionMismatch("offdiagonal length must be max(0,length(diag)-1)"))
    uplo in (:U, :L, 'U', 'L') || throw(ArgumentError("uplo must be :U or :L"))
    threshold=Int(recursion_threshold)
    threshold>=2 || throw(ArgumentError("recursion_threshold must be at least two"))
    T=_reflector_type(promote_type(eltype(d), eltype(e)))
    return n, T, threshold
end
function _bidiagonal_svd_requirement(n, T, threshold, team)
    info=zeros(UInt, 2)
    args=(UInt(n), _tag(T), UInt(threshold), pointer(info))
    GC.@preserve info begin
        isnothing(team) ? _native_call(RustCrate.bidiagonal_svd_requirements, args...) :
        _native_call(RustCrate.bidiagonal_svd_team_requirements, team, args...)
    end
    return (bytes = Int(info[1]), alignment = Int(info[2]))
end
"""
    scratch_requirements(bidiagonal_svd, d, e; uplo=:U, recursion_threshold=128, team=nothing)

Query dedicated square bidiagonal SVD workspace, including both coefficient
copies and the kernel's declared workspace at the actual team shape. Outputs
and Julia conversions allocate separately. Saved-factor back-transform products
use separate output/work storage. Values-only dispatch still computes full
native factors; it uses the same requirement.
"""
function scratch_requirements(
    ::typeof(bidiagonal_svd),
    d::AbstractVector{<:Number},
    e::AbstractVector{<:Number};
    uplo = :U,
    recursion_threshold::Integer = 128,
    team = nothing,
)
    lock(_matrix_lock) do
        _iterative_resource_check(nothing, team)
        n, T, threshold=_bidiagonal_svd_plan(d, e, uplo, recursion_threshold)
        return _bidiagonal_svd_requirement(n, T, threshold, team)
    end
end
function FaerScratch(
    ::typeof(bidiagonal_svd),
    d::AbstractVector{<:Number},
    e::AbstractVector{<:Number};
    kwargs...,
)
    req=scratch_requirements(bidiagonal_svd, d, e; kwargs...)
    return FaerScratch(req.bytes; alignment = req.alignment)
end
function reserve!(
    s::FaerScratch,
    ::typeof(bidiagonal_svd),
    d::AbstractVector{<:Number},
    e::AbstractVector{<:Number};
    kwargs...,
)
    req=scratch_requirements(bidiagonal_svd, d, e; kwargs...)
    return reserve!(s, req.bytes; alignment = req.alignment)
end
function _bidiagonal_factor_validate(f::FaerBidiagonalization; full::Bool = false)
    _dense_rhs_validate(f.U)
    _dense_rhs_validate(f.V)
    m, n=size(f)
    k=size(f.B, 1)
    k==min(m, n) && length(f.B.ev)==max(0, k-1) ||
        throw(DimensionMismatch("saved bidiagonal factor dimensions do not match"))
    size(f.U, 2) in (k, m) && size(f.V, 2) in (k, n) ||
        throw(DimensionMismatch("saved bidiagonal basis dimensions do not match"))
    !full ||
        (size(f.U, 2)==m && size(f.V, 2)==n) ||
        throw(ArgumentError("full SVD requires a reduction made with full=true"))
    return nothing
end
const _BidiagonalSvdFunction=Union{typeof(LinearAlgebra.svd),typeof(LinearAlgebra.svdvals)}
function scratch_requirements(
    f::_BidiagonalSvdFunction,
    a::Union{FaerBidiagonalization,_NativeBidiagonal};
    full::Bool = false,
    recursion_threshold::Integer = 128,
    team = nothing,
)
    lock(_matrix_lock) do
        _iterative_resource_check(nothing, team)
        a isa FaerBidiagonalization && _bidiagonal_factor_validate(a; full)
        b=a isa FaerBidiagonalization ? a.B : a
        return scratch_requirements(
            bidiagonal_svd,
            b.dv,
            b.ev;
            uplo = b.uplo,
            recursion_threshold,
            team,
        )
    end
end
function FaerScratch(
    f::_BidiagonalSvdFunction,
    a::Union{FaerBidiagonalization,_NativeBidiagonal};
    kwargs...,
)
    req=scratch_requirements(f, a; kwargs...)
    return FaerScratch(req.bytes; alignment = req.alignment)
end
function reserve!(
    s::FaerScratch,
    f::_BidiagonalSvdFunction,
    a::Union{FaerBidiagonalization,_NativeBidiagonal};
    kwargs...,
)
    req=scratch_requirements(f, a; kwargs...)
    return reserve!(s, req.bytes; alignment = req.alignment)
end
_bidiagonal_product(team, a, b) = isnothing(team) ? a*b : team_mul(team, a, b)
function _bidiagonal_values(d, e; kwargs...)
    core=bidiagonal_svd(d, e; kwargs...)
    try
        return core.S
    finally
        close(core.U)
        close(core.Vt)
    end
end
