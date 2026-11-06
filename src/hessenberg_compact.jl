"""
    FaerHessenbergQ

Implicit unitary Q from `hessenberg(A; compact=true)`. Uses saved native
Householder factors without forming dense Q. Basis/factor properties return
independent Julia arrays. Transpose/adjoint/conj share immutable native storage;
copy/deepcopy duplicate it, and each Julia wrapper closes independently.
"""
mutable struct FaerHessenbergQ{T,H} <: AbstractFaerCompactQ{T}
    handle::Union{Nothing,H}
    n::Int
    k::Int
    blocksize::Int
    transposed::Bool
    conjugated::Bool
end
_compact_q_wrapper(::FaerHessenbergQ) = FaerHessenbergQ
_schur_validate(q::FaerHessenbergQ) = _compact_q_validate(q)

"""
    FaerCompactHessenberg

Hessenberg factor with native H and implicit Q, returned by
`hessenberg(A; compact=true)`. The bang reduction shares H with A. Saved `schur`
applies Q implicitly; `schur!` updates H and replaces Q with the explicit Schur
basis, returning a Schur object sharing both. Previously retained implicit Q
wrappers remain valid snapshots after that replacement.
"""
mutable struct FaerCompactHessenberg{T,M<:FaerMatrix{T},QT<:FaerHessenbergQ{T}} <:
               LinearAlgebra.Factorization{T}
    const H::M
    Q::Union{QT,M}
end
Base.size(f::FaerCompactHessenberg) = size(f.H)
Base.size(f::FaerCompactHessenberg, d::Integer) = size(f.H, d)
Base.isopen(f::FaerCompactHessenberg) = isopen(f.H)&&isopen(f.Q)
function Base.close(f::FaerCompactHessenberg)
    lock(_matrix_lock) do
        close(f.H)
        close(f.Q)
    end
    return nothing
end
function Base.copy(f::FaerCompactHessenberg)
    lock(_matrix_lock) do
        _schur_plan(f.H, f, nothing)
        return typeof(f)(copy(f.H), copy(f.Q))
    end
end
function Base.deepcopy_internal(f::FaerCompactHessenberg, d::IdDict)
    get!(d, f) do
        lock(_matrix_lock) do
            _schur_plan(f.H, f, nothing)
            return typeof(f)(Base.deepcopy_internal(f.H, d), Base.deepcopy_internal(f.Q, d))
        end
    end
end
FaerMatrix(f::FaerCompactHessenberg) = lock(()->f.Q*f.H*f.Q', _matrix_lock)
Base.Matrix(f::FaerCompactHessenberg) = Matrix(FaerMatrix(f))
Base.Array(f::FaerCompactHessenberg) = Matrix(f)
Base.iterate(f::FaerCompactHessenberg) = (f.H, Val(:Q))
Base.iterate(f::FaerCompactHessenberg, ::Val{:Q}) = (f.Q, Val(:done))
Base.iterate(f::FaerCompactHessenberg, ::Val{:done}) = nothing
function _compact_hessenberg_requirement(n, T, team)
    info=zeros(UInt, 2)
    GC.@preserve info begin
        isnothing(team) ?
        _native_call(
            RustCrate.compact_hessenberg_requirements,
            UInt(n),
            _tag(T),
            pointer(info),
        ) :
        _native_call(
            RustCrate.compact_hessenberg_team_requirements,
            team,
            UInt(n),
            _tag(T),
            pointer(info),
        )
    end
    return (bytes = Int(info[1]), alignment = Int(info[2]))
end
function _compact_hessenberg(a; scratch = nothing, team = nothing)
    lock(_matrix_lock) do
        _iterative_resource_check(scratch, team)
        n=_schur_plan(a, nothing, team)
        T=eltype(a)
        req=_compact_hessenberg_requirement(n, T, team)
        work=isnothing(scratch) ? FaerScratch(req.bytes; alignment = req.alignment) :
             scratch
        try
            cap=scratch_capacity(work)
            cap.bytes>=req.bytes&&cap.alignment>=req.alignment||throw(
                ArgumentError(
                    "compact Hessenberg scratch capacity/alignment is insufficient; call reserve!",
                ),
            )
            data=Matrix{T}(a)
            data isa Matrix{T}&&size(data)==(n, n)||throw(
                DimensionMismatch(
                    "Hessenberg input conversion changed scalar storage or shape",
                ),
            )
            out=Matrix{T}(undef, n, n)
            h=GC.@preserve data out begin
                args=(
                    work,
                    Ptr{UInt8}(pointer(data)),
                    UInt(n),
                    Ptr{UInt8}(pointer(out)),
                    _tag(T),
                )
                isnothing(team) ? _native_call(RustCrate.OwnedCompactTridiagonal, args...) :
                _native_call(RustCrate.OwnedCompactTridiagonal, args..., team)
            end
            q=_compact_q(T, h; wrapper = FaerHessenbergQ)
            return FaerCompactHessenberg(FaerMatrix(out), q)
        finally
            isnothing(scratch)&&close(work)
        end
    end
end

const _SavedCompactSchurFunction=Union{
    typeof(LinearAlgebra.schur),
    typeof(LinearAlgebra.schur!),
}
function scratch_requirements(
    ::_SavedCompactSchurFunction,
    f::FaerCompactHessenberg;
    team = nothing,
)
    lock(_matrix_lock) do
        n=_schur_plan(f.H, f, team)
        req=_schur_requirement(n, eltype(f), 2, team)
        if f.Q isa FaerHessenbergQ
            qreq=_householder_workspace_query(f.Q, n, false; team)
            return (
                bytes = max(req.bytes, qreq.bytes),
                alignment = max(req.alignment, qreq.alignment),
            )
        end
        return req
    end
end
function FaerScratch(fun::_SavedCompactSchurFunction, f::FaerCompactHessenberg; kwargs...)
    req=scratch_requirements(fun, f; kwargs...)
    return FaerScratch(req.bytes; alignment = req.alignment)
end
function reserve!(
    work::FaerScratch,
    fun::_SavedCompactSchurFunction,
    f::FaerCompactHessenberg;
    kwargs...,
)
    req=scratch_requirements(fun, f; kwargs...)
    return reserve!(work, req.bytes; alignment = req.alignment)
end
function _schur_identity_result(a::FaerMatrix{T}, n, work, team) where {T}
    q=_zeros(T, (n, n))
    values=Vector{Complex{_realtype(T)}}(undef, n)
    fn=isnothing(team) ? RustCrate.schur_identity_workspace :
       RustCrate.schur_identity_workspace_team
    try
        h=GC.@preserve values begin
            args=(a, q, work, Ptr{UInt8}(pointer(values)), UInt(n), _tag(T))
            args=isnothing(team) ? args : (args..., team)
            T===Float64 ? _matrix_result(fn, (n, n), args...) :
            _typed_result(T, fn, (n, n), args...)
        end
        return LinearAlgebra.Schur(h, q, values)
    catch
        close(q)
        rethrow()
    end
end
function LinearAlgebra.schur(f::FaerCompactHessenberg; scratch = nothing, team = nothing)
    lock(_matrix_lock) do
        _iterative_resource_check(scratch, team)
        n=_schur_plan(f.H, f, team)
        if f.Q isa FaerMatrix
            return _saved_schur(FaerHessenberg(f.H, f.Q); scratch, team)
        end
        req=scratch_requirements(LinearAlgebra.schur, f; team)
        work=isnothing(scratch) ? FaerScratch(req.bytes; alignment = req.alignment) :
             scratch
        result=nothing
        success=false
        try
            cap=scratch_capacity(work)
            cap.bytes>=req.bytes&&cap.alignment>=req.alignment||throw(
                ArgumentError(
                    "compact saved Schur scratch capacity/alignment is insufficient; call reserve!",
                ),
            )
            result=_schur_identity_result(f.H, n, work, team)
            # Z is a fresh unpublished result: apply directly, without the
            # transactional extra staging needed for arbitrary user arrays.
            args=(
                result.Z,
                f.Q,
                work,
                _householder_workspace_metadata(f.Q)...,
                false,
                f.Q.transposed,
                f.Q.conjugated,
            )
            isnothing(team) ?
            _native_call(RustCrate.compact_tridiagonal_workspace, args...) :
            _native_call(RustCrate.compact_tridiagonal_workspace_team, args..., team)
            success=true
            return result
        finally
            if !success&&!isnothing(result)
                close(result.T)
                close(result.Z)
            end
            isnothing(scratch)&&close(work)
        end
    end
end
function LinearAlgebra.schur!(f::FaerCompactHessenberg; scratch = nothing, team = nothing)
    lock(_matrix_lock) do
        f.H===f.Q&&throw(ArgumentError("H and Q must have distinct storage"))
        result=LinearAlgebra.schur(f; scratch, team)
        adopted=false
        try
            # Check the field's concrete binding type before the first write.
            # A reload cannot turn an assignment failure into a partial update.
            result.Z isa typeof(f.H)||throw(
                ArgumentError(
                    "Schur result native binding differs; recreate the factor after reload",
                ),
            )
            copyto!(f.H, result.T)
            if f.Q isa FaerHessenbergQ
                f.Q=result.Z
                adopted=true
            else
                copyto!(f.Q, result.Z)
            end
            return LinearAlgebra.Schur(f.H, f.Q, result.values)
        finally
            close(result.T)
            adopted||close(result.Z)
        end
    end
end
