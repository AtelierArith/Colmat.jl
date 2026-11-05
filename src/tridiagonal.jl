"""
    tridiagonalize(A; blocksize=0) -> FaerTridiagonalization

Reduce a finite Hermitian matrix through faer's native tridiagonalization.
The result owns explicit native Q and a real `SymTridiagonal` H, satisfying
`A ≈ Q*H*Q'`. Complex off-diagonal phases are absorbed into Q. Zero blocksize
selects faer's recommendation; compact reflector storage is currently internal.
"""
struct FaerTridiagonalization{T,R,M<:FaerMatrix{T}} <: LinearAlgebra.Factorization{T}
    H::LinearAlgebra.SymTridiagonal{R,Vector{R}}
    Q::M
end
Base.size(f::FaerTridiagonalization) = size(f.Q)
Base.size(f::FaerTridiagonalization, d::Integer) = size(f.Q, d)
Base.isopen(f::FaerTridiagonalization) = isopen(f.Q)
Base.close(f::FaerTridiagonalization) = close(f.Q)
function Base.copy(f::FaerTridiagonalization)
    lock(_matrix_lock) do
        _handle(f.Q)
        return FaerTridiagonalization(copy(f.H), copy(f.Q))
    end
end
Base.deepcopy_internal(f::FaerTridiagonalization, d::IdDict) = get!(() -> copy(f), d, f)
FaerMatrix(f::FaerTridiagonalization) =
    lock(()->f.Q*FaerMatrix{eltype(f.Q)}(Matrix(f.H))*f.Q', _matrix_lock)
Base.Matrix(f::FaerTridiagonalization) = Matrix(FaerMatrix(f))
Base.Array(f::FaerTridiagonalization) = Matrix(f)
function _tridiagonal_open(a)
    a isa _RotationArray && _rotation_open(a)
    return nothing
end
function _tridiagonal_open(a::Union{LinearAlgebra.Hermitian,LinearAlgebra.Symmetric})
    return _tridiagonal_open(parent(a))
end
function tridiagonalize(a::AbstractMatrix{<:Number}; blocksize::Integer = 0)
    lock(_matrix_lock) do
        _tridiagonal_open(a)
        n=LinearAlgebra.checksquare(a)
        T=_reflector_type(eltype(a))
        R=_realtype(T)
        data=Matrix{T}(a)
        all(isfinite, data) ||
            throw(ArgumentError("tridiagonalization input must be finite"))
        LinearAlgebra.ishermitian(data) ||
            throw(ArgumentError("tridiagonalization requires Hermitian input"))
        bs=Int(blocksize)
        0<=bs<=max(1, n-1) || throw(
            ArgumentError("blocksize must be zero (automatic) or between 1 and max(1,n-1)"),
        )
        q=zeros(T, n, n)
        diag=zeros(R, n)
        off=zeros(R, max(0, n-1))
        GC.@preserve data q diag off _native_call(
            RustCrate.tridiagonalize_data,
            Ptr{UInt8}(pointer(data)),
            UInt(n),
            Ptr{UInt8}(pointer(q)),
            Ptr{UInt8}(pointer(diag)),
            Ptr{UInt8}(pointer(off)),
            _tag(T),
            UInt(bs),
        )
        all(isfinite, q)&&all(isfinite, diag)&&all(isfinite, off) ||
            throw(ArgumentError("tridiagonalization overflowed the working type"))
        return FaerTridiagonalization(
            LinearAlgebra.SymTridiagonal(diag, off),
            FaerMatrix(q),
        )
    end
end
"""Stage native tridiagonalization, then replace A by explicit H; Q is independently owned."""
function tridiagonalize!(a::FaerMatrix; kwargs...)
    lock(_matrix_lock) do
        f=tridiagonalize(a; kwargs...)
        copyto!(a, Matrix(f.H))
        return f
    end
end
const _TridiagonalHermitian=LinearAlgebra.Hermitian{
    T,
    M,
} where {T<:_NativeScalar,M<:_BorrowMatrix}
const _TridiagonalSymmetric=LinearAlgebra.Symmetric{
    T,
    M,
} where {T<:Union{Float32,Float64},M<:_BorrowMatrix}
const _TridiagonalSelfAdjoint=Union{_TridiagonalHermitian,_TridiagonalSymmetric}
LinearAlgebra.hessenberg(a::_TridiagonalSelfAdjoint; kwargs...) =
    tridiagonalize(a; kwargs...)
function LinearAlgebra.hessenberg!(a::_TridiagonalSelfAdjoint; kwargs...)
    lock(_matrix_lock) do
        f=tridiagonalize(a; kwargs...)
        _operator_copy_result!(parent(a), Matrix(f.H))
        return f
    end
end
function _tridiagonal_evd(
    d::AbstractVector{<:Number},
    e::AbstractVector{<:Number},
    vectors;
    recursion_threshold::Integer = 128,
    sortby = nothing,
    scratch = nothing,
    team = nothing,
)
    lock(_matrix_lock) do
        _iterative_resource_check(scratch, team)
        _tridiagonal_open(d)
        _tridiagonal_open(e)
        _dense_rhs_validate(d)
        _dense_rhs_validate(e)
        n=length(d)
        length(e)==max(0, n-1) ||
            throw(DimensionMismatch("subdiagonal length must be max(0,length(diag)-1)"))
        T=_reflector_type(promote_type(eltype(d), eltype(e)))
        R=_realtype(T)
        threshold=Int(recursion_threshold)
        threshold>=2 || throw(ArgumentError("recursion_threshold must be at least two"))
        req=_tridiagonal_evd_requirement(n, T, vectors, threshold, team)
        work=isnothing(scratch) ? FaerScratch(req.bytes; alignment = req.alignment) :
             scratch
        try
            cap=scratch_capacity(work)
            cap.bytes>=req.bytes && cap.alignment>=req.alignment || throw(
                ArgumentError(
                    "tridiagonal EVD scratch capacity/alignment is insufficient; call reserve!",
                ),
            )
            diag=Vector{T}(d)
            off=Vector{T}(e)
            all(isfinite, diag)&&all(isfinite, off) ||
                throw(ArgumentError("tridiagonal coefficients must be finite"))
            all(isreal, diag) ||
                throw(ArgumentError("Hermitian tridiagonal diagonal must be real"))
            values=zeros(R, n)
            q=zeros(T, vectors ? n : 0, vectors ? n : 0)
            GC.@preserve diag off values q begin
                args=(
                    work,
                    Ptr{UInt8}(pointer(diag)),
                    Ptr{UInt8}(pointer(off)),
                    UInt(n),
                    Ptr{UInt8}(pointer(values)),
                    Ptr{UInt8}(pointer(q)),
                    _tag(T),
                    vectors,
                    UInt(threshold),
                )
                isnothing(team) ?
                _native_call(RustCrate.tridiagonal_evd_workspace, args...) :
                _native_call(RustCrate.tridiagonal_evd_workspace_team, args..., team)
            end
            all(isfinite, values)&&all(isfinite, q) || throw(
                ArgumentError("tridiagonal eigendecomposition overflowed the working type"),
            )
            if !isnothing(sortby)
                order=sortperm(values; by = sortby)
                values=values[order]
                vectors && (q=q[:, order])
            end
            return vectors ? LinearAlgebra.Eigen(values, FaerMatrix(q)) : values
        finally
            isnothing(scratch) && close(work)
        end
    end
end

"""
    tridiagonal_eigen(d, e; recursion_threshold=128, sortby=nothing, scratch=nothing, team=nothing)

Call faer's dedicated Hermitian tridiagonal eigensolver. d is real-valued;
e is the subdiagonal and the superdiagonal is conj(e). All four native scalar
types and promotion are supported. Returns `Eigen` with real ascending values
and native vectors; sortby overrides ordering. Inputs are unchanged. The native
QR or divide-and-conquer path is selected by recursion_threshold. Coefficients
use O(n) storage; eigenvectors require O(n²). Extreme inputs are scaled safely.
Use `scratch_requirements`, `FaerScratch(tridiagonal_eigen,d,e)` and `reserve!`
for reusable kernel workspace. The requirement includes the scaled band and
native temporary eigenvalues. A supplied team lends its actual execution token;
coefficient conversion, output and sorting arrays allocate separately. Saved
tridiagonal factors and native `SymTridiagonal` eigen/eigvals forward these
resources; a saved factor's separate Q back-transform also uses the team.
"""
tridiagonal_eigen(d::AbstractVector{<:Number}, e::AbstractVector{<:Number}; kwargs...) =
    _tridiagonal_evd(d, e, true; kwargs...)
"""Native tridiagonal eigenvalues only, using O(n) coefficient/work storage; see [`tridiagonal_eigen`](@ref)."""
tridiagonal_eigvals(d::AbstractVector{<:Number}, e::AbstractVector{<:Number}; kwargs...) =
    _tridiagonal_evd(d, e, false; kwargs...)
function LinearAlgebra.eigen(f::FaerTridiagonalization; kwargs...)
    lock(_matrix_lock) do
        _dense_rhs_validate(f.Q)
        size(f.Q)==size(f.H) || throw(DimensionMismatch("saved tridiagonal Q/H shapes do not match"))
        E=tridiagonal_eigen(f.H.dv, f.H.ev; kwargs...)
        team=get(kwargs, :team, nothing)
        vectors=isnothing(team) ? f.Q*E.vectors : team_mul(team, f.Q, E.vectors)
        return LinearAlgebra.Eigen(E.values, vectors)
    end
end
function LinearAlgebra.eigvals(f::FaerTridiagonalization; kwargs...)
    lock(_matrix_lock) do
        _dense_rhs_validate(f.Q)
        size(f.Q)==size(f.H) || throw(DimensionMismatch("saved tridiagonal Q/H shapes do not match"))
        return tridiagonal_eigvals(f.H.dv, f.H.ev; kwargs...)
    end
end
const _NativeSymTridiagonal=LinearAlgebra.SymTridiagonal{
    T,
    V,
} where {T<:Union{Float32,Float64},V<:FaerVector{T}}
LinearAlgebra.eigen(a::_NativeSymTridiagonal; kwargs...) =
    tridiagonal_eigen(a.dv, a.ev; kwargs...)
LinearAlgebra.eigvals(a::_NativeSymTridiagonal; kwargs...) =
    tridiagonal_eigvals(a.dv, a.ev; kwargs...)
