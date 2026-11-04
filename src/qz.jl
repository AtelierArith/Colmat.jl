"""
    hessenberg(A::FaerMatrix, B::FaerMatrix) -> FaerGeneralizedHessenberg

Generalized Hessenberg reduction of a square matrix pair. H is upper Hessenberg,
T is upper triangular, and explicit native orthogonal/unitary Q and Z satisfy
`A ≈ Q*H*Z'` and `B ≈ Q*T*Z'`. Inputs are copied. `schur(F)` reuses this reduction.
"""
struct FaerGeneralizedHessenberg{Ty,M<:FaerMatrix{Ty}} <: LinearAlgebra.Factorization{Ty}
    H::M
    T::M
    Q::M
    Z::M
end
const _NativeQZ=LinearAlgebra.GeneralizedSchur{T,<:FaerMatrix{T}} where {T<:_NativeScalar}
const _NativePairFactor=Union{FaerGeneralizedHessenberg,_NativeQZ}
_pair_first(f::FaerGeneralizedHessenberg) = f.H
_pair_first(f::_NativeQZ) = f.S
_pair_matrices(f::_NativePairFactor) = (_pair_first(f), f.T, f.Q, f.Z)
Base.size(f::_NativePairFactor) = size(_pair_first(f))
Base.size(f::_NativePairFactor, d::Integer) = size(_pair_first(f), d)
function _check_pair_factor(f::_NativePairFactor)
    n=LinearAlgebra.checksquare(_pair_first(f))
    for m in _pair_matrices(f)
        _handle(m)
        size(m)==(n, n) ||
            throw(DimensionMismatch("pair factors must have matching square dimensions"))
    end
    return n
end
function Base.isopen(f::_NativePairFactor)
    lock(_matrix_lock) do ;
        return all(isopen, _pair_matrices(f))
    end
end
function Base.close(f::_NativePairFactor)
    lock(_matrix_lock) do ;
        foreach(close, _pair_matrices(f))
    end
    return nothing
end
function Base.copy(f::FaerGeneralizedHessenberg)
    lock(_matrix_lock) do
        _check_pair_factor(f)
        return FaerGeneralizedHessenberg(map(copy, _pair_matrices(f))...)
    end
end
function Base.copy(f::_NativeQZ)
    lock(_matrix_lock) do
        _check_pair_factor(f)
        return LinearAlgebra.GeneralizedSchur(
            copy(f.S),
            copy(f.T),
            copy(f.α),
            copy(f.β),
            copy(f.Q),
            copy(f.Z),
        )
    end
end
function Base.deepcopy_internal(f::FaerGeneralizedHessenberg, d::IdDict)
    get!(d, f) do
        lock(_matrix_lock) do
            _check_pair_factor(f)
            return FaerGeneralizedHessenberg(
                map(m->Base.deepcopy_internal(m, d), _pair_matrices(f))...,
            )
        end
    end
end
function Base.deepcopy_internal(f::_NativeQZ, d::IdDict)
    get!(d, f) do
        lock(_matrix_lock) do
            _check_pair_factor(f)
            return LinearAlgebra.GeneralizedSchur(
                Base.deepcopy_internal(f.S, d),
                Base.deepcopy_internal(f.T, d),
                Base.deepcopy_internal(f.α, d),
                Base.deepcopy_internal(f.β, d),
                Base.deepcopy_internal(f.Q, d),
                Base.deepcopy_internal(f.Z, d),
            )
        end
    end
end
_qz_matrix(a::FaerMatrix) = a
_qz_matrix(a::FaerMatrixView) = copy(a)
function _native_qz(
    a::FaerMatrix{T},
    b::FaerMatrix{T};
    only_hessenberg = false,
    initial = nothing,
) where {T}
    lock(_matrix_lock) do
        n=LinearAlgebra.checksquare(a)
        size(b)==(n, n) ||
            throw(DimensionMismatch("QZ inputs must have matching square shapes"))
        t, q, z=(_zeros(T, (n, n)) for _ = 1:3)
        alpha=Vector{Complex{_realtype(T)}}(undef, only_hessenberg ? 0 : n)
        beta=Vector{T}(undef, only_hessenberg ? 0 : n)
        iq, iz=isnothing(initial) ? (Ptr{UInt8}(C_NULL), Ptr{UInt8}(C_NULL)) :
               (initial.Q, initial.Z)
        s=GC.@preserve alpha beta if T===Float64
            _matrix_result(
                RustCrate.qz_from,
                (n, n),
                a,
                b,
                t,
                q,
                z,
                Ptr{UInt8}(pointer(alpha)),
                Ptr{UInt8}(pointer(beta)),
                only_hessenberg,
                iq,
                iz,
                UInt(n),
                _tag(T),
            )
        else
            _typed_result(
                T,
                RustCrate.qz_from,
                (n, n),
                a,
                b,
                t,
                q,
                z,
                Ptr{UInt8}(pointer(alpha)),
                Ptr{UInt8}(pointer(beta)),
                only_hessenberg,
                iq,
                iz,
                UInt(n),
                _tag(T),
            )
        end
        return only_hessenberg ? FaerGeneralizedHessenberg(s, t, q, z) :
               LinearAlgebra.GeneralizedSchur(s, t, alpha, beta, q, z)
    end
end
function _promoted_qz(a, b; kwargs...)
    P=promote_type(eltype(a), eltype(b))
    return _native_qz(_coerce(_qz_matrix(a), P), _coerce(_qz_matrix(b), P); kwargs...)
end
"""
    schur(A::FaerMatrix, B::FaerMatrix)

Native generalized Schur (QZ) factorization of a matching square matrix pair.
Return `GeneralizedSchur` with native S/T/Q/Z and independent Julia α/β vectors.
`A ≈ F.Q*F.S*F.Z'`, `B ≈ F.Q*F.T*F.Z'`. Homogeneous α/β are retained, including
zero β (infinite eigenvalues) and α=β=0 (indeterminate eigenvalues). Mixed native
scalar types are promoted before the reduction.
"""
LinearAlgebra.schur(a::FaerMatrix, b::FaerMatrix) = _promoted_qz(a, b)
LinearAlgebra.schur(a::FaerMatrixView, b::_BorrowMatrix) = _promoted_qz(a, b)
LinearAlgebra.schur(a::FaerMatrix, b::FaerMatrixView) = _promoted_qz(a, b)
LinearAlgebra.schur(a::FaerMatrix, b::AbstractMatrix{<:Number}) =
    LinearAlgebra.schur(a, FaerMatrix{promote_type(eltype(a), eltype(b))}(b))
LinearAlgebra.schur(a::AbstractMatrix{<:Number}, b::FaerMatrix) =
    LinearAlgebra.schur(FaerMatrix{promote_type(eltype(a), eltype(b))}(a), b)
LinearAlgebra.hessenberg(a::FaerMatrix, b::FaerMatrix) =
    _promoted_qz(a, b; only_hessenberg = true)
LinearAlgebra.hessenberg(a::FaerMatrixView, b::_BorrowMatrix) =
    _promoted_qz(a, b; only_hessenberg = true)
LinearAlgebra.hessenberg(a::FaerMatrix, b::FaerMatrixView) =
    _promoted_qz(a, b; only_hessenberg = true)
function LinearAlgebra.schur(f::FaerGeneralizedHessenberg)
    lock(_matrix_lock) do
        _check_pair_factor(f)
        return _native_qz(f.H, f.T; initial = f)
    end
end
function _distinct_qz_storage(matrices)
    owners=map(m->_handle(m).ptr, matrices)
    any(owners[i]==owners[j] for i in eachindex(owners) for j = (i+1):length(owners)) &&
        throw(ArgumentError("mutating matrix-pair factors require distinct native storage"))
    return nothing
end
function _distinct_pair_storage(a, b)
    eltype(a)===eltype(b) ||
        throw(ArgumentError("mutating a matrix pair requires matching scalar types"))
    _distinct_qz_storage((a, b))
end
"""
    schur!(A::FaerMatrix, B::FaerMatrix)

Compute native QZ and overwrite A/B with S/T after successful completion. The
returned factor shares A/B, preserving their dimensions and existing views.
Independent work results and scratch are allocated. A/B must be distinct and
have the same scalar type.
"""
function LinearAlgebra.schur!(a::FaerMatrix, b::FaerMatrix)
    lock(_matrix_lock) do
        _distinct_pair_storage(a, b)
        f=LinearAlgebra.schur(a, b)
        copyto!(a, f.S)
        copyto!(b, f.T)
        return LinearAlgebra.GeneralizedSchur(a, b, f.α, f.β, f.Q, f.Z)
    end
end
"""
    hessenberg!(A::FaerMatrix, B::FaerMatrix)

Compute generalized Hessenberg form, then overwrite A with H and B with T.
The returned factor shares these matrices; Q/Z are independent native owners.
"""
function LinearAlgebra.hessenberg!(a::FaerMatrix, b::FaerMatrix)
    lock(_matrix_lock) do
        _distinct_pair_storage(a, b)
        f=LinearAlgebra.hessenberg(a, b)
        copyto!(a, f.H)
        copyto!(b, f.T)
        return FaerGeneralizedHessenberg(a, b, f.Q, f.Z)
    end
end
function LinearAlgebra.schur!(f::FaerGeneralizedHessenberg)
    lock(_matrix_lock) do
        matrices=_pair_matrices(f)
        _distinct_qz_storage(matrices)
        g=LinearAlgebra.schur(f)
        for (a, b) in zip(matrices, (g.S, g.T, g.Q, g.Z))
            copyto!(a, b)
        end
        return LinearAlgebra.GeneralizedSchur(f.H, f.T, g.α, g.β, f.Q, f.Z)
    end
end
function LinearAlgebra.eigvals(f::_NativeQZ)
    lock(_matrix_lock) do
        n=_check_pair_factor(f)
        length(f.α)==length(f.β)==n ||
            throw(DimensionMismatch("homogeneous eigenvalue lengths must match"))
        return f.α ./ f.β
    end
end
Base.iterate(f::FaerGeneralizedHessenberg) = (f.H, Val(:T))
Base.iterate(f::FaerGeneralizedHessenberg, ::Val{:T}) = (f.T, Val(:Q))
Base.iterate(f::FaerGeneralizedHessenberg, ::Val{:Q}) = (f.Q, Val(:Z))
Base.iterate(f::FaerGeneralizedHessenberg, ::Val{:Z}) = (f.Z, Val(:done))
Base.iterate(f::FaerGeneralizedHessenberg, ::Val{:done}) = nothing

# Intersection with the mixed Julia-array/owned-matrix method.
LinearAlgebra.schur(a::FaerMatrixView{<:Number}, b::FaerMatrix) = _promoted_qz(a, b)

function _ordqz(f::_NativeQZ, select)
    lock(_matrix_lock) do
        n=_check_pair_factor(f)
        length(select)==n ||
            throw(DimensionMismatch("QZ selection length must match the factor dimension"))
        data=UInt8.(select)
        T=eltype(f)
        t, q, z=(_zeros(T, (n, n)) for _ = 1:3)
        alpha=Vector{Complex{_realtype(T)}}(undef, n)
        beta=Vector{T}(undef, n)
        args=(
            f.S,
            f.T,
            f.Q,
            f.Z,
            t,
            q,
            z,
            pointer(data),
            UInt(n),
            Ptr{UInt8}(pointer(alpha)),
            Ptr{UInt8}(pointer(beta)),
            _tag(T),
        )
        s=GC.@preserve data alpha beta if T===Float64
            _matrix_result(RustCrate.qz_reorder_from, (n, n), args...)
        else
            _typed_result(T, RustCrate.qz_reorder_from, (n, n), args...)
        end
        return LinearAlgebra.GeneralizedSchur(s, t, alpha, beta, q, z)
    end
end
"""
    ordschur(F::GeneralizedSchur{T,<:FaerMatrix}, select)

Reorder saved native generalized Schur factors so selected eigenvalue blocks
appear first, preserving order within each group. Selecting either member of a
real conjugate pair selects both. Return independent S/T/Q/Z and homogeneous
α/β computed from the saved factors, including infinite eigenvalues. A rejected
native move throws while preserving F. Work factors and scratch are allocated.
"""
LinearAlgebra.ordschur(f::_NativeQZ, select::Union{Vector{Bool},BitVector}) =
    _ordqz(f, select)
# Compare actual element byte intervals: strided alpha/beta views can share a
# parent while remaining disjoint, including interleaved and reversed views.
function _qz_metadata_alias(a::StridedVector, b::StridedVector)
    (isempty(a)||isempty(b)) && return false
    aw, bw=Base.elsize(a), Base.elsize(b)
    (iszero(aw)||iszero(bw)) && return false
    sa, sb=stride(a, 1), stride(b, 1)
    GC.@preserve a b begin
        ap=UInt(pointer(a, sa<0 ? length(a) : 1))
        bp=UInt(pointer(b, sb<0 ? length(b) : 1))
        da, db=UInt(abs(sa)*aw), UInt(abs(sb)*bw)
        (ap+UInt(length(a)-1)*da+UInt(aw)<=bp || bp+UInt(length(b)-1)*db+UInt(bw)<=ap) &&
            return false
        i, j=1, 1
        while i<=length(a) && j<=length(b)
            if ap<bp+UInt(bw) && bp<ap+UInt(aw)
                return true
            elseif ap+UInt(aw)<=bp
                ap+=da
                i+=1
            else
                bp+=db
                j+=1
            end
        end
    end
    return false
end

"""
    ordschur!(F::GeneralizedSchur{T,<:FaerMatrix}, select)

Reorder saved native generalized Schur factors and update F after all moves and
metadata conversions succeed. Existing matrix owners and views are preserved.
S/T/Q/Z must have distinct storage; α/β must be disjoint mutable strided vectors
of the factor dimension. Work factors and scratch are allocated.
"""
function LinearAlgebra.ordschur!(f::_NativeQZ, select::Union{Vector{Bool},BitVector})
    lock(_matrix_lock) do
        n=_check_pair_factor(f)
        matrices=_pair_matrices(f)
        _distinct_qz_storage(matrices)
        f.α isa StridedVector && f.β isa StridedVector || throw(
            ArgumentError(
                "mutating QZ requires mutable strided homogeneous eigenvalue storage",
            ),
        )
        length(f.α)==n && length(f.β)==n || throw(
            DimensionMismatch(
                "QZ homogeneous eigenvalue lengths must match the factor dimension",
            ),
        )
        _qz_metadata_alias(f.α, f.β) &&
            throw(ArgumentError("QZ α/β must have disjoint storage"))
        result=_ordqz(f, select)
        alpha=Vector{eltype(f.α)}(result.α)
        beta=Vector{eltype(f.β)}(result.β)
        foreach(copyto!, matrices, _pair_matrices(result))
        copyto!(f.α, alpha)
        copyto!(f.β, beta)
    end
    return f
end
