# Whole-array broadcast, map and concatenation for native dense arrays.
#
# Every native dense leaf (FaerMatrix, FaerVector, FaerRow, FaerView and their
# Adjoint/Transpose wrappers) is snapshotted once into a Julia array in bulk,
# borrowed Julia arrays are unwrapped to their parents, and Julia's own
# broadcast/map/cat kernel runs on the result. Julia's shape, eltype and
# result-container rules therefore apply unchanged; a dense `Array` result
# whose scalar is native is then loaded into one native owner in one bulk
# transfer, and in-place forms compute into an independent Julia buffer before
# one bulk write. Snapshots are taken before anything is written, so every
# destination/source alias sees the old values.
"""
    FaerArrayStyle{N}()

Broadcast style of the native dense arrays [`FaerMatrix`](@ref),
[`FaerVector`](@ref), [`FaerRow`](@ref), native views and their adjoints and
transposes, and of [`FaerDiagonal`](@ref). It wins over Julia's dense,
structured and sparse styles; the native sparse style wins over it.

A `FaerDiagonal` operand is read as a `Diagonal` of its stored entries, so a
result that Julia keeps diagonal (`D .* 2`, `abs.(D)`, `D .* D`) is a
`FaerDiagonal` and other results (`D .+ 1`, `D .+ A`) are dense. `D .= ...`
computes into a `Diagonal` buffer (nonzero off-diagonal results throw before
anything is written) and writes the stored entries in one bulk transfer.

`f.(args...)` snapshots each native operand once in bulk, runs Julia's
broadcast on the snapshots (so shapes, dimension expansion and result eltypes
follow Julia exactly) and returns:

- a [`FaerVector`](@ref) or [`FaerMatrix`](@ref) when Julia would return a
  dense `Vector`/`Matrix` of `Float32`, `Float64`, `ComplexF32` or `ComplexF64`
  (a [`FaerRow`](@ref) when the result has one row and every native array
  operand is a `FaerRow`);
- otherwise Julia's own result, for example a `BitMatrix` for comparisons, an
  `Array{Int}`, a sparse matrix or an array with more than two dimensions.

`dest .= f.(args...)` (and `.+=` etc.) computes into a Julia buffer of
`eltype(dest)` first and then writes `dest` in one bulk transfer, so a failure
(including `InexactError`) leaves `dest` unchanged and aliased operands are
read before the write. [`FaerBorrowedArray`](@ref) operands are read in place
and keep their Julia broadcast style when no native array takes part.
"""
struct FaerArrayStyle{N} <: Broadcast.AbstractArrayStyle{N} end
FaerArrayStyle{M}(::Val{N}) where {M,N} = FaerArrayStyle{N}()

const _DenseNativeArray=Union{FaerMatrix,FaerVector,FaerRow,FaerView}
const _DenseNativeWrapped=Union{
    LinearAlgebra.Adjoint{<:Any,<:_DenseNativeArray},
    LinearAlgebra.Transpose{<:Any,<:_DenseNativeArray},
}
const _DenseNativeLeaf=Union{_DenseNativeArray,_DenseNativeWrapped}

Broadcast.BroadcastStyle(::Type{<:FaerVector}) = FaerArrayStyle{1}()
Broadcast.BroadcastStyle(::Type{<:Union{FaerMatrix,FaerRow}}) = FaerArrayStyle{2}()
Broadcast.BroadcastStyle(::Type{<:FaerView{<:Any,N}}) where {N} = FaerArrayStyle{N}()
Broadcast.BroadcastStyle(::Type{<:LinearAlgebra.Adjoint{<:Any,<:_DenseNativeArray}}) =
    FaerArrayStyle{2}()
Broadcast.BroadcastStyle(::Type{<:LinearAlgebra.Transpose{<:Any,<:_DenseNativeArray}}) =
    FaerArrayStyle{2}()
# A FaerDiagonal operand is read as a `Diagonal` of its stored entries, so
# Julia's structured rules decide whether the result stays diagonal.
Broadcast.BroadcastStyle(::Type{<:FaerDiagonal}) = FaerArrayStyle{2}()
for S in (
    LinearAlgebra.StructuredMatrixStyle,
    SparseArrays.HigherOrderFns.SparseMatStyle,
    SparseArrays.HigherOrderFns.SparseVecStyle,
    SparseArrays.HigherOrderFns.PromoteToSparse,
)
    @eval Broadcast.BroadcastStyle(::FaerArrayStyle{N}, ::$S) where {N} =
        FaerArrayStyle{max(N, 2)}()
end
Broadcast.BroadcastStyle(::FaerSparseStyle, ::FaerArrayStyle{N}) where {N} =
    N<=2 ? FaerSparseStyle() : Broadcast.DefaultArrayStyle{N}()

# One bulk native-to-Julia transfer per leaf.
_dense_snapshot(a::FaerMatrix) = Matrix(a)
_dense_snapshot(a::FaerVector) = Vector(a)
_dense_snapshot(a::FaerRow) = reshape(Vector(a), 1, a.len)
_dense_snapshot(a::FaerMatrixView) = Matrix(a)
_dense_snapshot(a::FaerVectorView) = Vector(a)
_dense_snapshot(a::LinearAlgebra.Adjoint) = adjoint(_dense_snapshot(parent(a)))
_dense_snapshot(a::LinearAlgebra.Transpose) = transpose(_dense_snapshot(parent(a)))

# Replaces native leaves (deduplicated by identity) with Julia snapshots and
# borrowed arrays with their parents. `rows[]` stays true while every native
# array leaf is a FaerRow; `sparse[]` records a native sparse leaf (only
# concatenation and `map` see those, sparse broadcast has its own style).
mutable struct _DenseSnapshots
    cache::IdDict{Any,Any}
    rows::Bool
    native::Bool
    sparse::Bool
end
_DenseSnapshots() = _DenseSnapshots(IdDict{Any,Any}(), true, false, false)
function _julia_leaf(s::_DenseSnapshots, a::_DenseNativeLeaf)
    s.native=true
    a isa FaerRow || (s.rows=false)
    return get!(() -> _dense_snapshot(a), s.cache, a)
end
function _julia_leaf(s::_DenseSnapshots, a::FaerDiagonal)
    s.native=true
    s.rows=false
    return get!(() -> LinearAlgebra.Diagonal(Vector(a)), s.cache, a)
end
function _julia_leaf(s::_DenseSnapshots, a::FaerSparseMatrix)
    s.native=true
    s.sparse=true
    s.rows=false
    return get!(() -> SparseArrays.SparseMatrixCSC(a), s.cache, a)
end
_julia_leaf(::_DenseSnapshots, a::FaerBorrowedArray) = parent(a)
_julia_leaf(::_DenseSnapshots, a) = a
_julia_leaf(s::_DenseSnapshots, bc::Broadcast.Broadcasted) =
    Broadcast.broadcasted(bc.f, map(a -> _julia_leaf(s, a), bc.args)...)
function _julia_leaves(s::_DenseSnapshots, args)
    lock(_matrix_lock) do
        map(a -> _julia_leaf(s, a), args)
    end
end
_sparse_dense_leaf(a) = lock(() -> _dense_snapshot(a), _matrix_lock)
_julia_sparse_broadcasted(a::Union{FaerVector,FaerRow,FaerView,_DenseNativeWrapped}) =
    _sparse_dense_leaf(a)
_julia_sparse_broadcasted(a::FaerDiagonal) =
    lock(() -> LinearAlgebra.Diagonal(Vector(a)), _matrix_lock)

# Julia-to-native result transfers. Broadcast results (`A ./ 0`, `log.(A)`)
# may hold Inf/NaN, which owners store like Julia arrays; results are written
# into a fresh owner with one bulk view assignment that reads the Julia buffer
# in place.
_native_owner_from(owner, data) = (copyto!(owner, data); owner)
# `Array{Union{}}` (empty results of inferred-bottom eltypes) stays Julia's.
_native_array_result(r, ::Bool) = r
_native_array_result(r::Vector{T}, ::Bool) where {T<:_NativeScalar} =
    T===Union{} ? r : _native_owner_from(FaerVector{T}(undef, length(r)), r)
function _native_array_result(r::Matrix{T}, rows::Bool) where {T<:_NativeScalar}
    T===Union{} && return r
    rows && size(r, 1)==1 && return _native_owner_from(FaerRow{T}(undef, size(r, 2)), r)
    return _native_owner_from(FaerMatrix{T}(undef, size(r)...), r)
end
_native_array_result(r::LinearAlgebra.Diagonal{T,Vector{T}}, ::Bool) where {T<:_NativeScalar} =
    FaerDiagonal{T}(r.diag)
function _native_dense_result(s::_DenseSnapshots, r)
    s.native || return r
    return _native_array_result(r, s.rows)
end
# A sparse result is native only when a native sparse operand took part, as
# SparseArrays makes any concatenation with a sparse operand sparse.
_native_dense_result(s::_DenseSnapshots, r::SparseArrays.SparseMatrixCSC{<:_NativeScalar,Int}) =
    s.sparse ? FaerSparseMatrix(r) : r

# `A .± B` (equal shapes and scalar types) and `α .* A` reuse the native
# whole-matrix kernels, which avoid the snapshots entirely.
function _dense_native_kernel(bc::Broadcast.Broadcasted)
    length(bc.args)==2 || return nothing
    a, b=bc.args
    if (bc.f===(+) || bc.f===(-)) && a isa FaerMatrix && b isa FaerMatrix
        eltype(a)===eltype(b) && size(a)==size(b) && return bc.f(a, b)
    elseif bc.f===(*) && _native_scale(a, b)
        return a*b
    elseif bc.f===(*) && _native_scale(b, a)
        return b*a
    end
    return nothing
end
# Only where the native scaling rounds exactly like Julia's scalar `*`: real
# scalars of real matrices, or complex scalars of the matrix's own type
# (mixed real/complex products differ for infinite parts). Bool is excluded
# because `false*NaN` is zero in Julia.
_native_scale(α, a) = false
# The scalar must also not widen the matrix, keeping the fast path to the
# kernels that need no promotion.
_native_scale(α::Union{Float32,Float64,Base.BitInteger}, ::FaerMatrix{T}) where {T<:Union{Float32,Float64}} =
    promote_type(typeof(α), T)===T
_native_scale(α::T, a::FaerMatrix{T}) where {T<:Union{ComplexF32,ComplexF64}} = true
function Base.copy(bc::Broadcast.Broadcasted{FaerArrayStyle{N}}) where {N}
    fast=_dense_native_kernel(bc)
    isnothing(fast) || return fast
    s=_DenseSnapshots()
    args=_julia_leaves(s, bc.args)
    r=Broadcast.materialize(Broadcast.broadcasted(bc.f, args...))
    return _native_dense_result(s, r)
end

const _DenseNativeDest=Union{FaerMatrix,FaerVector,FaerRow,FaerView}
# Sources that the bulk `copyto!` writes without restaging them through a
# temporary native owner.
_direct_copy_source(dest, a) = false
_direct_copy_source(::FaerMatrix{T}, ::FaerMatrix{T}) where {T} = true
_direct_copy_source(dest, a::Union{Array,FaerBorrowedArray}) =
    eltype(a)===eltype(dest) && axes(a)==axes(dest)
function Base.copyto!(dest::_DenseNativeDest, bc::Broadcast.Broadcasted{FaerArrayStyle{N}}) where {N}
    if bc.f===identity && length(bc.args)==1
        x=only(bc.args)
        # `dest .= x` and same-type `dest .= B` keep the bulk fill/copy paths,
        # which already snapshot aliased sources and convert before writing.
        x isa Number && return fill!(dest, x)
        _direct_copy_source(dest, x) && return copyto!(dest, x)
    end
    # `A .+= B`, `A .*= 2`: a fresh native result, then one native assignment.
    if dest isa FaerMatrix
        fast=_dense_native_kernel(bc)
        !isnothing(fast) && eltype(fast)===eltype(dest) && size(fast)==size(dest) &&
            return copyto!(dest, fast)
    end
    s=_DenseSnapshots()
    args=_julia_leaves(s, bc.args)
    buf=Array{eltype(dest)}(undef, size(dest))
    Broadcast.materialize!(buf, Broadcast.broadcasted(bc.f, args...))
    return copyto!(dest, buf)
end
# `D .= D .* 2`: a Julia `Diagonal` buffer (Julia rejects nonzero off-diagonal
# results before anything is written), then one bulk write of the entries.
function Base.copyto!(dest::FaerDiagonal, bc::Broadcast.Broadcasted{FaerArrayStyle{N}}) where {N}
    s=_DenseSnapshots()
    args=_julia_leaves(s, bc.args)
    buf=LinearAlgebra.Diagonal(Vector{eltype(dest)}(undef, size(dest, 1)))
    Broadcast.materialize!(buf, Broadcast.broadcasted(bc.f, args...))
    copyto!(diagonalview(dest), buf.diag)
    return dest
end
function Base.copyto!(dest::AbstractArray, bc::Broadcast.Broadcasted{FaerArrayStyle{N}}) where {N}
    s=_DenseSnapshots()
    args=_julia_leaves(s, bc.args)
    # Unwrapping a borrowed destination lets Julia's unaliasing see its memory.
    target=dest isa FaerBorrowedArray ? parent(dest) : dest
    Broadcast.materialize!(target, Broadcast.broadcasted(bc.f, args...))
    return dest
end

"""
    map(f, A, Bs...)
    map!(f, dest, A, Bs...)

When `A` is a native dense array (`FaerMatrix`, `FaerVector`, `FaerRow` or a
native view), every native argument is snapshotted once in bulk and Julia's
`map` runs on the snapshots, so argument shape checks are Julia's. Dense
native-scalar results are returned as native owners as for broadcast (see
[`FaerArrayStyle`](@ref)). `map!` into a native destination computes into a
Julia buffer and writes it in one bulk transfer, leaving the destination
unchanged on failure. When the first native argument follows Julia arrays
(`map(f, J, A)`), or the destination is a Julia array (`map!(f, J, A)`), the
same transfer applies (the leading Julia arrays accepted by concatenation,
up to three of them before the native argument).
"""
function Base.map(f, a::_DenseNativeArray, rest::AbstractArray...)
    s=_DenseSnapshots()
    args=_julia_leaves(s, (a, rest...))
    return _native_dense_result(s, map(f, args...))
end
function Base.map!(f, dest::_DenseNativeDest, a::AbstractArray, rest::AbstractArray...)
    s=_DenseSnapshots()
    args=_julia_leaves(s, (a, rest...))
    buf=Array{eltype(dest)}(undef, size(dest))
    map!(f, buf, args...)
    return copyto!(dest, buf)
end
