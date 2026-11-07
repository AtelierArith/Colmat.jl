# Native integer-vector and logical indexing, split views and strides.
#
# `A[I, J]` and `A[I, J] = B` with integer vectors, ranges, `:` or Bool masks
# on native dense owners and views run as one native gather/scatter
# (`views::gather`) over the descriptor of `A`. Results use Base's container
# rule (`similar(A, T, shape)`), so matrix results are native owners and
# vector results of matrices stay Julia vectors, exactly as before.
# Integer-vector views stay Julia `SubArray`s over the native array; their
# bulk copies and assignments use the same native gather/scatter.

const _NativeIndexed2=Union{FaerMatrix,FaerRow,FaerMatrixView}
const _NativeIndexed1=Union{FaerVector,FaerVectorView}
const _NativeIndexed=Union{_NativeIndexed2,_NativeIndexed1}
const _ArrayIndex=Union{Colon,AbstractVector}

# Index forms the native kernels take after `to_indices` (masks become
# `LogicalIndex`, `:` a `Slice`); others keep Base's generic path.
_gatherable(i::Integer) = true
_gatherable(i::AbstractVector) = eltype(i)<:Integer
_gatherable(::Any) = false
_zero_based(i::Integer) = UInt[i-1]
function _zero_based(I::AbstractVector)
    out=Vector{UInt}(undef, length(I))
    for (k, i) in enumerate(I)
        out[k]=i-1
    end
    return out
end
# Row and column lists of a view-relative selection; vectors are columns.
_index_lists(::_NativeIndexed1, idx) = (_zero_based(idx[1]), UInt[0])
_index_lists(::_NativeIndexed2, idx) = (_zero_based(idx[1]), _zero_based(idx[2]))

# A 1-d native/Julia descriptor is a column; a selection with one row and
# several columns needs it transposed.
function _oriented(d, target, nr, nc)
    ndims(target)==1 && nr==1 && nc!=1 && (d[8]=1-d[8])
    return d
end

function _gather_into!(dest, a, rows, cols)
    target=_gemm_destination(dest)
    staged=isnothing(target)
    staged && (target=FaerMatrix{eltype(a)}(undef, length(rows), length(cols)))
    p, d=_view_source(target)
    _oriented(d, target, length(rows), length(cols))
    q, e=_view_source(a)
    GC.@preserve d e rows cols _native_call(
        RustCrate.view_gather_into,
        p,
        pointer(d),
        q,
        pointer(e),
        rows,
        cols,
    )
    staged && copyto!(dest, reshape(Matrix(target), size(dest)))
    return dest
end

function _native_getindex(a, I...)
    idx=to_indices(a, I)
    all(_gatherable, idx) ||
        return invoke(getindex, Tuple{AbstractArray,Vararg{Any}}, a, I...)
    lock(_matrix_lock) do
        _view_source(a)  # Throws for closed or invalidated arrays.
        checkbounds(a, idx...)
        rows, cols=_index_lists(a, idx)
        dest=similar(a, eltype(a), Base.index_shape(idx...))
        return _gather_into!(dest, a, rows, cols)
    end
end
Base.getindex(a::_NativeIndexed2, I::_ArrayIndex, J::_ArrayIndex) = _native_getindex(a, I, J)
Base.getindex(a::_NativeIndexed2, i::Integer, J::_ArrayIndex) = _native_getindex(a, i, J)
Base.getindex(a::_NativeIndexed2, I::_ArrayIndex, j::Integer) = _native_getindex(a, I, j)
Base.getindex(a::_NativeIndexed1, I::_ArrayIndex) = _native_getindex(a, I)
# Linear indexing of matrices (`A[mask]`, `A[[1, 5]]`, `A[:]`): one bulk read,
# Julia's indexing, then one bulk write into Base's result container.
function Base.getindex(a::_NativeIndexed2, I::Union{Colon,AbstractArray})
    x=Array(a)[I]
    dest=similar(a, eltype(a), size(x))
    return dest isa Array ? x : copyto!(dest, x)
end

# Source of a scatter as an `nr x nc` descriptor of `T`: native and Julia
# strided arrays of `T` are read in place (vectors transposed as needed);
# anything else is converted into a Julia buffer first, so conversion errors
# leave the destination untouched.
function _scatter_source(x, ::Type{T}, nr, nc) where {T}
    if eltype(x)===T
        y=_gemm_input(x)
        if !isnothing(y)
            q, e=_view_source(y)
            m, n=e[8]==0 ? (e[4], e[7]) : (e[7], e[4])
            (m, n)==(nr, nc) && return y, q, e
            if (n, m)==(nr, nc) && min(m, n)==1
                e[8]=1-e[8]
                return y, q, e
            end
        end
    end
    staged=copyto!(Matrix{T}(undef, nr, nc), x)
    y=FaerBorrowedArray(staged)
    q, e=_view_source(y)
    return y, q, e
end

function _native_setindex!(a, x::AbstractArray, I...)
    lock(_matrix_lock) do
        p, d=_view_source(a)
        idx=to_indices(a, I)
        if !(all(_gatherable, idx) && _injective_descriptor(d) && d[9]==0)
            invoke(setindex!, Tuple{AbstractArray,Any,Vararg{Any}}, a, x, I...)
            return a
        end
        checkbounds(a, idx...)
        Base.setindex_shape_check(x, map(length, idx)...)
        rows, cols=_index_lists(a, idx)
        y, q, e=_scatter_source(x, eltype(a), length(rows), length(cols))
        GC.@preserve d e rows cols y _native_call(
            RustCrate.view_scatter_into,
            p,
            pointer(d),
            rows,
            cols,
            q,
            pointer(e),
        )
        return a
    end
end
Base.setindex!(a::_NativeIndexed2, x::AbstractArray, I::_ArrayIndex, J::_ArrayIndex) =
    _native_setindex!(a, x, I, J)
Base.setindex!(a::_NativeIndexed2, x::AbstractArray, i::Integer, J::_ArrayIndex) =
    _native_setindex!(a, x, i, J)
Base.setindex!(a::_NativeIndexed2, x::AbstractArray, I::_ArrayIndex, j::Integer) =
    _native_setindex!(a, x, I, j)
Base.setindex!(a::_NativeIndexed1, x::AbstractArray, I::_ArrayIndex) =
    _native_setindex!(a, x, I)

# `view(A, I, J)` with integer vectors or masks is a Julia SubArray over the
# native array (`copy` already uses `getindex`). Bulk conversions gather into
# Julia memory, and bulk writes (`copyto!`, `S .= B`, `fill!`, `S .= x`)
# scatter natively.
const _NativeSubArray{T,N}=SubArray{T,N,<:_NativeIndexed}
function Base.Array{T,N}(s::_NativeSubArray{T,N}) where {T,N}
    idx=parentindices(s)
    a=parent(s)
    all(_gatherable, idx) || return copyto!(Array{T,N}(undef, size(s)), s)
    lock(_matrix_lock) do
        _view_source(a)
        rows, cols=_index_lists(a, idx)
        return _gather_into!(Array{T,N}(undef, size(s)), a, rows, cols)
    end
end
# Dense sources only: sparse and CHOLMOD sources keep their own methods.
const _ScatterSource=Union{Array,SubArray,FaerMatrix,FaerVector,FaerRow,FaerView,FaerBorrowedArray}
function Base.copyto!(s::_NativeSubArray, src::_ScatterSource)
    length(src)==length(s) ||
        return invoke(copyto!, Tuple{AbstractArray,AbstractArray}, s, src)
    x=size(src)==size(s) ? src : reshape(src, size(s))
    _native_setindex!(parent(s), x, parentindices(s)...)
    return s
end
Base.fill!(s::_NativeSubArray{T}, x) where {T} = copyto!(s, fill(convert(T, x), size(s)))

"""
    split_at_row(A, k) -> (top, bottom)
    split_at_col(A, k) -> (left, right)
    split_at(A, i, j) -> (top_left, top_right, bottom_left, bottom_right)

Split a native matrix, `FaerRow` or matrix view into borrowed native views,
mirroring faer's `split_at_row`, `split_at_col` and `split_at`. `k` (or
`i`/`j`) is the size of the first part, `0 <= k <= size(A, d)`: `top` is
`view(A, 1:k, :)` and `bottom` is `view(A, k+1:end, :)`. A vector
(`FaerVector` or a vector view, including a row view) splits along its only
axis with `split_at_row(v, k)`. The parts share `A`'s storage owner, so writes
through them update `A`; like other views they are invalidated by shape
changes of the owner. An out-of-range `k` throws `BoundsError`.
"""
function split_at_row end
@doc (@doc split_at_row) split_at_col
@doc (@doc split_at_row) split_at

function _split_point(a, d, k)
    0<=k<=size(a, d) || throw(BoundsError(a, d==1 ? (k, :) : (:, k)))
    return Int(k)
end
function split_at_row(a::_NativeIndexed2, k::Integer)
    k=_split_point(a, 1, k)
    return view(a, 1:k, :), view(a, (k+1):size(a, 1), :)
end
function split_at_col(a::_NativeIndexed2, k::Integer)
    k=_split_point(a, 2, k)
    return view(a, :, 1:k), view(a, :, (k+1):size(a, 2))
end
function split_at(a::_NativeIndexed2, i::Integer, j::Integer)
    i=_split_point(a, 1, i)
    j=_split_point(a, 2, j)
    m, n=size(a)
    return view(a, 1:i, 1:j), view(a, 1:i, (j+1):n), view(a, (i+1):m, 1:j),
    view(a, (i+1):m, (j+1):n)
end
function split_at_row(v::_NativeIndexed1, k::Integer)
    k=_split_point(v, 1, k)
    return view(v, 1:k), view(v, (k+1):length(v))
end

# Effective element strides reported by faer for the live storage: owners are
# column-major with a padded column stride (`stride(A, 2) >= size(A, 1)`, see
# `capacity`); views multiply by their steps. There is deliberately no
# `pointer`/`elsize`: native storage is not addressable from Julia code.
function _native_strides(a)
    lock(_matrix_lock) do
        p, d=_view_source(a)
        out=zeros(Int, 2)
        GC.@preserve d _native_call(RustCrate.view_strides, p, pointer(d), out)
        return ndims(a)==1 ? (out[1],) : (out[1], out[2])
    end
end
Base.strides(a::Union{FaerMatrix,FaerVector,FaerRow,FaerView}) = _native_strides(a)
