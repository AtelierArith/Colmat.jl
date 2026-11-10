"""
    loadnpy(source; view=false)
    npyview(source)

Load NPY bytes, an IO or a path through faer's Npy API. Float32/Float64 and
ComplexF32/ComplexF64, C/Fortran order and little/big endian are supported.
Scalar/vector shapes map to 1×1 / n×1 matrices; ranks above two are rejected.
Header, dtype, dimensions and exact payload length are validated before native
access. Values, including NaN/Inf, are preserved.

Default output is an independent FaerMatrix copied directly from the source
bytes by faer's `Npy::to_mat`, which reads unaligned and foreign-endian payloads
without an intermediate normalized buffer (contiguous byte vectors are read in
place). view=true returns a read-only FaerNpyMatrix that owns aligned
native-endian storage and uses as_aligned_ref, which requires such storage. The
view's bytes are snapshotted; this is not a memory mapping or an alias of Julia
input bytes (use [`npyborrow`](@ref) for those). The normalized buffer is retained until close/GC. Copying a view
materializes an independent native matrix; deepcopy retains the read-only type
and duplicates its aligned storage. See [`npyinfo`](@ref) for metadata.
"""
function loadnpy end
"""Read-only, owned, aligned NPY matrix view. Construct using [`npyview`](@ref)."""
mutable struct FaerNpyMatrix{T,H} <: AbstractMatrix{T}
    handle::Union{Nothing,H}
    const dims::Tuple{Int,Int}
    const shape::Tuple{Vararg{Int}}
    const fortran_order::Bool
    const version::Tuple{Int,Int}
    const byteorder::Symbol
    const aligned::Bool
end
function _handle(a::FaerNpyMatrix)
    h = a.handle
    isnothing(h) && throw(ArgumentError("NPY view has been closed"))
    return h
end
_native_argument(a::FaerNpyMatrix) = _handle(a)
_is_native_owner(::FaerNpyMatrix) = true
Base.size(a::FaerNpyMatrix) = a.dims
function Base.size(a::FaerNpyMatrix, d::Integer)
    if d < 1
        throw(ArgumentError("dimension must be positive"))
    elseif d <= 2
        a.dims[d]
    else
        1
    end
end
Base.isopen(a::FaerNpyMatrix) = lock(() -> !isnothing(a.handle), _matrix_lock)
function Base.close(a::FaerNpyMatrix)
    lock(_matrix_lock) do ;
        return a.handle = nothing
    end
    return nothing
end
function _npy_view(input::AbstractVector{UInt8})
    data = Vector{UInt8}(input)
    lock(_matrix_lock) do
        h = RustCrate.OwnedNpy()
        info = zeros(UInt, 9)
        try
            GC.@preserve data info begin
                _native_call(RustCrate.npy_load, h, pointer(data), UInt(length(data)))
                _native_call(RustCrate.npy_info, h, info)
            end
        catch err
            err isa RustCall.RustError && throw(ArgumentError(sprint(showerror, err)))
            rethrow()
        end
        T = (Float32, Float64, ComplexF32, ComplexF64)[Int(info[1]) + 1]
        dims = (Int(info[2]), Int(info[3]))
        rank = Int(info[4])
        shape = if rank == 0
            ()
        elseif rank == 1
            (dims[1],)
        else
            dims
        end
        return FaerNpyMatrix{T,typeof(h)}(
            h,
            dims,
            shape,
            info[5] != 0,
            (Int(info[6]), Int(info[7])),
            (:irrelevant, :little, :big)[Int(info[8]) + 1],
            info[9] != 0,
        )
    end
end
# Contiguous byte sources are passed in place; faer's to_mat reads unaligned
# and foreign-endian payloads directly (tensor4all/faer#215).
_npy_source(data::DenseVector{UInt8}) = data
_npy_source(data::Base.FastContiguousSubArray{UInt8,1,<:DenseVector{UInt8}}) = data
_npy_source(data::AbstractVector{UInt8}) = Vector{UInt8}(data)
function _npy_matrix(input::AbstractVector{UInt8})
    data = _npy_source(input)
    lock(_matrix_lock) do
        info = zeros(UInt, 8)
        try
            GC.@preserve data info begin
                p = pointer(data)
                len = UInt(length(data))
                _native_call(RustCrate.npy_source_info, p, len, info)
                T = (Float32, Float64, ComplexF32, ComplexF64)[Int(info[1]) + 1]
                m, n = Int(info[2]), Int(info[3])
                return if T === Float64
                    _matrix_result(
                        RustCrate.npy_load_direct, (m, n), p, len, UInt(m), UInt(n)
                    )
                else
                    _typed_result(
                        T,
                        RustCrate.npy_load_direct,
                        (m, n),
                        p,
                        len,
                        _tag(T),
                        UInt(m),
                        UInt(n),
                    )
                end
            end
        catch err
            err isa RustCall.RustError && throw(ArgumentError(sprint(showerror, err)))
            rethrow()
        end
    end
end
function loadnpy(data::AbstractVector{UInt8}; view::Bool=false)
    return view ? _npy_view(data) : _npy_matrix(data)
end
loadnpy(path::AbstractString; kwargs...) = loadnpy(read(path); kwargs...)
loadnpy(io::IO; kwargs...) = loadnpy(read(io); kwargs...)
"""Read an NPY source into an owned read-only aligned matrix view; see [`loadnpy`](@ref)."""
npyview(source) = loadnpy(source; view=true)
"""
    npyinfo(source)

Return dtype, original shape/order/version/byteorder, matrix_size and alignment
metadata. Sources can be an existing open FaerNpyMatrix, an [`npyborrow`](@ref)
borrow (metadata recorded when it was borrowed, plus `writable`; the source is
not reread), NPY bytes, an IO or a path. Raw bytes are normalized to aligned native byteorder for safe faer views;
byteorder reports the source's order, normalized_byteorder reports storage.
IO inputs are read from their current position to EOF.
"""
function npyinfo(a::FaerNpyMatrix{T}) where {T}
    lock(_matrix_lock) do
        _handle(a)
        return (
            dtype=T,
            shape=a.shape,
            matrix_size=size(a),
            fortran_order=a.fortran_order,
            version=a.version,
            byteorder=a.byteorder,
            aligned=a.aligned,
            normalized_byteorder=Base.ENDIAN_BOM == 0x04030201 ? :little : :big,
        )
    end
end
npyinfo(source) = npyinfo(npyview(source))
function Base.getindex(a::FaerNpyMatrix{T}, i::Int, j::Int) where {T}
    lock(_matrix_lock) do
        _handle(a)
        checkbounds(a, i, j)
        out = Ref{T}()
        GC.@preserve out _native_call(
            RustCrate.npy_element,
            a,
            UInt(i - 1),
            UInt(j - 1),
            Ptr{UInt8}(Base.unsafe_convert(Ptr{T}, out)),
            _tag(T),
            UInt(size(a, 1)),
            UInt(size(a, 2)),
        )
        return out[]
    end
end
function Base.setindex!(::FaerNpyMatrix, args...)
    return throw(ArgumentError("NPY views are read-only; copy to FaerMatrix first"))
end
function Base.Matrix(a::FaerNpyMatrix{T}) where {T}
    lock(_matrix_lock) do
        _handle(a)
        out = Matrix{T}(undef, size(a))
        GC.@preserve out _native_call(
            RustCrate.npy_export,
            a,
            Ptr{UInt8}(pointer(out)),
            _tag(T),
            UInt(size(a, 1)),
            UInt(size(a, 2)),
        )
        return out
    end
end
Base.Matrix{T}(a::FaerNpyMatrix) where {T} = Matrix{T}(Matrix(a))
Base.Array(a::FaerNpyMatrix) = Matrix(a)
function FaerMatrix(a::FaerNpyMatrix{T}) where {T}
    lock(_matrix_lock) do
        _handle(a)
        return if T === Float64
            _matrix_result(
                RustCrate.npy_copy_from, size(a), a, UInt(size(a, 1)), UInt(size(a, 2))
            )
        else
            _typed_result(
                T,
                RustCrate.npy_copy_from,
                size(a),
                a,
                _tag(T),
                UInt(size(a, 1)),
                UInt(size(a, 2)),
            )
        end
    end
end
FaerMatrix{T}(a::FaerNpyMatrix{T}) where {T<:_NativeScalar} = FaerMatrix(a)
Base.copy(a::FaerNpyMatrix) = FaerMatrix(a)
function Base.deepcopy_internal(a::FaerNpyMatrix{T}, d::IdDict) where {T}
    get!(d, a) do
        lock(_matrix_lock) do
            h = _native_call(
                RustCrate.npy_duplicate, a, _tag(T), UInt(size(a, 1)), UInt(size(a, 2))
            )
            return FaerNpyMatrix{T,typeof(h)}(
                h, a.dims, a.shape, a.fortran_order, a.version, a.byteorder, a.aligned
            )
        end
    end
end

"""
    npyborrow(bytes::Vector{UInt8}; writable=false, fallback=true)
    npyborrow(path::AbstractString; writable=false, fallback=true)
    npyborrow(io::IOStream; writable=false, fallback=true)

Borrow an NPY payload without copying it. A byte vector is aliased in place; a
path or an open `IOStream` (whose NPY data starts at its current position and
runs to the end of the file) is memory-mapped with `MAP_SHARED`, so writes by
other handles are visible. The result is a [`FaerBorrowedMatrix`](@ref) whose
parent addresses the payload directly, C-order arrays through row strides
`(n, 1)`; native consumers (`copy`, `FaerMatrix`, `+`/`-`/`*`, `mean`/`var`,
`sum`, ...) read it in place like [`faerview`](@ref) borrows. Scalar/vector
shapes map to 1×1 / n×1 as in [`loadnpy`](@ref), and the header, dtype, shape
and exact payload length are validated first. [`npyinfo`](@ref) of the borrow
returns its metadata (plus `writable`) without rereading the source.

Aliasing requires a native-endian payload whose data offset is a multiple of the
element alignment and, for bytes, a `Vector{UInt8}` source. Otherwise, with
`fallback=true`, the independent `FaerMatrix` from `loadnpy` is returned (big
endian on a little-endian host, odd header padding, `SubArray` byte sources);
`fallback=false` throws `ArgumentError` instead, and so does `writable=true`
(a copy would silently not write back).

By default the borrow is read-only: `setindex!`, broadcasts, `copyto!`,
`fill!` and in-place products/solves into it throw `ArgumentError`, and native
routines never write it. `writable=true` maps the file read-write (a path is
opened `"r+"`; a stream must be writable) or allows writes into the byte
vector: `setindex!`, `.=`, `copyto!`, `fill!`, `mul!` and `ldiv!`/`rdiv!`
then write the payload in place (dense `mul!`, `copyto!` and triangular solves
natively through a faer `MatMut`). `Mmap.sync!(B)` flushes a mapped borrow to
the file; unmapping by the finalizer also leaves the written pages in the file.

All borrows of one file (same device and inode, so hard links included) in a
process share a single mapping, read-write once any of them is writable. The
byte-range overlap checks of native destinations and Julia's `mightalias`
therefore see two borrows of one file as aliased memory, and inputs overlapping
the destination are snapshot; separate mappings would alias at different
virtual addresses that no range check can relate. A read-only borrow keeps
rejecting writes when it shares a read-write mapping. Copies of a file have
their own inode and do not alias. When the file size has changed since it was
mapped, the next borrow remaps it and every borrow of the file moves to the new
mapping; borrows whose payload no longer fits then throw `ArgumentError` on
access. Writes by other processes, or through `Mmap.mmap` arrays created
outside `npyborrow`, are visible but neither tracked by the overlap checks nor
ordered with native calls.

The borrow keeps the byte vector or mapping alive; a mapping is unmapped by its
finalizer once every borrow of the file and its views are unreachable (closing
or unlinking the file does not invalidate it). A source vector shrunk by
`resize!` is detected before every access, but truncating a mapped file while
it is accessed raises `SIGBUS` as for any `mmap`.
"""
function npyborrow end

# The bytes behind NPY borrows: a borrowed byte vector, or the process-wide
# mapping of one file (keyed by device and inode) shared by all its borrows.
# A remap replaces `bytes`; retired mappings stay referenced so a pointer read
# by an in-flight call stays valid until no borrow of the file remains.
mutable struct _NpyMapping
    bytes::Vector{UInt8}
    writable::Bool
    const key::Union{Nothing,Tuple{UInt64,UInt64}}
    const retired::Vector{Vector{UInt8}}
end
function _NpyMapping(bytes::Vector{UInt8}, writable::Bool)
    return _NpyMapping(bytes, writable, nothing, Vector{UInt8}[])
end
# Guarded by `_matrix_lock`.
const _npy_mappings = Dict{Tuple{UInt64,UInt64},WeakRef}()

# The aliased payload of a validated NPY source: element (i, j) is at byte
# `offset + ((i-1)*s1 + (j-1)*s2)*sizeof(T)` of `source.bytes`.
struct _NpyBuffer{T} <: AbstractMatrix{T}
    source::_NpyMapping
    offset::Int
    dims::Tuple{Int,Int}
    fortran::Bool
    writable::Bool
    shape::Tuple{Vararg{Int}}
    version::Tuple{Int,Int}
    byteorder::Symbol
end
Base.size(a::_NpyBuffer) = a.dims
Base.parent(a::_NpyBuffer) = a.source.bytes
Base.strides(a::_NpyBuffer) = a.fortran ? (1, a.dims[1]) : (a.dims[2], 1)
Base.stride(a::_NpyBuffer, d::Integer) = d <= 2 ? strides(a)[d] : length(a)
Base.elsize(::Type{<:_NpyBuffer{T}}) where {T} = sizeof(T)
Base.similar(::_NpyBuffer, ::Type{T}, dims::Dims) where {T} = Array{T}(undef, dims)
# Borrows of one vector or one file mapping share their data id, so Julia's
# unaliasing (broadcasts, views) treats them as the same memory.
function Base.dataids(a::_NpyBuffer)
    return if isnothing(a.source.key)
        Base.dataids(a.source.bytes)
    else
        (UInt(objectid(a.source)),)
    end
end
# A private read-only copy of the payload (Julia's broadcast unaliasing).
function Base.unaliascopy(a::_NpyBuffer{T}) where {T}
    bytes = _npy_live(a).source.bytes[a.offset .+ (1:(prod(a.dims) * sizeof(T)))]
    return _NpyBuffer{T}(
        _NpyMapping(bytes, false),
        0,
        a.dims,
        a.fortran,
        false,
        a.shape,
        a.version,
        a.byteorder,
    )
end
function _npy_live(a::_NpyBuffer{T}) where {T}
    length(a.source.bytes) >= a.offset + prod(a.dims) * sizeof(T) ||
        throw(ArgumentError("NPY borrow source bytes were resized below the payload"))
    return a
end
function Base.unsafe_convert(::Type{Ptr{T}}, a::_NpyBuffer{T}) where {T}
    return Ptr{T}(pointer(_npy_live(a).source.bytes) + a.offset)
end
Base.pointer(a::_NpyBuffer{T}) where {T} = Base.unsafe_convert(Ptr{T}, a)
function _npy_index(a::_NpyBuffer, i::Int, j::Int)
    s1, s2 = strides(a)
    return 1 + (i - 1) * s1 + (j - 1) * s2
end
Base.@propagate_inbounds function Base.getindex(a::_NpyBuffer, i::Int, j::Int)
    @boundscheck checkbounds(a, i, j)
    bytes = a.source.bytes
    return GC.@preserve a bytes unsafe_load(pointer(a), _npy_index(a, i, j))
end
Base.@propagate_inbounds function Base.setindex!(
    a::_NpyBuffer{T}, x, i::Int, j::Int
) where {T}
    a.writable ||
        throw(ArgumentError("NPY borrows are read-only; copy to FaerMatrix first"))
    @boundscheck checkbounds(a, i, j)
    v = convert(T, x)
    bytes = a.source.bytes
    GC.@preserve a bytes unsafe_store!(pointer(a), v, _npy_index(a, i, j))
    return a
end
# Validated element-aligned and injective (C/Fortran order) at construction.
_julia_borrowable(::_NpyBuffer) = true
const _NpyBorrow{T} = FaerBorrowedMatrix{T,<:_NpyBuffer{T}}
_borrowed_writable(b::_NpyBorrow) = parent(b).writable
# Borrows expose their parent's memory to Julia's unaliasing.
Base.dataids(b::FaerBorrowedArray) = Base.dataids(parent(b))
Base.unaliascopy(b::FaerBorrowedArray) = FaerBorrowedArray(Base.unaliascopy(parent(b)))
"""
    Mmap.sync!(B::FaerBorrowedMatrix[, flags])

Flush the file mapping behind an [`npyborrow`](@ref) borrow of a path or
stream to the file (a no-op for borrowed byte vectors).
"""
function Mmap.sync!(b::_NpyBorrow, flags::Integer=Mmap.MS_SYNC)
    m = parent(b).source
    isnothing(m.key) || Mmap.sync!(m.bytes, flags)
    return b
end
function npyinfo(b::_NpyBorrow{T}) where {T}
    a = _npy_live(parent(b))
    return (
        dtype=T,
        shape=a.shape,
        matrix_size=a.dims,
        fortran_order=a.fortran,
        version=a.version,
        byteorder=a.byteorder,
        aligned=true,
        normalized_byteorder=Base.ENDIAN_BOM == 0x04030201 ? :little : :big,
        writable=a.writable,
    )
end

# Borrow the NPY data in `source.bytes[start+1:end]`.
function _npy_borrow(source::_NpyMapping, start::Int, writable::Bool, fallback::Bool)
    bytes = source.bytes
    info = zeros(UInt, 8)
    lock(_matrix_lock) do
        try
            GC.@preserve bytes info _native_call(
                RustCrate.npy_source_info,
                pointer(bytes) + start,
                UInt(max(length(bytes) - start, 0)),
                info,
            )
        catch err
            err isa RustCall.RustError && throw(ArgumentError(sprint(showerror, err)))
            rethrow()
        end
    end
    T = (Float32, Float64, ComplexF32, ComplexF64)[Int(info[1]) + 1]
    m, n = Int(info[2]), Int(info[3])
    rank = Int(info[4])
    native = info[8] == 0 || info[8] == (Base.ENDIAN_BOM == 0x04030201 ? 1 : 2)
    buf = _NpyBuffer{T}(
        source,
        length(bytes) - m * n * sizeof(T),
        (m, n),
        info[5] != 0,
        writable,
        if rank == 0
            ()
        elseif rank == 1
            (m,)
        else
            (m, n)
        end,
        (Int(info[6]), Int(info[7])),
        (:irrelevant, :little, :big)[Int(info[8]) + 1],
    )
    native && _aligned(buf) && return FaerBorrowedArray(buf)
    reason = native ? "not element-aligned" : "foreign-endian"
    writable &&
        throw(ArgumentError("NPY payload is $reason; it cannot be borrowed writable"))
    fallback || throw(
        ArgumentError("NPY payload is $reason; it cannot be borrowed without copying")
    )
    return loadnpy(start == 0 ? bytes : view(bytes, (start + 1):length(bytes)))
end
function npyborrow(bytes::Vector{UInt8}; writable::Bool=false, fallback::Bool=true)
    return _npy_borrow(_NpyMapping(bytes, writable), 0, writable, fallback)
end
function npyborrow(bytes::AbstractVector{UInt8}; writable::Bool=false, fallback::Bool=true)
    fallback && !writable ||
        throw(ArgumentError("only a Vector{UInt8} NPY source can be borrowed in place"))
    return loadnpy(bytes)
end
function npyborrow(path::AbstractString; writable::Bool=false, fallback::Bool=true)
    open(path, writable ? "r+" : "r") do io
        m = _npy_mapping(io, writable, () -> open(path, "r+"))
        return _npy_borrow(m, 0, writable, fallback)
    end
end
function npyborrow(io::IOStream; writable::Bool=false, fallback::Bool=true)
    writable &&
        !iswritable(io) &&
        throw(ArgumentError("a writable NPY borrow needs a stream opened for writing"))
    start = Int(position(io))
    return _npy_borrow(_npy_mapping(io, writable, nothing), start, writable, fallback)
end

# The shared mapping of the whole file open as `io`, (re)mapped when absent,
# resized or needed read-write. `reopen()` opens a writable handle when `io`
# is read-only but the shared mapping must stay read-write.
function _npy_mapping(io::IOStream, writable::Bool, reopen)
    st = stat(io)
    # Files without an inode number cannot be identified; they are not shared.
    key = st.inode == 0 ? nothing : (UInt64(st.device), UInt64(st.inode))
    len = Int(filesize(io))
    lock(_matrix_lock) do
        filter!(p -> !isnothing(p.second.value), _npy_mappings)
        m = isnothing(key) ? nothing : get(_npy_mappings, key, WeakRef()).value
        rw = writable || (m isa _NpyMapping && m.writable)
        m isa _NpyMapping && length(m.bytes) == len && m.writable >= rw && return m
        src = io
        if rw && !iswritable(io)
            isnothing(reopen) && throw(
                ArgumentError(
                    "the file is borrowed writable and must be remapped; " *
                    "borrow it through its path or a writable stream",
                ),
            )
            src = reopen()
        end
        bytes = try
            Mmap.mmap(src, Vector{UInt8}, (len,), 0; grow=false, shared=true)
        finally
            src === io || close(src)
        end
        if m isa _NpyMapping
            push!(m.retired, m.bytes)
            m.bytes = bytes
            m.writable = rw
            return m
        end
        m = _NpyMapping(bytes, rw, key, Vector{UInt8}[])
        isnothing(key) || (_npy_mappings[key] = WeakRef(m))
        return m
    end
end
