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
input bytes. The normalized buffer is retained until close/GC. Copying a view
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
    h=a.handle
    isnothing(h) && throw(ArgumentError("NPY view has been closed"))
    return h
end
_native_argument(a::FaerNpyMatrix) = _handle(a)
_is_native_owner(::FaerNpyMatrix) = true
Base.size(a::FaerNpyMatrix) = a.dims
Base.size(a::FaerNpyMatrix, d::Integer) =
    d<1 ? throw(ArgumentError("dimension must be positive")) : d<=2 ? a.dims[d] : 1
Base.isopen(a::FaerNpyMatrix) = lock(()->!isnothing(a.handle), _matrix_lock)
function Base.close(a::FaerNpyMatrix)
    lock(_matrix_lock) do ;
        a.handle=nothing
    end
    return nothing
end
function _npy_view(input::AbstractVector{UInt8})
    data=Vector{UInt8}(input)
    lock(_matrix_lock) do
        h=RustCrate.OwnedNpy()
        info=zeros(UInt, 9)
        try
            GC.@preserve data info begin
                _native_call(RustCrate.npy_load, h, pointer(data), UInt(length(data)))
                _native_call(RustCrate.npy_info, h, pointer(info))
            end
        catch err
            err isa RustCall.RustError && throw(ArgumentError(sprint(showerror, err)))
            rethrow()
        end
        T=(Float32, Float64, ComplexF32, ComplexF64)[Int(info[1])+1]
        dims=(Int(info[2]), Int(info[3]))
        rank=Int(info[4])
        shape=rank==0 ? () : rank==1 ? (dims[1],) : dims
        return FaerNpyMatrix{T,typeof(h)}(
            h,
            dims,
            shape,
            info[5]!=0,
            (Int(info[6]), Int(info[7])),
            (:irrelevant, :little, :big)[Int(info[8])+1],
            info[9]!=0,
        )
    end
end
# Contiguous byte sources are passed in place; faer's to_mat reads unaligned
# and foreign-endian payloads directly (tensor4all/faer#215).
_npy_source(data::DenseVector{UInt8}) = data
_npy_source(data::Base.FastContiguousSubArray{UInt8,1,<:DenseVector{UInt8}}) = data
_npy_source(data::AbstractVector{UInt8}) = Vector{UInt8}(data)
function _npy_matrix(input::AbstractVector{UInt8})
    data=_npy_source(input)
    lock(_matrix_lock) do
        info=zeros(UInt, 8)
        try
            GC.@preserve data info begin
                p=pointer(data)
                len=UInt(length(data))
                _native_call(RustCrate.npy_source_info, p, len, pointer(info))
                T=(Float32, Float64, ComplexF32, ComplexF64)[Int(info[1])+1]
                m, n=Int(info[2]), Int(info[3])
                return T===Float64 ?
                       _matrix_result(RustCrate.npy_load_direct, (m, n), p, len, UInt(m), UInt(n)) :
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
        catch err
            err isa RustCall.RustError && throw(ArgumentError(sprint(showerror, err)))
            rethrow()
        end
    end
end
loadnpy(data::AbstractVector{UInt8}; view::Bool = false) =
    view ? _npy_view(data) : _npy_matrix(data)
loadnpy(path::AbstractString; kwargs...) = loadnpy(read(path); kwargs...)
loadnpy(io::IO; kwargs...) = loadnpy(read(io); kwargs...)
"""Read an NPY source into an owned read-only aligned matrix view; see [`loadnpy`](@ref)."""
npyview(source) = loadnpy(source; view = true)
"""
    npyinfo(source)

Return dtype, original shape/order/version/byteorder, matrix_size and alignment
metadata. Sources can be an existing open FaerNpyMatrix, NPY bytes, an IO or a
path. Raw bytes are normalized to aligned native byteorder for safe faer views;
byteorder reports the source's order, normalized_byteorder reports storage.
IO inputs are read from their current position to EOF.
"""
function npyinfo(a::FaerNpyMatrix{T}) where {T}
    lock(_matrix_lock) do
        _handle(a)
        return (
            dtype = T,
            shape = a.shape,
            matrix_size = size(a),
            fortran_order = a.fortran_order,
            version = a.version,
            byteorder = a.byteorder,
            aligned = a.aligned,
            normalized_byteorder = Base.ENDIAN_BOM==0x04030201 ? :little : :big,
        )
    end
end
npyinfo(source) = npyinfo(npyview(source))
function Base.getindex(a::FaerNpyMatrix{T}, i::Int, j::Int) where {T}
    lock(_matrix_lock) do
        _handle(a)
        checkbounds(a, i, j)
        out=Ref{T}()
        GC.@preserve out _native_call(
            RustCrate.npy_element,
            a,
            UInt(i-1),
            UInt(j-1),
            Ptr{UInt8}(Base.unsafe_convert(Ptr{T}, out)),
            _tag(T),
            UInt(size(a, 1)),
            UInt(size(a, 2)),
        )
        return out[]
    end
end
Base.setindex!(::FaerNpyMatrix, args...) =
    throw(ArgumentError("NPY views are read-only; copy to FaerMatrix first"))
function Base.Matrix(a::FaerNpyMatrix{T}) where {T}
    lock(_matrix_lock) do
        _handle(a)
        out=Matrix{T}(undef, size(a))
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
        return T===Float64 ?
               _matrix_result(
            RustCrate.npy_copy_from,
            size(a),
            a,
            UInt(size(a, 1)),
            UInt(size(a, 2)),
        ) :
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
FaerMatrix{T}(a::FaerNpyMatrix{T}) where {T<:_NativeScalar} = FaerMatrix(a)
Base.copy(a::FaerNpyMatrix) = FaerMatrix(a)
function Base.deepcopy_internal(a::FaerNpyMatrix{T}, d::IdDict) where {T}
    get!(d, a) do
        lock(_matrix_lock) do
            h=_native_call(
                RustCrate.npy_duplicate,
                a,
                _tag(T),
                UInt(size(a, 1)),
                UInt(size(a, 2)),
            )
            return FaerNpyMatrix{T,typeof(h)}(
                h,
                a.dims,
                a.shape,
                a.fortran_order,
                a.version,
                a.byteorder,
                a.aligned,
            )
        end
    end
end
