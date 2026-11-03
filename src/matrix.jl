# RustCall's generated handles own the Rust boxes and finalize them on GC.
# Serialize access (including close) so Rust mutable borrows never overlap.
const _matrix_lock = ReentrantLock()

"""
    FaerMatrix(A::AbstractMatrix)
    FaerMatrix{T}(A::AbstractMatrix)

An `AbstractMatrix{T}` owning a t4a-faer matrix in Rust. Float32, Float64,
ComplexF32 and ComplexF64 inputs preserve their scalar type; other real inputs
use Float64. The explicit constructor converts to a supported `T`.
Construction copies `A`; arithmetic stays in Rust. `Matrix(A)` returns an
independent Julia copy. Operations currently use `Par::Seq`.
"""
mutable struct FaerMatrix{T,H} <: AbstractMatrix{T}
    handle::Union{Nothing,H}
    dims::Tuple{Int,Int}
end

_owned(handle::H, dims) where {H} = FaerMatrix{Float64,H}(handle, dims)
function _handle(a::FaerMatrix)
    h = a.handle
    isnothing(h) && throw(ArgumentError("FaerMatrix has been closed"))
    return h
end
_native_argument(x) = x
_native_argument(a::FaerMatrix) = _handle(a)
_is_native_owner(::Any) = false
_is_native_owner(::FaerMatrix) = true
function _native_call(f, args...)
    lock(_matrix_lock) do
        native = map(_native_argument, args)
        # v0.9.1 does not lower non-receiver &OwnedMatrix arguments to
        # opaque pointers. Borrow explicit pointers while retaining all owners.
        ffi_args = ntuple(length(args)) do i
            i > 1 && _is_native_owner(args[i]) ? Ptr{UInt8}(native[i].ptr) : native[i]
        end
        return GC.@preserve native RustCall.result_to_exception(f(ffi_args...))
    end
end

# Output is always a fresh Rust owner, so its mutable borrow cannot alias inputs.
function _matrix_result(f, dims, args...)
    lock(_matrix_lock) do
        h = RustCrate.OwnedMatrix()
        _native_call(f, h, args...)
        return _owned(h, dims)
    end
end

function FaerMatrix(a::AbstractMatrix{<:Real})
    data = _dense(a)
    return GC.@preserve data _matrix_result(
        RustCrate.load_buffer,
        size(data),
        pointer(data),
        UInt(size(data, 1)),
        UInt(size(data, 2)),
    )
end
FaerMatrix(a::FaerMatrix{Float64}) = copy(a)

Base.size(a::FaerMatrix) = a.dims
Base.IndexStyle(::Type{<:FaerMatrix}) = IndexCartesian()
function Base.getindex(a::FaerMatrix{Float64}, i::Int, j::Int)
    @boundscheck checkbounds(a, i, j)
    return _native_call(RustCrate.read_element, a, UInt(i - 1), UInt(j - 1))
end
function Base.setindex!(a::FaerMatrix{Float64}, x, i::Int, j::Int)
    @boundscheck checkbounds(a, i, j)
    _native_call(RustCrate.write_element, a, UInt(i - 1), UInt(j - 1), Float64(x))
    return a
end

function Base.Matrix(a::FaerMatrix{Float64})
    out = Matrix{Float64}(undef, size(a))
    GC.@preserve out _native_call(RustCrate.copy_buffer, a, pointer(out), UInt(length(out)))
    return out
end
Base.Array(a::FaerMatrix{Float64}) = Matrix(a)
Base.Matrix{T}(a::FaerMatrix{Float64}) where {T} = convert(Matrix{T}, Matrix(a))
Base.copy(a::FaerMatrix{Float64}) = _matrix_result(RustCrate.duplicate_from, size(a), a)
function Base.deepcopy_internal(a::FaerMatrix{Float64}, dict::IdDict)
    return get!(dict, a) do
        copy(a)
    end
end

function Base.similar(a::FaerMatrix{Float64}, ::Type{T}, dims::Dims) where {T}
    if T <: _NativeScalar && length(dims)==2
        return _zeros(T, dims)
    end
    return Array{T}(undef, dims)
end

Base.zero(a::FaerMatrix{Float64}) = similar(a, Float64, size(a))
function Base.fill!(a::FaerMatrix{Float64}, x)
    _native_call(RustCrate.fill_value, a, Float64(x))
    return a
end
function Base.copyto!(dest::FaerMatrix{Float64}, src::FaerMatrix{Float64})
    size(dest) == size(src) || throw(DimensionMismatch("matrix sizes must match"))
    lock(_matrix_lock) do
        _handle(dest)
        _handle(src)
        # Avoid passing aliased &mut and & references through the Rust ABI.
        dest === src || _native_call(RustCrate.assign, dest, src)
    end
    return dest
end

"""Invalidate a matrix and release its reference to the GC-managed Rust owner."""
function Base.close(a::FaerMatrix)
    lock(_matrix_lock) do
        a.handle = nothing
    end
    return nothing
end
Base.isopen(a::FaerMatrix) = lock(() -> !isnothing(a.handle), _matrix_lock)

function _add(a::FaerMatrix{Float64}, b::FaerMatrix{Float64}, sign::Float64)
    size(a) == size(b) || throw(DimensionMismatch("matrix sizes must match"))
    return _matrix_result(RustCrate.add_from, size(a), a, b, sign)
end
Base.:+(a::FaerMatrix{Float64}, b::FaerMatrix{Float64}) = _add(a, b, 1.0)
Base.:-(a::FaerMatrix{Float64}, b::FaerMatrix{Float64}) = _add(a, b, -1.0)
Base.:+(a::FaerMatrix{Float64}) = copy(a)
Base.:-(a::FaerMatrix{Float64}) = -1.0 * a
Base.:*(x::Real, a::FaerMatrix{Float64}) =
    _matrix_result(RustCrate.scale_from, size(a), a, Float64(x))
Base.:*(a::FaerMatrix{Float64}, x::Real) = x * a
Base.:/(a::FaerMatrix{Float64}, x::Real) = inv(Float64(x)) * a
Base.:\(x::Real, a::FaerMatrix{Float64}) = a / x

function Base.:*(a::FaerMatrix{Float64}, b::FaerMatrix{Float64})
    size(a, 2) == size(b, 1) || throw(DimensionMismatch("inner dimensions must match"))
    return _matrix_result(RustCrate.multiply_from, (size(a, 1), size(b, 2)), a, b)
end
function Base.:*(a::FaerMatrix{Float64}, b::AbstractVector{<:Real})
    return vec(Matrix(a * FaerMatrix(reshape(b, :, 1))))
end
function Base.:\(a::FaerMatrix{Float64}, b::FaerMatrix{Float64})
    return LinearAlgebra.factorize(a) \ b
end
Base.:\(a::FaerMatrix{Float64}, b::AbstractMatrix{<:Real}) = a \ FaerMatrix{Float64}(b)
Base.:\(a::FaerMatrix{Float64}, b::AbstractVector{<:Real}) =
    vec(Matrix(a \ FaerMatrix(reshape(b, :, 1))))

# Julia's transpose/adjoint wrappers remain lazy; copying materializes in Rust.
function Base.copy(a::LinearAlgebra.Transpose{Float64,<:FaerMatrix{Float64}})
    return _matrix_result(RustCrate.transpose_from, reverse(size(parent(a))), parent(a))
end
function Base.copy(a::LinearAlgebra.Adjoint{Float64,<:FaerMatrix{Float64}})
    return copy(transpose(parent(a)))
end

function LinearAlgebra.mul!(
    c::FaerMatrix{Float64},
    a::FaerMatrix{Float64},
    b::FaerMatrix{Float64},
    alpha::Number,
    beta::Number,
)
    size(a, 2) == size(b, 1) || throw(DimensionMismatch("inner dimensions must match"))
    size(c) == (size(a, 1), size(b, 2)) ||
        throw(DimensionMismatch("output dimensions must match"))
    # Compute before writing, making C === A and C === B safe. beta == 0
    # must not read C, which may contain NaNs or unspecified values.
    lock(_matrix_lock) do
        _handle(a)
        _handle(b)
        _handle(c)
        if iszero(alpha)
            return iszero(beta) ? fill!(c, 0.0) : copyto!(c, beta * c)
        end
        product = alpha * (a * b)
        copyto!(c, iszero(beta) ? product : product + beta * c)
    end
    return c
end
LinearAlgebra.mul!(c::FaerMatrix{Float64}, a::FaerMatrix{Float64}, b::FaerMatrix{Float64}) =
    LinearAlgebra.mul!(c, a, b, 1.0, 0.0)

faer_mul(a::FaerMatrix{Float64}, b::FaerMatrix{Float64}) = a * b
faer_solve(a::FaerMatrix{Float64}, b::AbstractMatrix{<:Real}) = a \ b
faer_solve(a::FaerMatrix{Float64}, b::AbstractVector{<:Real}) = a \ b

"""
    svd(A::FaerMatrix{Float64}; full=false)

Compute the SVD directly from the Rust-owned matrix using t4a-faer. Return a
`LinearAlgebra.SVD` with independent Julia factors. The default is a thin SVD;
`full=true` returns square orthogonal factors. `A` is not modified.

This method uses faer's SVD algorithm; LAPACK-specific algorithm and tolerance
keywords are not supported.
"""
function LinearAlgebra.svd(a::FaerMatrix{Float64}; full::Bool = false)
    m, n = size(a)
    k = min(m, n)
    u = Matrix{Float64}(undef, m, full ? m : k)
    s = Vector{Float64}(undef, k)
    v = Matrix{Float64}(undef, n, full ? n : k)
    GC.@preserve u s v begin
        _native_call(RustCrate.svd_buffers, a, pointer(u), pointer(s), pointer(v), full)
    end
    return SVD(u, s, copy(transpose(v)))
end

"""
    svdvals(A::FaerMatrix{Float64})

Compute singular values directly from Rust-owned storage, without singular
vectors. Values are nonnegative and sorted in descending order.
"""
function LinearAlgebra.svdvals(a::FaerMatrix{Float64})
    s = Vector{Float64}(undef, min(size(a)...))
    GC.@preserve s _native_call(RustCrate.singular_values_buffer, a, pointer(s))
    return s
end

faer_svd(a::FaerMatrix{Float64}) = LinearAlgebra.svd(a)
