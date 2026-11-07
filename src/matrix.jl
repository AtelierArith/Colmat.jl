# RustCall's generated handles own the Rust boxes and finalize them on GC.
# Serialize access (including close) so Rust mutable borrows never overlap.
const _matrix_lock = ReentrantLock()

"""
    FaerMatrix(A::AbstractMatrix)
    FaerMatrix{T}(A::AbstractMatrix)

An `AbstractMatrix{T}` owning a t4a-faer matrix in Rust. Float32, Float64,
ComplexF32 and ComplexF64 inputs preserve their scalar type; other real inputs
(integers, `Rational`, `Float16`, `BigFloat`, ...) use Float64 and other complex
inputs use ComplexF64. `FaerMatrix{T}(A)` converts to a supported `T` and
throws `ArgumentError` for any other `T`. Scalar `*`, `/` and `\\` follow
`promote_type` and throw `ArgumentError` when the promoted type is not one of the
four native types; faer's `fx128`/`cx128` and generic field traits are not exposed.
Construction validates converted storage and shape before copying `A` to Rust.
Arithmetic stays in Rust. `Matrix(A)` returns an
independent Julia copy. Operations currently use `Par::Seq`.
"""
mutable struct FaerMatrix{T,H} <: AbstractMatrix{T}
    handle::Union{Nothing,H}
    dims::Tuple{Int,Int}
    epoch::UInt
end
FaerMatrix{T,H}(handle, dims::Tuple{Int,Int}) where {T,H} =
    FaerMatrix{T,H}(handle, dims, UInt(0))

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
# Only genuinely type-erased or optional owner arguments still cross as raw
# pointers. Typed Rust references go through RustCall's checked handle borrowing.
_raw_owner_argument(x, owner) = _is_native_owner(owner) ? Ptr{UInt8}(x.ptr) : x
const _raw_owner_arguments = (
    (RustCrate.operator_bicgstab, (3, 4)),
    (RustCrate.operator_cg, (3,)),
    (RustCrate.operator_load, (2,)),
    (RustCrate.operator_lsmr, (3,)),
    (RustCrate.qz_from, (10, 11)),
    (RustCrate.sparse_analyze, (6,)),
    (RustCrate.statistics_view_reduce, (1,)),
    (RustCrate.vector_binary_views, (2, 4)),
    (RustCrate.vector_from_view, (2,)),
    (RustCrate.view_assign, (3,)),
    (RustCrate.view_measure, (1,)),
    (RustCrate.view_binary, (2, 4)),
    (RustCrate.view_copy, (2,)),
)
function _raw_owner_positions(f)
    for (member, positions) in _raw_owner_arguments
        f === member && return positions
    end
    return ()
end
_operator_callback_owners(::Any) = nothing
function _native_call_plain(f, args...)
    native = map(_native_argument, args)
    raw_positions = _raw_owner_positions(f)
    ffi_args = ntuple(length(args)) do i
        i in raw_positions ? _raw_owner_argument(native[i], args[i]) : native[i]
    end
    return GC.@preserve native RustCall.result_to_exception(f(ffi_args...))
end
function _native_call(f, args...)
    lock(_matrix_lock) do
        any(x -> !isnothing(_operator_callback_owners(x)), args) ?
        _native_call_with_callbacks(f, args...) : _native_call_plain(f, args...)
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

function _matrix_from_dense(data::Matrix{Float64})
    return GC.@preserve data _matrix_result(
        RustCrate.load_buffer,
        size(data),
        pointer(data),
        UInt(size(data, 1)),
        UInt(size(data, 2)),
    )
end
FaerMatrix(a::AbstractMatrix{<:Real}) = _matrix_from_dense(_dense(a))
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
function Base.copy(a::FaerMatrix{T}) where {T}
    lock(_matrix_lock) do
        h = _native_call(RustCrate.duplicate, a)
        return FaerMatrix{T,typeof(h)}(h, size(a))
    end
end
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
        a.epoch += 1
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
function Base.:*(x::Real, a::FaerMatrix{Float64})
    _native_promotion(typeof(x), Float64)
    return _matrix_result(RustCrate.scale_from, size(a), a, Float64(x))
end
Base.:*(a::FaerMatrix{Float64}, x::Real) = x * a
function Base.:/(a::FaerMatrix{Float64}, x::Real)
    _native_promotion(typeof(x), Float64)
    return inv(Float64(x)) * a
end
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
function LinearAlgebra.svd(
    a::FaerMatrix{Float64};
    full::Bool = false,
    native::Bool = false,
    scratch = nothing,
    team = nothing,
)
    native && return _owned_spectral(a; full, scratch, team)
    (!isnothing(scratch) || !isnothing(team)) &&
        return _svd_workspace_result(a; full, scratch, team)
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
function LinearAlgebra.svdvals(a::FaerMatrix{Float64}; scratch = nothing, team = nothing)
    (!isnothing(scratch) || !isnothing(team)) &&
        return _svd_workspace_result(a; values_only = true, scratch, team)
    s = Vector{Float64}(undef, min(size(a)...))
    GC.@preserve s _native_call(RustCrate.singular_values_buffer, a, pointer(s))
    return s
end

faer_svd(a::FaerMatrix{Float64}) = LinearAlgebra.svd(a)
