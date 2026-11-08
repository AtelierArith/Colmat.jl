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
    (RustCrate.borrowed_sparse_mul, (10,)),
    (RustCrate.borrowed_sparse_product, (10,)),
    (RustCrate.sparse_owner_mul, (4,)),
    (RustCrate.sparse_owner_product, (4,)),
    (RustCrate.dense_triangular_into, (3,)),
    (RustCrate.dense_triangular_raw, (1,)),
    (RustCrate.dense_triangular_requirements, (1,)),
    (RustCrate.dense_triangular_zero_diagonal, (1,)),
    (RustCrate.factor_solve_into, (2,)),
    (RustCrate.diagonal_scale_into, (3, 5)),
    (RustCrate.diagonal_scaled, (2, 4)),
    (RustCrate.diagonal_scaled_vector, (2, 4)),
    (RustCrate.diagonal_matrix, (3,)),
    (RustCrate.diagonal_matrix_values, (4,)),
    (RustCrate.diagonal_of_view, (2,)),
    (RustCrate.diagonal_vector, (3,)),
    (RustCrate.diagonal_vector_values, (4,)),
    (RustCrate.householder_mul_into, (1, 3)),
    (RustCrate.sequence_mul_into, (1, 3)),
    (RustCrate.compact_tridiagonal_mul_into, (1, 3)),
    (RustCrate.compact_bidiagonal_mul_into, (9, 11)),
    (RustCrate.operator_mul_into, (1, 3)),
    (RustCrate.operator_bicgstab, (3, 4)),
    (RustCrate.operator_cg, (3,)),
    (RustCrate.pivoted_llt_into, (1,)),
    (RustCrate.operator_load, (2,)),
    (RustCrate.operator_lsmr, (3,)),
    (RustCrate.partial_workspace, (7,)),
    (RustCrate.permute_from_view, (2,)),
    (RustCrate.qz_from, (10, 11)),
    (RustCrate.sparse_kron, (2, 4)),
    (RustCrate.sparse_analyze, (6,)),
    (RustCrate.sparse_analyze_workspace, (9,)),
    (RustCrate.sparse_analyze_raw, (10,)),
    (RustCrate.sparse_analyze_workspace_raw, (11,)),
    (RustCrate.statistics_view_reduce, (1,)),
    (RustCrate.vector_binary_views, (2, 4)),
    (RustCrate.vector_from_view, (2,)),
    (RustCrate.structured_product, (2, 5)),
    (RustCrate.structured_product_into, (4, 7)),
    (RustCrate.structured_vector, (2, 5)),
    (RustCrate.structured_view_into, (1, 4, 7)),
    (RustCrate.selfadjoint_expand, (2,)),
    (RustCrate.vector_view_kron, (2, 4)),
    (RustCrate.view_assign, (3,)),
    (RustCrate.view_convert, (2,)),
    (RustCrate.view_assign_into, (1, 3)),
    (RustCrate.view_gather_into, (1, 3)),
    (RustCrate.view_scatter_into, (1, 5)),
    (RustCrate.view_strides, (1,)),
    (RustCrate.view_measure, (1,)),
    (RustCrate.view_binary, (2, 4)),
    (RustCrate.view_copy, (2,)),
    (RustCrate.view_promoted, (2,)),
    (RustCrate.view_dot, (1, 3)),
    (RustCrate.view_kron, (2, 4)),
    (RustCrate.view_kron_into, (3, 5)),
    (RustCrate.view_gemm, (1, 3, 5)),
    (RustCrate.view_predicate, (1,)),
    (RustCrate.view_structure, (1,)),
    (RustCrate.initialize_view, (2,)),
    (RustCrate.solve_in_place_view, (2,)),
    (RustCrate.view_svd_buffers, (1,)),
    (RustCrate.view_eigen_buffers, (1,)),
    (RustCrate.view_reduce, (1,)),
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
        vec(data),
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
    GC.@preserve out _native_call(RustCrate.copy_buffer, a, vec(out), UInt(length(out)))
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

# Dense mul! into native destinations is in dense_destinations.jl.

faer_mul(a::FaerMatrix{Float64}, b::FaerMatrix{Float64}) = a * b
faer_solve(a::FaerMatrix{Float64}, b::AbstractMatrix{<:Real}) = a \ b
faer_solve(a::FaerMatrix{Float64}, b::AbstractVector{<:Real}) = a \ b

"""
    svd(A::FaerMatrix; full=false, alg=nothing, scratch, team)

Compute the SVD directly from the Rust-owned matrix using t4a-faer. Return a
`LinearAlgebra.SVD` with independent Julia factors. The default is a thin SVD;
`full=true` returns square orthogonal factors. `A` is not modified.

`alg=LinearAlgebra.DivideAndConquer()` (or `nothing`) uses faer's default
driver; `alg=QRIteration()` forces the bidiagonal implicit QR algorithm at every
size (also for `svdvals`, `svd!` and `svdvals!`). Other `SvdParams` controls are
available through [`with_svd_tuning`](@ref). LAPACK-specific tolerance keywords
are not supported.
"""
function LinearAlgebra.svd(
    a::FaerMatrix{Float64};
    full::Bool = false,
    native::Bool = false,
    scratch = nothing,
    team = nothing,
    alg = nothing,
)
    alg===nothing || return _svd_alg(alg) do
        LinearAlgebra.svd(a; full, native, scratch, team)
    end
    native && return _owned_spectral(a; full, scratch, team)
    _svd_default_path(scratch, team) || return _svd_workspace_result(a; full, scratch, team)
    m, n = size(a)
    k = min(m, n)
    u = Matrix{Float64}(undef, m, full ? m : k)
    s = Vector{Float64}(undef, k)
    v = Matrix{Float64}(undef, n, full ? n : k)
    GC.@preserve u s v begin
        _native_call(RustCrate.svd_buffers, a, vec(u), s, vec(v), full)
    end
    return SVD(u, s, copy(transpose(v)))
end

"""
    svdvals(A::FaerMatrix{Float64})

Compute singular values directly from Rust-owned storage, without singular
vectors. Values are nonnegative and sorted in descending order.
"""
function LinearAlgebra.svdvals(
    a::FaerMatrix{Float64};
    scratch = nothing,
    team = nothing,
    alg = nothing,
)
    alg===nothing || return _svd_alg(()->LinearAlgebra.svdvals(a; scratch, team), alg)
    _svd_default_path(scratch, team) ||
        return _svd_workspace_result(a; values_only = true, scratch, team)
    s = Vector{Float64}(undef, min(size(a)...))
    GC.@preserve s _native_call(RustCrate.singular_values_buffer, a, s)
    return s
end

faer_svd(a::FaerMatrix{Float64}) = LinearAlgebra.svd(a)
