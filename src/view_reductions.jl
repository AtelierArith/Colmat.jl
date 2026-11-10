# dot, norms, sums, extrema and finite/NaN tests read native owners,
# strided/reversed views and faerview arrays in place through view descriptors
# (deps/rust/src/view_reductions.rs). Forms that need Julia callbacks or
# Julia's init/promotion rules run Julia's generic code on one snapshot (the
# parent array itself for borrowed arrays).
const _ReductionView = Union{FaerView,FaerBorrowedArray}
const _ReductionOwner = Union{FaerMatrix,FaerVector,FaerRow}
const _ReductionArray = Union{_ReductionOwner,_ReductionView}
const _RealNative = Union{Float32,Float64}
const _REDUCE_SUM = Int32(0)
const _REDUCE_MINIMUM = Int32(1)
const _REDUCE_MAXIMUM = Int32(2)
const _REDUCE_EXTREMA = Int32(3)
const _REDUCE_NORM = Int32(4)

_reduction_snapshot(a::FaerBorrowedArray) = parent(a)
_reduction_snapshot(a) = Array(a)

# `output` addresses `len` output slots preserved by the caller.
function _view_reduce(output::Ptr, a, len, code, axis, p=0.0)
    owner, d = _view_source(a)
    GC.@preserve d _native_call(
        RustCrate.view_reduce,
        owner,
        pointer(d),
        Ptr{UInt8}(output),
        UInt(len),
        code,
        axis,
        Float64(p),
    )
    return nothing
end

function _native_sum(a, dims)
    lock(_matrix_lock) do
        axis, len, shape, scalar = _statistics_layout(a, dims)
        output = Vector{eltype(a)}(undef, len)
        GC.@preserve output _view_reduce(pointer(output), a, len, _REDUCE_SUM, axis)
        return _statistics_output(output, shape, scalar)
    end
end
"""
    sum(A::Union{FaerView,FaerBorrowedArray}; dims=:)

Sum a native view or borrowed Julia array in place with faer's reductions.
`dims` and output shapes follow `Base.sum` (`dims` may be an integer or a tuple;
dimensions past `ndims(A)` reduce nothing). Summation order is faer's, so
results can differ from Julia's pairwise sum by rounding. `init` and the
function-first form `sum(f, A)` use Julia's generic implementation.
"""
function Base.sum(a::_ReductionView; dims=:, kw...)
    isempty(kw) || return sum(_reduction_snapshot(a); dims, kw...)
    return _native_sum(a, dims)
end

# Julia throws when any reduced dimension is empty, even for an empty output.
function _check_reduced_extent(a, dims)
    extent = if dims isa Colon
        length(a)
    else
        prod(
            d -> d <= ndims(a) ? size(a, d) : 1,
            unique(dims isa Integer ? (dims,) : dims);
            init=1,
        )
    end
    extent == 0 && throw(
        ArgumentError(
            "reducing over an empty collection is not allowed; consider supplying `init` to the reducer",
        ),
    )
    return nothing
end
function _native_extremum(a, dims, code)
    lock(_matrix_lock) do
        axis, len, shape, scalar = _statistics_layout(a, dims)
        _check_reduced_extent(a, dims)
        T = eltype(a)
        output = Vector{code == _REDUCE_EXTREMA ? Tuple{T,T} : T}(undef, len)
        GC.@preserve output _view_reduce(pointer(output), a, len, code, axis)
        return _statistics_output(output, shape, scalar)
    end
end
for (f, code) in (
    (:minimum, :_REDUCE_MINIMUM), (:maximum, :_REDUCE_MAXIMUM), (:extrema, :_REDUCE_EXTREMA)
)
    # faerview arrays keep Julia's vectorized generic reductions over the
    # parent (zero-copy, several times faster than this scalar kernel).
    for A in (:FaerMatrix, :FaerVector, :FaerRow, :FaerView)
        @eval begin
            function Base.$f(a::$A{T}; dims=:, kw...) where {T<:_RealNative}
                isempty(kw) || return $f(_reduction_snapshot(a); dims, kw...)
                return _native_extremum(a, dims, $code)
            end
            function Base.$f(g, a::$A{T}; dims=:, kw...) where {T<:_RealNative}
                g === identity && isempty(kw) && return _native_extremum(a, dims, $code)
                return $f(g, _reduction_snapshot(a); dims, kw...)
            end
        end
    end
end

# `sum!`/`minimum!`/`maximum!`/`extrema!(r, A; init)`: the reduced dimensions
# are those where `r` has extent 1 (Base's rules); the O(length(r)) result is
# computed by the kernels above and written into `r` with one `copyto!`
# (native for native/borrowed destinations), so a destination viewing `A`
# sees the reduction of the old values. Shapes Base rejects, empty sources,
# mixed scalar types and complex min/max run Base on one snapshot.
function _reduction_dims(r, a)
    ndims(r) <= ndims(a) || return nothing
    all(d -> size(r, d) == 1 || size(r, d) == size(a, d), 1:ndims(a)) || return nothing
    return Tuple(d for d in 1:ndims(a) if size(r, d) == 1 && size(a, d) != 1)
end
_reduction_values(r::_ReductionArray) = _reduction_snapshot(r)
_reduction_values(r) = r
# Base's in-place reduction of a snapshot of `a`; native destinations are
# updated in one transfer.
function _julia_reduce!(f!, r, a, init)
    src = a isa _ReductionArray ? _reduction_snapshot(a) : a
    r isa FaerBorrowedArray && return (f!(parent(r), src; init); r)
    r isa _ReductionArray || return f!(r, src; init)
    work = f!(Array(r), src; init)
    copyto!(r, work)
    return r
end
function _native_reduce!(f!, reduce, combine, r, a, init, real_only)
    _check_writable(r)
    T = eltype(a)
    dims = a isa _ReductionArray ? _reduction_dims(r, a) : nothing
    S = f! === extrema! ? Tuple{T,T} : T
    native =
        !isnothing(dims) &&
        !isempty(a) &&
        eltype(r) === S &&
        (!real_only || T <: _RealNative)
    native || return _julia_reduce!(f!, r, a, init)
    out = reduce(a, dims)
    init || (out = combine.(reshape(_reduction_values(r), size(out)), out))
    copyto!(r, reshape(out, size(r)))
    return r
end
_combine_extrema(x, y) = (min(x[1], y[1]), max(x[2], y[2]))
# Borrowed sources keep Julia's vectorized min/max over the parent.
function _reduce_extremum(f, code)
    return (a, dims) ->
        a isa FaerBorrowedArray ? f(parent(a); dims) : _native_extremum(a, dims, code)
end
for (f!, f, code, combine) in (
    (:minimum!, :minimum, :_REDUCE_MINIMUM, :min),
    (:maximum!, :maximum, :_REDUCE_MAXIMUM, :max),
    (:extrema!, :extrema, :_REDUCE_EXTREMA, :_combine_extrema),
)
    for (R, A) in (
        (:AbstractArray, :_ReductionArray),
        (:_ReductionArray, :AbstractArray),
        (:_ReductionArray, :_ReductionArray),
    )
        @eval Base.$f!(r::$R, a::$A; init::Bool=true) =
            _native_reduce!($f!, _reduce_extremum($f, $code), $combine, r, a, init, true)
    end
end
for (R, A) in (
    (:AbstractArray, :_ReductionArray),
    (:_ReductionArray, :AbstractArray),
    (:_ReductionArray, :_ReductionArray),
)
    @eval Base.sum!(r::$R, a::$A; init::Bool=true) =
        _native_reduce!(sum!, _native_sum, +, r, a, init, false)
end
"""
    sum!(r, A; init=true)
    minimum!(r, A; init=true)
    maximum!(r, A; init=true)
    extrema!(r, A; init=true)

In-place reductions with `A` a native owner, view or borrowed Julia array, or
`r` one of those. Reduced dimensions are those where `r` has extent 1, as in
Base. faer's kernels compute the `length(r)` results (borrowed sources keep
Julia's vectorized min/max over the parent), which are written into `r` with
one native transfer; `init=false` combines them with the current values of `r`
(`+`, `min`, `max`; NaN propagates). A destination viewing `A` receives the
reduction of the old values. Shape errors, empty sources, scalar types that
differ between `r` and `A`, complex min/max and Julia sources run Base's
method on one snapshot of `A` (writing a native `r` in one transfer).
"""
Base.sum!(::AbstractArray, ::_ReductionArray)

function _native_predicate(a, code)
    lock(_matrix_lock) do
        owner, d = _view_source(a)
        GC.@preserve d return _native_call(
            RustCrate.view_predicate, owner, pointer(d), code
        )
    end
end
function Base.all(::typeof(isfinite), a::_ReductionView; dims=:)
    return if dims isa Colon
        _native_predicate(a, Int32(0))
    else
        all(isfinite, _reduction_snapshot(a); dims)
    end
end
function Base.any(::typeof(isnan), a::_ReductionView; dims=:)
    return if dims isa Colon
        _native_predicate(a, Int32(1))
    else
        any(isnan, _reduction_snapshot(a); dims)
    end
end

# Every real `p` but NaN runs natively: 1/2/Inf with faer's reductions, 0 and
# -Inf directly, other p as LinearAlgebra.generic_normp (Float64 accumulation,
# rescaling by the extreme magnitude; integer p by repeated multiplication).
function _native_norm(a, p::Real)
    isnan(p) && return LinearAlgebra.norm(_reduction_snapshot(a), p)
    R = _realtype(eltype(a))
    lock(_matrix_lock) do
        value = Ref{R}()
        GC.@preserve value _view_reduce(
            Base.unsafe_convert(Ptr{R}, value), a, 1, _REDUCE_NORM, _STATS_ALL, p
        )
        return value[]
    end
end
"""
    norm(A::Union{FaerView,FaerBorrowedArray}, p::Real=2)

Entrywise `p`-norm (as `LinearAlgebra.norm`, treating `A` as a vector) read in
place from a native view or borrowed Julia array. faer's `norm_l2`/`norm_l1`/
`norm_max` serve 2/1/Inf (NaN propagates as in Julia). Every other real `p`
except NaN is native too: nonzero counting for 0, NaN-propagating minimum
magnitude for -Inf, and Julia's `generic_normp` algorithm otherwise (Float64
accumulation for Float32 data, rescaling by the largest/smallest magnitude).
Integer `p` use repeated multiplication (`powi`), so results can differ from
Julia's compensated integer power by about an ulp. Complex magnitudes use
Julia's `abs`. Empty inputs give zero. This is not `opnorm`: `norm(A, 2)` is
the Frobenius norm.
"""
LinearAlgebra.norm(a::_ReductionView, p::Real=2) = _native_norm(a, p)

# Structure predicates read matrix views and borrowed matrices (and native
# owners for the triangular/diagonal tests) in place with early exit, instead
# of Julia's generic per-entry `getindex`.
const _StructureView = Union{FaerMatrixView,FaerBorrowedMatrix}
const _StructureMatrix = Union{FaerMatrix,FaerRow,_StructureView}
function _native_structure(a, code, k::Integer=0)
    # Offsets past the extents behave alike; clamping keeps `k` in Int64.
    band = Int64(clamp(k, -(Int64(1) << 40), Int64(1) << 40))
    lock(_matrix_lock) do
        owner, d = _view_source(a)
        GC.@preserve d return _native_call(
            RustCrate.view_structure, owner, pointer(d), Int32(code), band
        )
    end
end
"""
    isone(A), issymmetric(A), ishermitian(A)    # native matrix views, faerview matrices
    istriu(A, k=0), istril(A, k=0), isdiag(A)   # also FaerMatrix and FaerRow

Structure tests read native matrices, strided/reversed views and borrowed Julia
matrices in place and stop at the first violating entry. Entries compare with
`==` as in LinearAlgebra: `-0.0` counts as zero, NaN fails symmetry and
`ishermitian` requires a real diagonal; non-square matrices are neither `isone`
nor symmetric.
"""
Base.isone(a::_StructureView) = _native_structure(a, 0)
LinearAlgebra.issymmetric(a::_StructureView) = _native_structure(a, 1)
LinearAlgebra.ishermitian(a::_StructureView) = _native_structure(a, 2)
LinearAlgebra.istriu(a::_StructureMatrix, k::Integer=0) = _native_structure(a, 3, k)
LinearAlgebra.istril(a::_StructureMatrix, k::Integer=0) = _native_structure(a, 4, k)
LinearAlgebra.isdiag(a::_StructureMatrix) = _native_structure(a, 5)

function _native_dot(x, y)
    length(x) == length(y) || throw(
        DimensionMismatch(
            "dot product arguments have lengths $(length(x)) and $(length(y))"
        ),
    )
    P = promote_type(eltype(x), eltype(y))
    lock(_matrix_lock) do
        value = Ref{P}()
        ox, dx = _view_source(x)
        oy, dy = _view_source(y)
        GC.@preserve value dx dy _native_call(
            RustCrate.view_dot,
            ox,
            pointer(dx),
            oy,
            pointer(dy),
            Ptr{UInt8}(Base.unsafe_convert(Ptr{P}, value)),
            _tag(P),
        )
        return value[]
    end
end
"""
    dot(x, y)

For native vectors, rows, matrices, views and borrowed Julia arrays
(`faerview`) in any combination, compute `sum(conj(x[i]) * y[i])` with faer's
`inner_prod`, reading strided/reversed views in place. Vector-shaped operands
pair entries in order, as do equal-length operands of different shapes (in
column-major order, as Julia does; columns are read in place). Mixed scalar types promote; Rust stages
one converted copy of each promoted operand.
"""
LinearAlgebra.dot(x::_ReductionView, y::_ReductionArray) = _native_dot(x, y)
LinearAlgebra.dot(x::_ReductionOwner, y::_ReductionView) = _native_dot(x, y)
LinearAlgebra.dot(x::_ReductionOwner, y::_ReductionOwner) = _native_dot(x, y)

# `transpose(x) * y` of complex vector views and borrowed vectors: the
# conjugation flag of `adjoint(x)`'s descriptor cancels `dot`'s conjugation of
# `x`. Real `x` need no conjugation (`x' * y` and real `transpose(x) * y`
# already reach `dot`; `FaerVector` pairs are in collections.jl).
const _DotVectorView = Union{FaerVectorView,FaerBorrowedVector}
const _DotVector = Union{_DotVectorView,FaerVector}
_transpose_dot(x, y) = _native_dot(adjoint(parent(x)), y)
function Base.:*(x::LinearAlgebra.Transpose{<:Complex,<:_DotVectorView}, y::_DotVector)
    return _transpose_dot(x, y)
end
function Base.:*(x::LinearAlgebra.Transpose{<:Complex,<:FaerVector}, y::_DotVectorView)
    return _transpose_dot(x, y)
end
const _ComplexDotView = Union{FaerVectorView{<:Complex},FaerBorrowedVector{<:Complex}}
function Base.:*(
    x::LinearAlgebra.Transpose{<:Real,<:_DotVectorView},
    y::Union{_ComplexDotView,FaerVector{<:Complex}},
)
    return LinearAlgebra.dot(parent(x), y)
end
function Base.:*(x::LinearAlgebra.Transpose{<:Real,<:FaerVector}, y::_ComplexDotView)
    return LinearAlgebra.dot(parent(x), y)
end
