# Native mean/variance accept NaNs, unlike constructors used for factorizations.
# The kernels borrow the owner storage (or view) directly; no Julia snapshot.
# Julia arrays borrowed with faerview are read in place through the same kernels.
const _StatisticsArray = Union{_RotationArray,FaerBorrowedArray}
_statistics_open(a) = a isa FaerBorrowedArray ? nothing : _rotation_open(a)
# Reduction axes of the borrowed m x n matrix (vectors borrow as n x 1).
const _STATS_NONE = Int32(0)
const _STATS_ROWS = Int32(1)
const _STATS_COLS = Int32(2)
const _STATS_ALL = Int32(3)
function _statistics_layout(a, dims)
    scalar = dims isa Colon
    axes = if scalar
        ntuple(identity, ndims(a))
    elseif dims isa Integer
        (dims,)
    else
        dims
    end
    axes isa Tuple && all(d -> d isa Integer && d > 0, axes) ||
        throw(ArgumentError("statistics dims must be positive integers"))
    shape = ntuple(d -> d in axes ? 1 : size(a, d), ndims(a))
    m, n = ndims(a) == 1 ? (length(a), 1) : size(a)
    axis = if all(d -> d in axes, 1:ndims(a))
        _STATS_ALL
    elseif ndims(a) == 2 && 1 in axes
        _STATS_ROWS
    elseif ndims(a) == 2 && 2 in axes
        _STATS_COLS
    else
        _STATS_NONE
    end
    len = if axis == _STATS_ALL
        1
    elseif axis == _STATS_ROWS
        n
    elseif axis == _STATS_COLS
        m
    else
        m * n
    end
    return axis, len, shape, scalar
end
function _statistics_policy(nan)
    nan in (:propagate, :ignore) ||
        throw(ArgumentError("nan must be :propagate or :ignore"))
    return nan === :ignore
end
# W is the working scalar; means, when supplied, hold W values.
function _statistics_native(a, ::Type{W}, axis, len, ignore, corrected, means) where {W}
    variance = !isnothing(corrected)
    output = Vector{variance ? _realtype(W) : W}(undef, len)
    p, d = _view_source(a)
    meanptr = isnothing(means) ? Ptr{UInt8}(C_NULL) : Ptr{UInt8}(pointer(means))
    GC.@preserve d output means _native_call(
        RustCrate.statistics_view_reduce,
        p,
        pointer(d),
        Ptr{UInt8}(pointer(output)),
        UInt(len),
        meanptr,
        _tag(W),
        axis,
        variance,
        something(corrected, true),
        ignore,
    )
    return output
end
_statistics_output(result, shape, scalar) = scalar ? only(result) : reshape(result, shape)
"""
    mean(A::Union{FaerMatrix,FaerVector,FaerRow,FaerView,FaerBorrowedArray}; dims=:, nan=:propagate)

Compute native faer row/column means. `dims` and return shapes follow Statistics;
complex inputs retain their scalar type. `nan=:ignore` excludes entries with a
NaN component; default `:propagate` keeps Julia's arithmetic behavior. Empty
or entirely excluded slices give NaN. faer's kernels read native arrays and
strided/reversed views in place (a full reduction of a view whose entries are
not one strided column stages one contiguous column in Rust); outputs are
ordinary Julia arrays or a scalar.
"""
function Statistics.mean(a::_StatisticsArray; dims=:, nan::Symbol=:propagate)
    lock(_matrix_lock) do
        ignore = _statistics_policy(nan)
        _statistics_open(a)
        axis, len, shape, scalar = _statistics_layout(a, dims)
        T = eltype(a)
        result = _statistics_native(a, T, axis, len, ignore, nothing, nothing)
        return _statistics_output(result, shape, scalar)
    end
end
# Statistics.varm promotes each `x - mean` difference: full reductions return
# the promoted real type (the data's for empty input), dims reductions store
# into the data's real type. Promotion beyond the data scalar stages a
# converted copy in Rust because faer's kernels take one scalar type.
function _statistics_variance(a, m, dims, corrected, nan)
    lock(_matrix_lock) do
        ignore = _statistics_policy(nan)
        _statistics_open(a)
        m isa _RotationArray && _rotation_open(m)
        axis, len, shape, scalar = _statistics_layout(a, dims)
        S = eltype(a)
        W = promote_type(
            S,
            if isnothing(m)
                S
            elseif m isa Number
                typeof(m)
            else
                eltype(m)
            end,
        )
        W <: _NativeScalar || throw(
            ArgumentError("mean scalar type must promote with $S to a native scalar type"),
        )
        means = if isnothing(m)
            nothing
        elseif scalar
            m isa Number ||
                throw(DimensionMismatch("full-array variance requires a scalar mean"))
            W[m]
        else
            m isa AbstractArray && all(
                d -> size(m, d) == (d <= length(shape) ? shape[d] : 1),
                1:max(ndims(m), length(shape)),
            ) || throw(
                DimensionMismatch("mean shape must match the reduced output shape")
            )
            vec(Array{W}(m))
        end
        result = _statistics_native(a, W, axis, len, ignore, corrected, means)
        R = scalar && !isempty(a) ? _realtype(W) : _realtype(S)
        eltype(result) === R || (result = Vector{R}(result))
        return _statistics_output(result, shape, scalar)
    end
end
"""
    var(A::Union{FaerMatrix,FaerVector,FaerRow,FaerView,FaerBorrowedArray}; dims=:,
        corrected=true, mean=nothing, nan=:propagate)

Native faer variance with Julia's sample/population correction and real output
for complex inputs. `nan=:ignore` counts only included entries per slice.
Means and dimension shapes follow [`Statistics.mean`](@ref); an explicitly
supplied mean must be scalar for a full reduction, otherwise match output shape
(with optional trailing singleton dimensions). Result types follow
`Statistics.varm`: a full reduction returns the real type of
`promote_type(eltype(A), typeof(mean))` (the data's real type when `A` is
empty), and a `dims` reduction keeps the data's real type. A mean that promotes
the data scalar (for example a Float64 or complex mean for Float32 data) makes
Rust stage one converted copy of the data; other calls read it in place. The
promoted type must be one of the four native scalars, otherwise an
`ArgumentError` is thrown (e.g. `BigFloat` means).
"""
function Statistics.var(
    a::_StatisticsArray; dims=:, corrected::Bool=true, mean=nothing, nan::Symbol=:propagate
)
    return _statistics_variance(a, mean, dims, corrected, nan)
end
"""Native faer variance about a supplied mean; see [`Statistics.var`](@ref)."""
function Statistics.varm(
    a::_StatisticsArray, m::Number; dims=:, corrected::Bool=true, nan::Symbol=:propagate
)
    return _statistics_variance(a, m, dims, corrected, nan)
end

function Statistics.varm(
    a::_StatisticsArray,
    m::AbstractArray;
    dims=:,
    corrected::Bool=true,
    nan::Symbol=:propagate,
)
    return _statistics_variance(a, m, dims, corrected, nan)
end
