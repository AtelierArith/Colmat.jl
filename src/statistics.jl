# Native mean/variance accept NaNs, unlike constructors used for factorizations.
# Snapshot to exact primitive buffers without re-entering a finite-only loader.
function _statistics_layout(a, dims)
    _rotation_open(a)
    data=a isa AbstractVector ? Vector(a) : Matrix(a)
    scalar=dims isa Colon
    axes=scalar ? ntuple(identity, ndims(a)) : dims isa Integer ? (dims,) : dims
    axes isa Tuple && all(d->d isa Integer && d>0, axes) ||
        throw(ArgumentError("statistics dims must be positive integers"))
    shape=ntuple(d->d in axes ? 1 : size(a, d), ndims(a))
    if all(d->d in axes, 1:ndims(a))
        return reshape(data, length(data), 1), 1, shape, scalar
    elseif ndims(a)==2 && 1 in axes
        return data, 1, shape, scalar
    elseif ndims(a)==2 && 2 in axes
        return data, 2, shape, scalar
    end
    return reshape(data, 1, length(data)), 1, shape, scalar
end
function _statistics_policy(nan)
    nan in (:propagate, :ignore) ||
        throw(ArgumentError("nan must be :propagate or :ignore"))
    return nan===:ignore
end
function _statistics_native(
    data::AbstractMatrix{T},
    axis,
    ignore,
    means = nothing,
) where {T}
    variance=!isnothing(means)
    output=zeros(variance ? _realtype(T) : T, size(data, axis==1 ? 2 : 1))
    meanptr=variance ? Ptr{UInt8}(pointer(means)) : Ptr{UInt8}(C_NULL)
    GC.@preserve data output means _native_call(
        RustCrate.statistics_reduce,
        Ptr{UInt8}(pointer(data)),
        UInt(size(data, 1)),
        UInt(size(data, 2)),
        Ptr{UInt8}(pointer(output)),
        meanptr,
        _tag(T),
        Int32(axis),
        variance,
        ignore,
    )
    return output
end
_statistics_output(result, shape, scalar) = scalar ? only(result) : reshape(result, shape)
"""
    mean(A::Union{FaerMatrix,FaerVector,FaerRow,FaerView}; dims=:, nan=:propagate)

Compute native faer row/column means. `dims` and return shapes follow Statistics;
complex inputs retain their scalar type. `nan=:ignore` excludes entries with a
NaN component; default `:propagate` keeps Julia's arithmetic behavior. Empty
or entirely excluded slices give NaN. Native arrays/views are snapshotted to
primitive buffers; outputs are ordinary Julia arrays or a scalar.
"""
function Statistics.mean(a::_RotationArray; dims = :, nan::Symbol = :propagate)
    lock(_matrix_lock) do
        ignore=_statistics_policy(nan)
        data, axis, shape, scalar=_statistics_layout(a, dims)
        return _statistics_output(_statistics_native(data, axis, ignore), shape, scalar)
    end
end
function _statistics_variance(a, m, dims, corrected, nan)
    lock(_matrix_lock) do
        ignore=_statistics_policy(nan)
        m isa _RotationArray && _rotation_open(m)
        data, axis, shape, scalar=_statistics_layout(a, dims)
        T=isnothing(m) ? eltype(a) :
          promote_type(eltype(a), _reflector_type(m isa Number ? typeof(m) : eltype(m)))
        eltype(data)===T || (data=Matrix{T}(data))
        means=if isnothing(m)
            _statistics_native(data, axis, ignore)
        elseif scalar
            m isa Number ||
                throw(DimensionMismatch("full-array variance requires a scalar mean"))
            T[m]
        else
            m isa AbstractArray && all(
                d->size(m, d)==(d<=length(shape) ? shape[d] : 1),
                1:max(ndims(m), length(shape)),
            ) || throw(DimensionMismatch("mean shape must match the reduced output shape"))
            vec(Array{T}(m))
        end
        result=_statistics_native(data, axis, ignore, means)
        R=_realtype(T)
        # Native varm divides by N-1 and returns 0 for singleton slices.
        # Match Statistics and its supplied-mean arithmetic at N=0/1.
        for i in eachindex(result)
            slice=axis==1 ? view(data, :, i) : view(data, i, :)
            count=ignore ? Base.count(x->!isnan(x), slice) : length(slice)
            if count==0
                result[i]=R(NaN)
            elseif count==1
                x=ignore ? first(Iterators.filter(x->!isnan(x), slice)) : only(slice)
                result[i]=abs2(x-means[i])/R(corrected ? 0 : 1)
            elseif !corrected
                result[i]*=R(count-1)/R(count)
            end
        end
        return _statistics_output(result, shape, scalar)
    end
end
"""
    var(A::Union{FaerMatrix,FaerVector,FaerRow,FaerView}; dims=:,
        corrected=true, mean=nothing, nan=:propagate)

Native faer variance with Julia's sample/population correction and real output
for complex inputs. `nan=:ignore` counts only included entries per slice.
Means and dimension shapes follow [`Statistics.mean`](@ref); an explicitly
supplied mean must be scalar for a full reduction, otherwise match output shape
(with optional trailing singleton dimensions). Supplied means promote the native
working scalar type; variance uses its real type.
"""
Statistics.var(
    a::_RotationArray;
    dims = :,
    corrected::Bool = true,
    mean = nothing,
    nan::Symbol = :propagate,
) = _statistics_variance(a, mean, dims, corrected, nan)
"""Native faer variance about a supplied mean; see [`Statistics.var`](@ref)."""
Statistics.varm(
    a::_RotationArray,
    m::Number;
    dims = :,
    corrected::Bool = true,
    nan::Symbol = :propagate,
) = _statistics_variance(a, m, dims, corrected, nan)

Statistics.varm(
    a::_RotationArray,
    m::AbstractArray;
    dims = :,
    corrected::Bool = true,
    nan::Symbol = :propagate,
) = _statistics_variance(a, m, dims, corrected, nan)
