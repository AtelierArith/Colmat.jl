"""
    FaerRNG(seed=rand(UInt64))

An independently owned Rust StdRng stream, with Julia's AbstractRNG interface.
`seed!`, `copy`, `rand(rng,UInt64)` and native array sampling share its state.
Native rand/randn use faer's distributions; standard Julia arrays/scalars use
Julia's sampling algorithms over native 64-bit words (narrower/128-bit integers,
Float16/Float32 and complex draws included). Reproducibility requires
the same native dependency versions and call sequence, not a Julia RNG match.
`close` invalidates the owner; copied RNGs remain independent.
"""
mutable struct FaerRNG{H} <: Random.AbstractRNG
    handle::Union{Nothing,H}
end
function _handle(rng::FaerRNG)
    h=rng.handle
    isnothing(h) && throw(ArgumentError("FaerRNG has been closed"))
    return h
end
_native_argument(rng::FaerRNG) = _handle(rng)
_is_native_owner(::FaerRNG) = true
function _random_seed(seed::Integer)
    0<=seed<=typemax(UInt64) || throw(ArgumentError("native RNG seed must fit UInt64"))
    return UInt64(seed)
end
function FaerRNG(seed::Integer = Random.rand(Random.default_rng(), UInt64))
    seed=_random_seed(seed)
    lock(_matrix_lock) do
        h=RustCrate.OwnedRandom()
        rng=FaerRNG{typeof(h)}(h)
        _native_call(RustCrate.rng_seed, rng, seed)
        return rng
    end
end
Base.isopen(rng::FaerRNG) = lock(()->!isnothing(rng.handle), _matrix_lock)
function Base.close(rng::FaerRNG)
    lock(_matrix_lock) do ;
        rng.handle=nothing
    end
    return nothing
end
function Base.copy(rng::FaerRNG)
    lock(_matrix_lock) do
        h=_native_call(RustCrate.duplicate, rng)
        return FaerRNG{typeof(h)}(h)
    end
end
Base.deepcopy_internal(rng::FaerRNG, d::IdDict) = get!(() -> copy(rng), d, rng)
function Random.seed!(rng::FaerRNG, seed::Integer)
    lock(_matrix_lock) do
        _native_call(RustCrate.rng_seed, rng, _random_seed(seed))
        return rng
    end
end
function Random.seed!(rng::FaerRNG)
    lock(_matrix_lock) do
        _handle(rng)
        return Random.seed!(rng, Random.rand(Random.default_rng(), UInt64))
    end
end
Random.rng_native_52(::FaerRNG) = UInt64
Random.rand(rng::FaerRNG, ::Random.SamplerType{UInt64}) =
    _native_call(RustCrate.rng_next, rng)
Random.rand(
    rng::FaerRNG,
    ::Random.SamplerType{T},
) where {T<:Union{Bool,Int8,UInt8,Int16,UInt16,Int32,UInt32,Int64}} =
    Random.rand(rng, UInt64) % T
function Random.rand(rng::FaerRNG, ::Random.SamplerType{T}) where {T<:Union{Int128,UInt128}}
    hi=Random.rand(rng, UInt64)
    return ((UInt128(hi) << 64) | Random.rand(rng, UInt64)) % T
end
_random_owner(rng::FaerRNG, nonempty) = (_handle(rng); rng)
_random_owner(rng::Random.AbstractRNG, nonempty) =
    FaerRNG(nonempty ? Random.rand(rng, UInt64) : UInt64(0))
function _native_random(
    rng::Random.AbstractRNG,
    ::Type{T},
    m::Integer,
    n::Integer,
    kind,
    normal;
    unitary = false,
) where {T<:_NativeScalar}
    lock(_matrix_lock) do
        m, n=Int(m), Int(n)
        m>=0&&n>=0 || throw(ArgumentError("random dimensions must be nonnegative"))
        Base.checked_mul(Base.checked_mul(m, n), sizeof(T))
        source=_random_owner(rng, m>0&&n>0)
        if kind==0
            args=(source, UInt(m), UInt(n), normal, unitary)
            return T===Float64 ? _matrix_result(RustCrate.random_from, (m, n), args...) :
                   _typed_result(T, RustCrate.random_from, (m, n), args..., _tag(T))
        end
        C=kind==1 ? FaerVector : FaerRow
        len=kind==1 ? m : n
        return _collection_result(
            C,
            T,
            len,
            RustCrate.random_from,
            source,
            UInt(m),
            UInt(n),
            normal,
            Int32(kind),
            _tag(T),
        )
    end
end
for (fn, normal) in ((:rand, false), (:randn, true))
    @eval begin
        Random.$fn(
            rng::Random.AbstractRNG,
            ::Type{<:FaerMatrix{T}},
            m::Integer,
            n::Integer,
        ) where {T<:_NativeScalar} = _native_random(rng, T, m, n, 0, $normal)
        Random.$fn(
            rng::Random.AbstractRNG,
            ::Type{<:FaerVector{T}},
            n::Integer,
        ) where {T<:_NativeScalar} = _native_random(rng, T, n, 1, 1, $normal)
        Random.$fn(
            rng::Random.AbstractRNG,
            ::Type{<:FaerRow{T}},
            n::Integer,
        ) where {T<:_NativeScalar} = _native_random(rng, T, 1, n, 2, $normal)
        Random.$fn(
            ::Type{C},
            dims::Integer...,
        ) where {T<:_NativeScalar,C<:Union{FaerMatrix{T},FaerVector{T},FaerRow{T}}} =
            Random.$fn(Random.default_rng(), C, dims...)
        Random.$fn(
            rng::Random.AbstractRNG,
            ::Type{<:FaerMatrix{T}},
            dims::Dims{2},
        ) where {T<:_NativeScalar} = Random.$fn(rng, FaerMatrix{T}, dims...)
        Random.$fn(
            rng::Random.AbstractRNG,
            ::Type{<:FaerVector{T}},
            dims::Dims{1},
        ) where {T<:_NativeScalar} = Random.$fn(rng, FaerVector{T}, dims...)
        Random.$fn(
            rng::Random.AbstractRNG,
            ::Type{<:FaerRow{T}},
            dims::Dims{1},
        ) where {T<:_NativeScalar} = Random.$fn(rng, FaerRow{T}, dims...)
        Random.$fn(
            ::Type{C},
            dims::Dims,
        ) where {T<:_NativeScalar,C<:Union{FaerMatrix{T},FaerVector{T},FaerRow{T}}} =
            Random.$fn(Random.default_rng(), C, dims)
        Random.$fn(rng::Random.AbstractRNG, ::Type{FaerMatrix}, m::Integer, n::Integer) =
            Random.$fn(rng, FaerMatrix{Float64}, m, n)
        Random.$fn(rng::Random.AbstractRNG, ::Type{FaerVector}, n::Integer) =
            Random.$fn(rng, FaerVector{Float64}, n)
        Random.$fn(rng::Random.AbstractRNG, ::Type{FaerRow}, n::Integer) =
            Random.$fn(rng, FaerRow{Float64}, n)
        Random.$fn(rng::Random.AbstractRNG, ::Type{FaerMatrix}, dims::Dims{2}) =
            Random.$fn(rng, FaerMatrix{Float64}, dims...)
        Random.$fn(rng::Random.AbstractRNG, ::Type{FaerVector}, dims::Dims{1}) =
            Random.$fn(rng, FaerVector{Float64}, dims...)
        Random.$fn(rng::Random.AbstractRNG, ::Type{FaerRow}, dims::Dims{1}) =
            Random.$fn(rng, FaerRow{Float64}, dims...)
        Random.$fn(::Type{FaerMatrix}, dims::Dims{2}) =
            Random.$fn(Random.default_rng(), FaerMatrix{Float64}, dims...)
        Random.$fn(::Type{FaerVector}, dims::Dims{1}) =
            Random.$fn(Random.default_rng(), FaerVector{Float64}, dims...)
        Random.$fn(::Type{FaerRow}, dims::Dims{1}) =
            Random.$fn(Random.default_rng(), FaerRow{Float64}, dims...)
        Random.$fn(::Type{FaerMatrix}, m::Integer, n::Integer) =
            Random.$fn(Random.default_rng(), FaerMatrix{Float64}, m, n)
        Random.$fn(::Type{FaerVector}, n::Integer) =
            Random.$fn(Random.default_rng(), FaerVector{Float64}, n)
        Random.$fn(::Type{FaerRow}, n::Integer) =
            Random.$fn(Random.default_rng(), FaerRow{Float64}, n)
    end
end
# Whole-array views take the bulk native assignment path for owner targets.
function _random_assign!(a, staged)
    copyto!(a isa FaerView ? a : view(a, axes(a)...), staged)
    return a
end
function _native_random!(rng, a, normal)
    lock(_matrix_lock) do
        _rotation_open(a)
        kind=a isa AbstractVector ? 1 : a isa FaerRow ? 2 : 0
        result=_native_random(rng, eltype(a), size(a, 1), size(a, 2), kind, normal)
        return _random_assign!(a, result)
    end
end
"""Fill a native array/view from faer's uniform distribution, staging output before mutation."""
Random.rand!(rng::Random.AbstractRNG, a::_RotationArray) = _native_random!(rng, a, false)
Random.rand!(a::_RotationArray) = Random.rand!(Random.default_rng(), a)
function _julia_sampler!(rng, a, X)
    lock(_matrix_lock) do
        _rotation_open(a)
        staged=Random.rand!(rng, Array{eltype(a)}(undef, size(a)), X)
        return _random_assign!(a, staged)
    end
end
"""
    rand!(rng=default_rng(), A, X)

Fill a native matrix, vector, row or view with Julia samples from `X`: a
collection, range, scalar type, `Random.Sampler`, or any value Julia's
`rand(rng, X)` accepts (for example a Distributions.jl distribution). Samples
come from Julia's `rand!` on an `Array` of the same eltype and shape, so `rng` is
consumed exactly as for that array; a `FaerRNG` serves those draws from its
native stream. Unlike `rand!(rng, A)`, this path uses Julia's algorithms, not
faer's distributions. Target validation precedes sampling, and sampling and
conversion finish before one bulk native update, so failures leave `A` unchanged.
"""
Random.rand!(rng::Random.AbstractRNG, a::_RotationArray, sp::Random.Sampler) =
    _julia_sampler!(rng, a, sp)
# Resolves the intersection with MersenneTwister's bulk Float64 array method.
Random.rand!(
    rng::Random.MersenneTwister,
    a::typeintersect(_RotationArray, AbstractArray{Float64}),
    sp::Random.SamplerTrivial{<:Random.FloatInterval{Float64}},
) = _julia_sampler!(rng, a, sp)
"""Fill a native array/view with standard normal values; complex components have variance 1/2."""
Random.randn!(rng::Random.AbstractRNG, a::_RotationArray) = _native_random!(rng, a, true)
Random.randn!(a::_RotationArray) = Random.randn!(Random.default_rng(), a)
"""
    randunitary(rng, T, n)
    randunitary(T, n)
    randunitary(n)

Sample a native n×n orthogonal (real T) or unitary (complex T) matrix through
faer's UnitaryMat distribution, using sequential QR and diagonal phase correction.
T is one of the four native scalar types and defaults to Float64. Native array
sampling uses FaerRNG directly; another Julia RNG supplies one UInt64 seed per
nonempty call (empty calls do not consume a seed). rand/randn on FaerMatrix,
FaerVector and FaerRow follow the same rule. Complex normal components have
variance 1/2, giving unit expected squared modulus as in Julia.
"""
randunitary(rng::Random.AbstractRNG, ::Type{T}, n::Integer) where {T<:_NativeScalar} =
    _native_random(rng, T, n, n, 0, true; unitary = true)
randunitary(::Type{T}, n::Integer) where {T<:_NativeScalar} =
    randunitary(Random.default_rng(), T, n)
randunitary(rng::Random.AbstractRNG, n::Integer) = randunitary(rng, Float64, n)
randunitary(n::Integer) = randunitary(Random.default_rng(), Float64, n)
