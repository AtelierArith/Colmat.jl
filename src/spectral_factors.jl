"""
    svd(A::FaerMatrix; native=true, full=false, team=nothing)
    eigen(H::Hermitian; native=true, sortby, team=nothing)

Retain an actual Rust-owned faer SVD or self-adjoint EVD. Components are
independent copies: SVD exposes native U/V, Julia real S, and native-adjoint Vt;
EVD exposes native vectors and Julia real values. `pinv(F;atol,rtol,team,scratch)` and
`reconstruct(F;team)` reuse stored factors without another decomposition.
Construction uses backend-owned workspace; caller `scratch` is unsupported
for construction. Saved-factor pseudoinverse reconstruction accepts caller scratch. Default calls retain their existing Julia factorization results.
"""
mutable struct FaerSpectralFactor{T,K,H} <: LinearAlgebra.Factorization{T}
    handle::Union{Nothing,H}
    dims::Tuple{Int,Int}
    ucols::Int
    vcols::Int
    order::Vector{Int}
end
function _handle(f::FaerSpectralFactor)
    h=getfield(f, :handle)
    isnothing(h) && throw(ArgumentError("spectral factorization has been closed"))
    return h
end
_native_argument(f::FaerSpectralFactor) = _handle(f)
_is_native_owner(::FaerSpectralFactor) = true
Base.size(f::FaerSpectralFactor) = getfield(f, :dims)
Base.size(f::FaerSpectralFactor, d::Integer) =
    d<1 ? throw(ArgumentError("dimension must be positive")) : d<=2 ? size(f)[d] : 1
Base.isopen(f::FaerSpectralFactor) =
    lock(()->!isnothing(getfield(f, :handle)), _matrix_lock)
function Base.close(f::FaerSpectralFactor)
    lock(_matrix_lock) do
        if !isnothing(getfield(f, :handle))
            _native_call(RustCrate.spectral_close, f)
            setfield!(f, :handle, nothing)
        end
    end
    return nothing
end
function Base.copy(f::FaerSpectralFactor{T,K}) where {T,K}
    lock(_matrix_lock) do
        _handle(f)
        h=RustCrate.OwnedSpectral()
        _native_call(RustCrate.spectral_clone, h, f)
        return FaerSpectralFactor{T,K,typeof(h)}(
            h,
            size(f),
            getfield(f, :ucols),
            getfield(f, :vcols),
            copy(getfield(f, :order)),
        )
    end
end
Base.deepcopy_internal(f::FaerSpectralFactor, d::IdDict) = get!(() -> copy(f), d, f)
function _spectral_values(f::FaerSpectralFactor{T,K}) where {T,K}
    lock(_matrix_lock) do
        _handle(f)
        values=zeros(_realtype(T), min(size(f)...))
        GC.@preserve values _native_call(
            RustCrate.spectral_values,
            f,
            Ptr{UInt8}(pointer(values)),
            UInt(length(values)),
            _tag(T),
            K===:eigen,
        )
        return K===:eigen ? values[getfield(f, :order)] : values
    end
end
function _spectral_part(
    f::FaerSpectralFactor{T,K},
    part;
    atol = 0,
    rtol = 0,
    team = nothing,
) where {T,K}
    lock(_matrix_lock) do
        _handle(f)
        isnothing(team) ||
            team isa FaerTeam ||
            throw(ArgumentError("team must be a FaerTeam"))
        !isnothing(team) && _handle(team)
        at, rt=_spectral_tolerances(atol, rtol)
        m, n=size(f)
        dims=part==0 ? (m, n) :
             part==1 ? (m, getfield(f, :ucols)) :
             part==2 ? (n, getfield(f, :vcols)) : (n, m)
        args=(
            f,
            Int32(part),
            UInt(dims[1]),
            UInt(dims[2]),
            _tag(T),
            K===:eigen,
            at,
            rt,
            promote_type(typeof(rtol), _realtype(T))===Float32,
        )
        fun=isnothing(team) ? RustCrate.spectral_part : RustCrate.spectral_part_team
        tail=isnothing(team) ? () : (team,)
        return T===Float64 ? _matrix_result(fun, dims, args..., tail...) :
               _typed_result(T, fun, dims, args..., tail...)
    end
end
function Base.getproperty(f::FaerSpectralFactor{T,K}, name::Symbol) where {T,K}
    if K===:svd
        name===:U && return _spectral_part(f, 1)
        name===:V && return _spectral_part(f, 2)
        name===:Vt && return adjoint(_spectral_part(f, 2))
        name===:S && return _spectral_values(f)
    else
        name===:values && return _spectral_values(f)
        if name===:vectors
            u=_spectral_part(f, 1)
            getfield(f, :order)==collect(1:size(f, 1)) && return u
            try
                return permute_cols(u, getfield(f, :order))
            finally
                close(u)
            end
        end
    end
    return getfield(f, name)
end
Base.propertynames(f::FaerSpectralFactor{T,K}, private::Bool = false) where {T,K} =
    private ?
    (fieldnames(typeof(f))..., (K===:svd ? (:U, :S, :V, :Vt) : (:values, :vectors))...) :
    (K===:svd ? (:U, :S, :V, :Vt) : (:values, :vectors))
function _owned_spectral(
    a::FaerMatrix{T};
    eigen = false,
    lower = true,
    full = false,
    sortby = nothing,
    scratch = nothing,
    team = nothing,
) where {T}
    lock(_matrix_lock) do
        _handle(a)
        isnothing(scratch) || throw(
            ArgumentError(
                "native factor construction uses backend-owned workspace; omit scratch",
            ),
        )
        isnothing(team) ||
            team isa FaerTeam ||
            throw(ArgumentError("team must be a FaerTeam"))
        !isnothing(team) && _handle(team)
        eigen && LinearAlgebra.checksquare(a)
        h=RustCrate.OwnedSpectral()
        args=(a, h, eigen, lower, full)
        isnothing(team) ? _native_call(RustCrate.spectral_factor, args...) :
        _native_call(RustCrate.spectral_factor_team, args..., team)
        m, n=size(a)
        k=min(m, n)
        f=FaerSpectralFactor{T,eigen ? :eigen : :svd,typeof(h)}(
            h,
            (m, n),
            eigen ? n : full ? m : k,
            eigen ? n : full ? n : k,
            collect(1:k),
        )
        try
            eigen &&
                !isnothing(sortby) &&
                (f.order=sortperm(_spectral_values(f); by = sortby))
            return f
        catch
            close(f)
            rethrow()
        end
    end
end
LinearAlgebra.pinv(
    f::FaerSpectralFactor;
    atol::Real = 0,
    rtol::Real = _default_spectral_rtol(f, atol),
    team = nothing,
    scratch = nothing,
) =
    isnothing(scratch) ? _spectral_part(f, 3; atol, rtol, team) :
    _spectral_pinv_with_scratch(f, scratch; atol, rtol, team)
LinearAlgebra.svdvals(f::FaerSpectralFactor{T,:svd}) where {T} = f.S
LinearAlgebra.eigvals(f::FaerSpectralFactor{T,:eigen}) where {T} = f.values
LinearAlgebra.eigvecs(f::FaerSpectralFactor{T,:eigen}) where {T} = f.vectors
LinearAlgebra.rank(
    f::FaerSpectralFactor;
    atol::Real = 0,
    rtol::Real = _default_spectral_rtol(f, atol),
) = _spectral_rank(f, abs.(_spectral_values(f)), atol, rtol)
reconstruct(f::FaerSpectralFactor; team = nothing) = _spectral_part(f, 0; team)
FaerMatrix(f::FaerSpectralFactor) = reconstruct(f)
function Base.Matrix(f::FaerSpectralFactor)
    a=FaerMatrix(f)
    try
        return Matrix(a)
    finally
        close(a)
    end
end
Base.Array(f::FaerSpectralFactor) = Matrix(f)
function Base.Matrix{T}(f::FaerSpectralFactor) where {T}
    a=FaerMatrix(f)
    try
        return Matrix{T}(a)
    finally
        close(a)
    end
end
function FaerMatrix{T}(f::FaerSpectralFactor) where {T<:_NativeScalar}
    a=FaerMatrix(f)
    T===eltype(a) && return a
    try
        return FaerMatrix{T}(a)
    finally
        close(a)
    end
end
LinearAlgebra.issuccess(f::FaerSpectralFactor) = lock(_matrix_lock) do
    _handle(f)
    true
end
Base.length(::FaerSpectralFactor{T,:svd}) where {T} = 3
Base.length(::FaerSpectralFactor{T,:eigen}) where {T} = 2
function Base.iterate(f::FaerSpectralFactor{T,:svd}, state = 1) where {T}
    state==1 && return (f.U, 2)
    state==2 && return (f.S, 3)
    state==3 && return (f.V, 4)
    return nothing
end
function Base.iterate(f::FaerSpectralFactor{T,:eigen}, state = 1) where {T}
    state==1 && return (f.values, 2)
    state==2 && return (f.vectors, 3)
    return nothing
end
