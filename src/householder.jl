"""
    FaerHouseholder(x::AbstractVector)
    householder(x)

Own faer's elementary Hermitian reflection `H = I - v*v'/tau` such that
`H*x ≈ [H.beta; zeros(length(x)-1)]`. This tau is faer's real denominator,
not LAPACK's multiplier. `H.v` returns an independent native vector with first
entry one. `H.beta`, `H.tau`, `H.input_norm` and `H.head_with_beta_inv` report
native generation metadata. A zero tail gives the identity (`tau=Inf`).

Four native scalars are preserved; other real/complex inputs convert to
Float64/ComplexF64. Inputs must be finite. Construction copies the input and
scales it before calling faer. Empty input produces a 0-by-0 identity.
Native and Julia strided arrays support `*`, `lmul!`, `rmul!` and left solve.
Three/five-argument `mul!` stages the result, supports aliases and alpha/beta,
and completes scalar conversion before writing the destination.
In-place Julia arrays must use one of the four native scalar types.
Application to arrays does not form H as a dense matrix; `Matrix(H)` explicitly
does. Composing two reflections returns a dense native product.
Views stage an independent native result before copying back. In-place calls
convert the reflection's stored coefficients/metadata to the destination type
before mutation. Adjoint/copy/conjugate/transpose return independent owners.
"""
mutable struct FaerHouseholder{T,H} <: AbstractMatrix{T}
    handle::Union{Nothing,H}
    n::Int
end
function _handle(h::FaerHouseholder)
    p=getfield(h, :handle)
    isnothing(p) && throw(ArgumentError("Householder has been closed"))
    return p
end
_native_argument(h::FaerHouseholder) = _handle(h)
_is_native_owner(::FaerHouseholder) = true
function _householder_result(::Type{T}, n, f, args...) where {T<:_NativeScalar}
    lock(_matrix_lock) do
        p=RustCrate.OwnedHouseholder()
        _native_call(f, p, args...)
        return FaerHouseholder{T,typeof(p)}(p, n)
    end
end
_reflector_type(::Type{T}) where {T<:_NativeScalar} = T
_reflector_type(::Type{<:Real}) = Float64
_reflector_type(::Type{<:Complex}) = ComplexF64
function FaerHouseholder{T}(x::AbstractVector) where {T<:_NativeScalar}
    lock(_matrix_lock) do
        x isa _BorrowVector && _rotation_open(x)
        data=Vector{T}(x)
        all(isfinite, data) || throw(ArgumentError("Householder input must be finite"))
        return GC.@preserve data _householder_result(
            T,
            length(data),
            RustCrate.householder_load,
            Ptr{UInt8}(pointer(data)),
            UInt(length(data)),
            _tag(T),
        )
    end
end
FaerHouseholder(x::AbstractVector{T}) where {T} = FaerHouseholder{_reflector_type(T)}(x)
"""Create a Rust-owned elementary reflection; see [`FaerHouseholder`](@ref) for its denominator tau convention and application API."""
householder(x::AbstractVector) = FaerHouseholder(x)
Base.size(h::FaerHouseholder) = (getfield(h, :n), getfield(h, :n))
Base.size(h::FaerHouseholder, d::Integer) =
    d<1 ? throw(ArgumentError("dimension must be positive")) : d<=2 ? getfield(h, :n) : 1
Base.isopen(h::FaerHouseholder) = lock(()->!isnothing(getfield(h, :handle)), _matrix_lock)
function Base.close(h::FaerHouseholder)
    lock(_matrix_lock) do
        h.handle=nothing
    end
    return nothing
end
Base.copy(h::FaerHouseholder) =
    _householder_result(eltype(h), size(h, 1), RustCrate.householder_clone, h, false)
Base.deepcopy_internal(h::FaerHouseholder, d::IdDict) = get!(() -> copy(h), d, h)
Base.adjoint(h::FaerHouseholder) = copy(h)
Base.inv(h::FaerHouseholder) = copy(h)
Base.conj(h::FaerHouseholder) =
    _householder_result(eltype(h), size(h, 1), RustCrate.householder_clone, h, true)
Base.transpose(h::FaerHouseholder) = conj(h)
function _householder_info(h::FaerHouseholder{T}) where {T}
    out=Vector{T}(undef, 4)
    GC.@preserve out _native_call(RustCrate.householder_info, h, Ptr{UInt8}(pointer(out)))
    return out
end
function Base.getproperty(h::FaerHouseholder, name::Symbol)
    name===:v && return _collection_result(
        FaerVector,
        eltype(h),
        size(h, 1),
        RustCrate.householder_vector_from,
        h,
    )
    name in (:beta, :tau, :input_norm, :head_with_beta_inv) || return getfield(h, name)
    info=_householder_info(h)
    return name===:beta ? info[1] :
           name===:tau ? real(info[2]) : name===:input_norm ? real(info[3]) : info[4]
end
Base.propertynames(h::FaerHouseholder, private::Bool = false) =
    private ?
    (fieldnames(typeof(h))..., :v, :beta, :tau, :input_norm, :head_with_beta_inv) :
    (:v, :beta, :tau, :input_norm, :head_with_beta_inv)
function Base.getindex(h::FaerHouseholder{T}, i::Int, j::Int) where {T}
    checkbounds(h, i, j)
    out=Ref{T}()
    GC.@preserve out _native_call(
        RustCrate.householder_element,
        h,
        UInt(i-1),
        UInt(j-1),
        Ptr{UInt8}(Base.unsafe_convert(Ptr{T}, out)),
    )
    return out[]
end
LinearAlgebra.ishermitian(h::FaerHouseholder) = lock(_matrix_lock) do
    _handle(h)
    return true
end
function _householder_coerce(h::FaerHouseholder{T}, ::Type{T}) where {T}
    _handle(h)
    return h
end
function _householder_coerce(h::FaerHouseholder, ::Type{T}) where {T<:_NativeScalar}
    basis=Vector{T}(h.v)
    info=Vector{T}(_householder_info(h))
    return GC.@preserve basis info _householder_result(
        T,
        size(h, 1),
        RustCrate.householder_parts,
        Ptr{UInt8}(pointer(basis)),
        UInt(length(basis)),
        Ptr{UInt8}(pointer(info)),
        _tag(T),
    )
end
const _HouseholderMatrix=Union{_RotationMatrix,StridedMatrix{<:Number}}
const _HouseholderVector=Union{_BorrowVector,StridedVector{<:Number}}
const _HouseholderArray=Union{_HouseholderMatrix,_HouseholderVector}
function _householder_check(h, a, right)
    _handle(h)
    a isa _RotationArray && _rotation_open(a)
    size(a, right ? 2 : 1)==size(h, 1) ||
        throw(DimensionMismatch("Householder application dimensions must match"))
    return nothing
end
function _householder_apply!(
    h::FaerHouseholder,
    a::Union{FaerMatrix,FaerVector,FaerRow},
    right,
)
    lock(_matrix_lock) do
        _householder_check(h, a, right)
        reflected=_householder_coerce(h, eltype(a))
        _native_call(RustCrate.householder_apply, a, reflected, right)
        return a
    end
end
function _householder_apply!(h::FaerHouseholder, a::FaerView, right)
    lock(_matrix_lock) do
        _householder_check(h, a, right)
        result=copy(a)
        _householder_apply!(h, result, right)
        return copyto!(a, result)
    end
end
function _householder_apply!(
    h::FaerHouseholder,
    a::Union{StridedVector,StridedMatrix},
    right,
)
    lock(_matrix_lock) do
        _householder_check(h, a, right)
        eltype(a)<:_NativeScalar || throw(
            ArgumentError(
                "in-place Householder arrays require Float32, Float64, ComplexF32 or ComplexF64",
            ),
        )
        result=a isa AbstractVector ? FaerVector(a) : FaerMatrix(a)
        _householder_apply!(h, result, right)
        copyto!(a, result isa FaerVector ? Vector(result) : Matrix(result))
        return a
    end
end
LinearAlgebra.lmul!(h::FaerHouseholder, a::_HouseholderArray) =
    _householder_apply!(h, a, false)
LinearAlgebra.rmul!(a::_HouseholderMatrix, h::FaerHouseholder) =
    _householder_apply!(h, a, true)
function _householder_product(h, a, right)
    lock(_matrix_lock) do
        _householder_check(h, a, right)
        P=promote_type(eltype(h), _reflector_type(eltype(a)))
        result=a isa _RotationArray ? _rotation_copy(a, P) :
               a isa AbstractVector ? FaerVector{P}(a) : FaerMatrix{P}(a)
        _householder_apply!(_householder_coerce(h, P), result, right)
        return a isa StridedVector ? Vector(result) : result
    end
end
Base.:*(h::FaerHouseholder, a::_HouseholderArray) = _householder_product(h, a, false)
Base.:*(a::_HouseholderMatrix, h::FaerHouseholder) = _householder_product(h, a, true)
function Base.:*(a::FaerHouseholder, b::FaerHouseholder)
    lock(_matrix_lock) do
        _handle(a)
        _handle(b)
        size(a)==size(b) || throw(DimensionMismatch("reflection dimensions must match"))
        return a*FaerMatrix(b)
    end
end
Base.:\(h::FaerHouseholder, a::_HouseholderArray) = h*a
function FaerMatrix(h::FaerHouseholder{T}) where {T}
    lock(_matrix_lock) do
        _handle(h)
        result=FaerMatrix{T}(Matrix{T}(LinearAlgebra.I, size(h)...))
        return _householder_apply!(h, result, false)
    end
end
Base.Matrix(h::FaerHouseholder) = Matrix(FaerMatrix(h))
Base.Matrix{T}(h::FaerHouseholder) where {T} = Matrix{T}(Matrix(h))

function _householder_matrix_source(a, ::Type{T}) where {T}
    a isa _BorrowMatrix && return _coerce(copy(a), T)
    a isa _BorrowVector && return _coerce(_copy_view_matrix(a), T)
    return a isa AbstractVector ? FaerMatrix{T}(reshape(a, :, 1)) : FaerMatrix{T}(a)
end
function _householder_mul!(dest, h, a, alpha, beta, right)
    lock(_matrix_lock) do
        _householder_check(h, a, right)
        _householder_check(h, dest, right)
        size(dest)==size(a) || throw(
            DimensionMismatch(
                "Householder product destination must match the source shape",
            ),
        )
        T=eltype(dest)
        T<:_NativeScalar || throw(
            ArgumentError("Householder mul! destinations require a native scalar type"),
        )
        P=promote_type(
            eltype(h),
            _reflector_type(eltype(a)),
            T,
            typeof(alpha),
            typeof(beta),
        )
        P<:_NativeScalar ||
            throw(ArgumentError("unsupported Householder product scalar type"))
        dims=ndims(a)==1 ? (length(a), 1) : size(a)
        result=if iszero(alpha)
            _zeros(P, dims)
        else
            work=_householder_matrix_source(a, P)
            alpha*_householder_apply!(_householder_coerce(h, P), work, right)
        end
        !iszero(beta) && (result=result+beta*_householder_matrix_source(dest, P))
        # Convert the complete result before any destination writes, including
        # real destinations that cannot represent a complex product.
        buffer=ndims(dest)==1 ? vec(Matrix{T}(result)) : Matrix{T}(result)
        copyto!(dest, buffer)
        return dest
    end
end
LinearAlgebra.mul!(
    dest::_HouseholderMatrix,
    h::FaerHouseholder,
    a::_HouseholderMatrix,
    alpha::Number,
    beta::Number,
) = _householder_mul!(dest, h, a, alpha, beta, false)
LinearAlgebra.mul!(
    dest::_HouseholderVector,
    h::FaerHouseholder,
    a::_HouseholderVector,
    alpha::Number,
    beta::Number,
) = _householder_mul!(dest, h, a, alpha, beta, false)
LinearAlgebra.mul!(
    dest::_HouseholderMatrix,
    a::_HouseholderMatrix,
    h::FaerHouseholder,
    alpha::Number,
    beta::Number,
) = _householder_mul!(dest, h, a, alpha, beta, true)
LinearAlgebra.mul!(dest::_HouseholderMatrix, h::FaerHouseholder, a::_HouseholderMatrix) =
    _householder_mul!(dest, h, a, 1, 0, false)
LinearAlgebra.mul!(dest::_HouseholderVector, h::FaerHouseholder, a::_HouseholderVector) =
    _householder_mul!(dest, h, a, 1, 0, false)
LinearAlgebra.mul!(dest::_HouseholderMatrix, a::_HouseholderMatrix, h::FaerHouseholder) =
    _householder_mul!(dest, h, a, 1, 0, true)
