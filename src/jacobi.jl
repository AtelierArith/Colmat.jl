"""
    FaerJacobiRotation(c, s)

The unitary 2×2 faer rotation `[c -conj(s); s c]`, with real cosine c.
Coefficients must be finite, have a native scalar type and unit norm.
Multiplication and in-place application use faer's rotation kernels. Use
`LinearAlgebra.Givens(J, i, j)` to embed the same rotation in a larger array.
"""
struct FaerJacobiRotation{T<:_NativeScalar} <: AbstractMatrix{T}
    c::T
    s::T
    function FaerJacobiRotation{T}(c, s) where {T<:_NativeScalar}
        c, s=T(c), T(s)
        all(isfinite, (c, s)) && isreal(c) ||
            throw(ArgumentError("Jacobi coefficients must be finite and cosine real"))
        R=_realtype(T)
        isapprox(abs2(c)+abs2(s), one(R); rtol = 128eps(R), atol = 128eps(R)) ||
            throw(ArgumentError("Jacobi coefficients must have unit norm"))
        new{T}(c, s)
    end
end
FaerJacobiRotation(c::Number, s::Number) =
    FaerJacobiRotation{_reflector_type(promote_type(typeof(c), typeof(s)))}(c, s)
Base.size(::FaerJacobiRotation) = (2, 2)
function Base.getindex(j::FaerJacobiRotation, i::Int, k::Int)
    checkbounds(j, i, k)
    return i==k ? j.c : i==1 ? -conj(j.s) : j.s
end
Base.Matrix(j::FaerJacobiRotation) = [j.c -conj(j.s); j.s j.c]
Base.copy(j::FaerJacobiRotation) = j
function _jacobi_parameters(p::T, q::T, kind) where {T<:_NativeScalar}
    all(isfinite, (p, q)) || throw(ArgumentError("Jacobi parameters must be finite"))
    input=T[p, q]
    out=zeros(T, 3)
    GC.@preserve input out _native_call(
        RustCrate.jacobi_parameters,
        Ptr{UInt8}(pointer(input)),
        Ptr{UInt8}(pointer(out)),
        _tag(T),
        Int32(kind),
    )
    return FaerJacobiRotation(out[1], out[2]), out[3]
end
"""
    rotg(p, q)

Return `(J, r)` using faer's rotg, such that `J * [p,q] ≈ [r,0]`.
The native Jacobi sign convention can differ from Julia's `givens` convention.
The bridge corrects faer's complex zero-q returned r and scales real inputs to
preserve unit coefficients for subnormal inputs. Supports all native scalar types.
"""
function rotg(p::Number, q::Number)
    T=_reflector_type(promote_type(typeof(p), typeof(q)))
    return _jacobi_parameters(T(p), T(q), 0)
end
"""Return faer's real make_givens rotation J annihilating q in `J*[p,q]`."""
function make_givens(p::Real, q::Real)
    T=_reflector_type(promote_type(typeof(p), typeof(q)))
    input=T[p, q]
    all(isfinite, input) || throw(ArgumentError("make_givens parameters must be finite"))
    out=zeros(T, 2)
    GC.@preserve input out _native_call(
        RustCrate.jacobi_make_givens,
        Ptr{UInt8}(pointer(input)),
        Ptr{UInt8}(pointer(out)),
        _tag(T),
    )
    return FaerJacobiRotation(out...)
end
Base.transpose(j::FaerJacobiRotation) = first(_jacobi_parameters(j.c, j.s, 2))
Base.conj(j::FaerJacobiRotation) = first(_jacobi_parameters(j.c, j.s, 3))
Base.adjoint(j::FaerJacobiRotation) = first(_jacobi_parameters(j.c, j.s, 4))
Base.inv(j::FaerJacobiRotation) = adjoint(j)
function LinearAlgebra.Givens(j::FaerJacobiRotation, i1::Integer = 1, i2::Integer = 2)
    i1>0 && i2>0 && i1!=i2 ||
        throw(ArgumentError("Givens indices must be positive and distinct"))
    return LinearAlgebra.Givens(Int(i1), Int(i2), j.c, -conj(j.s))
end
function _jacobi_shape(j, a, right)
    a isa _RotationArray && _rotation_open(a)
    size(a, right ? 2 : 1)==2 || throw(
        DimensionMismatch(
            "standalone Jacobi rotation requires an axis of length two; use Givens(J,i,j) for embedding",
        ),
    )
end
function Base.:*(j::FaerJacobiRotation, a::_HouseholderArray)
    lock(_matrix_lock) do
        _jacobi_shape(j, a, false)
        T=promote_type(eltype(j), _reflector_type(eltype(a)))
        result=_householder_matrix_source(a, T)
        LinearAlgebra.lmul!(LinearAlgebra.Givens(j), result)
        return a isa _BorrowVector ? _vector_from_source(result) :
               a isa AbstractVector ? vec(Matrix(result)) : result
    end
end
function Base.:*(a::_HouseholderMatrix, j::FaerJacobiRotation)
    lock(_matrix_lock) do
        _jacobi_shape(j, a, true)
        result=_householder_matrix_source(
            a,
            promote_type(eltype(j), _reflector_type(eltype(a))),
        )
        LinearAlgebra.rmul!(result, LinearAlgebra.Givens(j))
        return result
    end
end
function LinearAlgebra.lmul!(j::FaerJacobiRotation, a::_HouseholderArray)
    lock(_matrix_lock) do
        _jacobi_shape(j, a, false)
        return _operator_copy_result!(a, j*a)
    end
end
function LinearAlgebra.rmul!(a::_HouseholderMatrix, j::FaerJacobiRotation)
    lock(_matrix_lock) do
        _jacobi_shape(j, a, true)
        return _operator_copy_result!(a, a*j)
    end
end
function _jacobi_2x2(j, a, b, c, d, right)
    T=promote_type(
        eltype(j),
        _reflector_type(promote_type(typeof(a), typeof(b), typeof(c), typeof(d))),
    )
    input=T[j.c, j.s, a, b, c, d]
    out=zeros(T, 4)
    all(isfinite, input) || throw(ArgumentError("Jacobi 2x2 entries must be finite"))
    GC.@preserve input out _native_call(
        RustCrate.jacobi_apply_2x2,
        Ptr{UInt8}(pointer(input)),
        Ptr{UInt8}(pointer(out)),
        _tag(T),
        right,
    )
    return Tuple(out)
end
"""Apply J on the left to a 2×2 block; return `(m00,m01,m10,m11)` in row order."""
apply_on_the_left_2x2(j::FaerJacobiRotation, a::Number, b::Number, c::Number, d::Number) =
    _jacobi_2x2(j, a, b, c, d, false)
"""Apply J on the right to a 2×2 block; return `(m00,m01,m10,m11)` in row order."""
apply_on_the_right_2x2(j::FaerJacobiRotation, a::Number, b::Number, c::Number, d::Number) =
    _jacobi_2x2(j, a, b, c, d, true)
