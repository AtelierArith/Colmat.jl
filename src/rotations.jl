_rotation_scalar(::LinearAlgebra.Givens{T}) where {T} = T
const _RotationMatrix=Union{_BorrowMatrix,FaerRow}
const _RotationArray=Union{_RotationMatrix,_BorrowVector}
_rotation_open(a::FaerView) = _view_owner(a)
_rotation_open(a::Union{FaerMatrix,FaerVector,FaerRow}) = _handle(a)
function _native_givens(p::T, q::T, i1::Integer, i2::Integer) where {T<:_NativeScalar}
    i1!=i2 || throw(ArgumentError("Givens indices must be distinct"))
    all(isfinite, (p, q)) || throw(ArgumentError("Givens inputs must be finite"))
    input=T[p, q]
    out=Vector{T}(undef, 3)
    GC.@preserve input out _native_call(
        RustCrate.givens_parameters,
        Ptr{UInt8}(pointer(input)),
        Ptr{UInt8}(pointer(out)),
        _tag(T),
    )
    c, s, r=out
    if i1>i2
        s=-conj(s)
        i1, i2=i2, i1
    end
    return LinearAlgebra.Givens(Int(i1), Int(i2), c, s), r
end
"""
    givens(A::FaerMatrix, i1, i2, j)
    givens(v::FaerVector, i1, i2)

Compute a standard Julia Givens rotation and r using faer's rotg. The rotation
zeros the original second selected row/entry; all four scalar types and native
views are supported. Inputs must be finite. Applying a Julia Givens with `*`,
`lmul!` or `rmul!` to native arrays uses faer's Jacobi kernels. In-place cosine
must be real and coefficients must convert to the destination scalar type.
Native views stage an independent result before copying back.
"""
function LinearAlgebra.givens(a::_RotationMatrix, i1::Integer, i2::Integer, j::Integer)
    lock(_matrix_lock) do
        _rotation_open(a)
        checkbounds(a, i1, j)
        checkbounds(a, i2, j)
        return _native_givens(a[i1, j], a[i2, j], i1, i2)
    end
end
function LinearAlgebra.givens(a::_BorrowVector, i1::Integer, i2::Integer)
    lock(_matrix_lock) do
        _rotation_open(a)
        checkbounds(a, i1)
        checkbounds(a, i2)
        return _native_givens(a[i1], a[i2], i1, i2)
    end
end
function _rotation_check(a, g, right)
    _rotation_open(a)
    g.i1!=g.i2 && g.i1>=1 && g.i2>=1 ||
        throw(ArgumentError("Givens indices must be positive and distinct"))
    bound=size(a, right ? 2 : 1)
    max(g.i1, g.i2)<=bound ||
        throw(DimensionMismatch("Givens indices exceed the selected axis"))
    isreal(g.c) || throw(ArgumentError("Givens cosine must be real"))
    return nothing
end
function _rotate!(a::Union{FaerMatrix,FaerVector,FaerRow}, g::LinearAlgebra.Givens, right)
    lock(_matrix_lock) do
        _rotation_check(a, g, right)
        T=eltype(a)
        coeff=T[g.c, g.s]
        GC.@preserve coeff _native_call(
            RustCrate.givens_apply,
            a,
            Ptr{UInt8}(pointer(coeff)),
            UInt(g.i1-1),
            UInt(g.i2-1),
            right,
        )
        return a
    end
end
function _rotate!(a::FaerView, g::LinearAlgebra.Givens, right)
    lock(_matrix_lock) do
        _rotation_check(a, g, right)
        result=copy(a)
        _rotate!(result, g, right)
        return copyto!(a, result)
    end
end
LinearAlgebra.lmul!(g::LinearAlgebra.Givens, a::_RotationArray) = _rotate!(a, g, false)
LinearAlgebra.rmul!(a::_RotationMatrix, g::LinearAlgebra.Givens) = _rotate!(a, g, true)
_rotation_copy(a::_BorrowMatrix, ::Type{T}) where {T} = _coerce(copy(a), T)
_rotation_copy(a::_BorrowVector, ::Type{T}) where {T} =
    T===eltype(a) ? copy(a) : FaerVector{T}(a)
_rotation_copy(a::FaerRow, ::Type{T}) where {T} =
    T===eltype(a) ? copy(a) : FaerRow{T}(vec(Matrix(a)))
function Base.:*(g::LinearAlgebra.Givens, a::_RotationArray)
    lock(_matrix_lock) do
        _rotation_check(a, g, false)
        return _rotate!(
            _rotation_copy(a, promote_type(_rotation_scalar(g), eltype(a))),
            g,
            false,
        )
    end
end
function Base.:*(a::_RotationMatrix, g::LinearAlgebra.Givens)
    lock(_matrix_lock) do
        _rotation_check(a, g, true)
        return _rotate!(
            _rotation_copy(a, promote_type(_rotation_scalar(g), eltype(a))),
            g,
            true,
        )
    end
end
