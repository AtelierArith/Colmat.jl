# Mixed operands. Native promotion converts natively from the borrowed source;
# strided Julia arrays of a native scalar type are borrowed in place
# (FaerBorrowedArray) instead of being copied into Julia staging arrays.
_native_promotable(::Type{S}, ::Type{T}) where {S,T} = !(S<:Complex && T<:Real)
function _native_convert(a, ::Type{T}) where {T<:_NativeScalar}
    lock(_matrix_lock) do
        p, d=_view_source(a)
        # Vectors and rows convert into an n×1 or 1×n owner (gemm operands).
        dims=(size(a, 1), size(a, 2))
        GC.@preserve a d return T===Float64 ?
                              _matrix_result(RustCrate.view_convert, dims, p, pointer(d)) :
                              _typed_result(
            T,
            RustCrate.view_convert,
            dims,
            p,
            pointer(d),
            _tag(T),
        )
    end
end
const _ConvertibleNative=Union{
    FaerMatrix,
    FaerMatrixView,
    FaerBorrowedMatrix,
    LinearAlgebra.Transpose{<:Any,<:Union{FaerMatrix,FaerMatrixView,FaerBorrowedMatrix}},
    LinearAlgebra.Adjoint{<:Any,<:Union{FaerMatrix,FaerMatrixView,FaerBorrowedMatrix}},
}
# Promote a dense operand to the native scalar `P` with at most one copy.
function _promote_operand(a, ::Type{P}) where {P}
    eltype(a)===P && return a
    a isa _ConvertibleNative &&
        _native_promotable(eltype(a), P) &&
        return _native_convert(a, P)
    return ndims(a)==1 ? FaerVector{P}(Vector(a)) : FaerMatrix{P}(Matrix(a))
end

# Julia strided matrices and vectors (with their lazy transposes/adjoints) are
# borrowed in place; anything else is copied once.
function _julia_operand(b)
    y=_gemm_input(b)
    isnothing(y) || return y
    return ndims(b)==1 ? FaerVector(b) : FaerMatrix(b)
end
# Native matrices, views and their lazy transposes/adjoints.
const _MixedNative=Union{_BorrowMatrix,_BorrowTranspose}
for (f, op) in ((:+, 0), (:-, 1))
    @eval begin
        Base.$f(a::_MixedNative, b::_GemmJuliaMatrix) = $f(a, _julia_operand(b))
        Base.$f(a::_GemmJuliaMatrix, b::_MixedNative) = $f(_julia_operand(a), b)
        # Lazy transposes with native or borrowed matrices read their parent in
        # place (untransposed pairs are in views.jl and borrowed.jl).
        Base.$f(a::_ProductTranspose, b::_ProductLeft) = _view_binary(a, b, $op)
        Base.$f(a::_ProductMatrix, b::_ProductTranspose) = _view_binary(a, b, $op)
    end
end
Base.:*(a::_MixedNative, b::_GemmJuliaMatrix) = a*_julia_operand(b)
Base.:*(a::_GemmJuliaMatrix, b::_MixedNative) = _julia_operand(a)*b

# `a * b` written by faer directly into a new native owner of type `R`.
function _mixed_product(a, b, ::Type{R}) where {R}
    size(a, 2)==size(b, 1) || throw(DimensionMismatch("inner dimensions must match"))
    P=_native_promotion(eltype(a), eltype(b))
    dest=if R===FaerVector
        FaerVector{P}(undef, size(a, 1))
    elseif R===FaerRow
        FaerRow{P}(undef, size(b, 2))
    else
        FaerMatrix{P}(undef, size(a, 1), size(b, 2))
    end
    return _dense_mul!(dest, a, b, true, false)
end
# Native matrices times Julia vectors are native vectors. `FaerMatrix * x`
# accepts any vector: native promoted scalars give a `FaerVector` (strided
# vectors of a native scalar are read in place, others are converted once);
# other promotions (BigFloat, ...) keep LinearAlgebra's exact generic product.
Base.:*(a::Union{FaerMatrixView,_BorrowTranspose}, b::_GemmJuliaVector) =
    _mixed_product(a, b, FaerVector)
function Base.:*(a::FaerMatrix, b::AbstractVector)
    P=promote_type(eltype(a), eltype(b))
    P<:_NativeScalar || return invoke(*, Tuple{AbstractMatrix,AbstractVector}, a, b)
    x=b isa _GemmJuliaVector ? b : Vector{P}(b)
    return _mixed_product(a, x, FaerVector)
end
# Intersections with the native and borrowed vector methods.
Base.:*(a::FaerMatrix, b::FaerVectorView) = _view_binary(a, b, 2; vector = true)
Base.:*(a::FaerMatrix, b::FaerBorrowedVector) = _view_binary(a, b, 2; vector = true)
# Rows times matrices stay rows, and a row times a vector is the one-element
# `FaerVector` of Julia's 1×n matrix-vector product. `v' * B` and
# `transpose(v) * B` use LinearAlgebra's `(B' * v)'` rewrite onto the
# matrix-vector methods.
Base.:*(a::FaerRow, b::Union{_GemmLeft,_GemmJuliaMatrix}) = _mixed_product(a, b, FaerRow)
Base.:*(a::FaerRow, b::Union{_GemmNativeVector,_GemmJuliaVector}) =
    _mixed_product(a, b, FaerVector)
# Lazily conjugated native matrices (`transpose(A')`, `adjoint(transpose(A))`,
# as produced by the `transpose(v) * A'` rewrite) are one descriptor with the
# conjugation flag.
const _MixedConjugated=Union{
    LinearAlgebra.Transpose{<:Any,<:LinearAlgebra.Adjoint{<:Any,<:_BorrowMatrix}},
    LinearAlgebra.Adjoint{<:Any,<:LinearAlgebra.Transpose{<:Any,<:_BorrowMatrix}},
}
Base.:*(a::_MixedConjugated, b::Union{_GemmNativeVector,_GemmJuliaVector}) =
    _mixed_product(a, b, FaerVector)
Base.:*(a::_MixedConjugated, b::Union{_GemmLeft,_GemmJuliaMatrix,_MixedConjugated}) =
    _mixed_product(a, b, FaerMatrix)
Base.:*(a::Union{_GemmLeft,_GemmJuliaMatrix}, b::_MixedConjugated) =
    _mixed_product(a, b, FaerMatrix)

# Scalar multiples of native views and lazy transposes: the operand is copied
# (and promoted) natively from its descriptor and scaled by the owner methods,
# instead of being broadcast through a Julia snapshot. Non-native promotions
# (BigFloat, ...) keep Julia's generic result.
const _MixedNativeView=Union{FaerMatrixView,_BorrowTranspose}
_native_owner(a, ::Type{P}) where {P} = _native_convert(a, P)
function Base.:*(x::Number, a::_MixedNativeView)
    P=promote_type(typeof(x), eltype(a))
    P<:_NativeScalar || return invoke(*, Tuple{Number,AbstractArray}, x, a)
    return x*_native_owner(a, P)
end
Base.:*(a::_MixedNativeView, x::Number) = x*a
function Base.:/(a::_MixedNativeView, x::Number)
    promote_type(eltype(a), typeof(x))<:_NativeScalar ||
        return invoke(/, Tuple{AbstractArray,Number}, a, x)
    return _native_owner(a, eltype(a))/x
end
Base.:-(a::FaerMatrixView) = -_native_owner(a, eltype(a))

# mul! with Julia operands and native or Julia destinations is handled by the
# central destination path in dense_destinations.jl.
# A native vector destination with Julia operands is the one combination
# missing from that table; route it through the same native gemm path.
for (A, B) in ((:_GemmLeft, :_GemmJuliaVector), (:_GemmJuliaMatrix, :_GemmNativeVector))
    @eval begin
        LinearAlgebra.mul!(
            c::FaerVector,
            a::$A,
            b::$B,
            alpha::Number,
            beta::Number;
            team = nothing,
        ) = _dense_mul!(c, a, b, alpha, beta; team)
        LinearAlgebra.mul!(c::FaerVector, a::$A, b::$B; team = nothing) =
            _dense_mul!(c, a, b, true, false; team)
    end
end
