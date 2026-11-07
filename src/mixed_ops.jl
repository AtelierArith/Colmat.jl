# Mixed operands. Native promotion converts natively from the borrowed source;
# strided Julia arrays of a native scalar type are borrowed in place
# (FaerBorrowedArray) instead of being copied into Julia staging arrays.
_native_promotable(::Type{S}, ::Type{T}) where {S,T} = !(S<:Complex && T<:Real)
function _native_convert(a, ::Type{T}) where {T<:_NativeScalar}
    lock(_matrix_lock) do
        p, d=_view_source(a)
        dims=size(a)
        GC.@preserve d return T===Float64 ?
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
    LinearAlgebra.Transpose{<:Any,<:Union{FaerMatrix,FaerMatrixView}},
    LinearAlgebra.Adjoint{<:Any,<:Union{FaerMatrix,FaerMatrixView}},
}
# Promote a dense operand to the native scalar `P` with at most one copy.
function _promote_operand(a, ::Type{P}) where {P}
    eltype(a)===P && return a
    a isa _ConvertibleNative &&
        _native_promotable(eltype(a), P) &&
        return _native_convert(a, P)
    return ndims(a)==1 ? FaerVector{P}(Vector(a)) : FaerMatrix{P}(Matrix(a))
end

_julia_operand(b) =
    b isa StridedVecOrMat && _julia_borrowable(b) ? FaerBorrowedArray(b) :
    ndims(b)==1 ? FaerVector(b) : FaerMatrix(b)
const _JuliaNativeMatrix=StridedMatrix{<:_NativeScalar}
const _JuliaNativeVector=StridedVector{<:_NativeScalar}
for f in (:+, :-)
    @eval begin
        Base.$f(a::_BorrowMatrix, b::_JuliaNativeMatrix) = $f(a, _julia_operand(b))
        Base.$f(a::_JuliaNativeMatrix, b::_BorrowMatrix) = $f(_julia_operand(a), b)
    end
end
Base.:*(a::_BorrowMatrix, b::_JuliaNativeMatrix) = a*_julia_operand(b)
Base.:*(a::_JuliaNativeMatrix, b::_BorrowMatrix) = _julia_operand(a)*b
# mul! with Julia operands and native or Julia destinations is handled by the
# central destination path in dense_destinations.jl.
# A native vector destination with Julia operands is the one combination
# missing from that table; route it through the same native gemm path.
for (A, B) in ((:_GemmLeft, :_GemmJuliaVector), (:_GemmJuliaMatrix, :_GemmNativeVector))
    @eval begin
        LinearAlgebra.mul!(c::FaerVector, a::$A, b::$B, alpha::Number, beta::Number) =
            _dense_mul!(c, a, b, alpha, beta)
        LinearAlgebra.mul!(c::FaerVector, a::$A, b::$B) = _dense_mul!(c, a, b, true, false)
    end
end
