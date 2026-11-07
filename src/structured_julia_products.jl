# Symmetric/Hermitian and banded wrappers in products with native operands.
# Without these methods LinearAlgebra's generic kernels read native operands
# (or write native destinations) element by element.

# faer has no self-adjoint product kernel: the stored triangle is expanded
# once, natively, into a full owner (Hermitian diagonals keep their real part)
# and the product then takes the dense gemm paths, which borrow Julia operands
# and write Julia or native destinations in place. Julia parents are borrowed
# for the expansion.
const _SelfAdjointNativeParent=Union{_BorrowMatrix,_BorrowTranspose}
const _NativeSelfAdjoint=Union{
    LinearAlgebra.Symmetric{<:Any,<:_SelfAdjointNativeParent},
    LinearAlgebra.Hermitian{<:Any,<:_SelfAdjointNativeParent},
}
const _JuliaSelfAdjoint=Union{
    LinearAlgebra.Symmetric{<:_NativeScalar,<:StridedMatrix},
    LinearAlgebra.Hermitian{<:_NativeScalar,<:StridedMatrix},
}
const _ProductSelfAdjoint=Union{_NativeSelfAdjoint,_JuliaSelfAdjoint}

function _selfadjoint_full(s)
    lock(_matrix_lock) do
        T=eltype(s)
        y=parent(s)
        g=y isa StridedMatrix ? _gemm_input(y) : y
        isnothing(g) && return FaerMatrix(Matrix(s))
        p, d=_view_source(g)
        args=(p, pointer(d), s.uplo=='L', s isa LinearAlgebra.Hermitian)
        GC.@preserve g d return T===Float64 ?
                              _matrix_result(RustCrate.selfadjoint_expand, size(s), args...) :
                              _typed_result(T, RustCrate.selfadjoint_expand, size(s), args...)
    end
end
_selfadjoint_operand(x::_ProductSelfAdjoint) = _selfadjoint_full(x)
_selfadjoint_operand(x) = x

# Each pair has one native operand; first and second argument types of the
# Julia-only and native-only halves are disjoint.
const _SelfAdjointPairs=(
    (:_NativeSelfAdjoint, :_GemmJuliaMatrix),
    (:_GemmJuliaMatrix, :_NativeSelfAdjoint),
    (:_ProductSelfAdjoint, :_GemmLeft),
    (:_GemmLeft, :_ProductSelfAdjoint),
)
for (A, B) in (
    _SelfAdjointPairs...,
    (:_NativeSelfAdjoint, :_ProductSelfAdjoint),
    (:_JuliaSelfAdjoint, :_NativeSelfAdjoint),
    (:_NativeSelfAdjoint, :_ProductTriangular),
    (:_JuliaSelfAdjoint, :_NativeProductTriangular),
    (:_ProductTriangular, :_NativeSelfAdjoint),
    (:_NativeProductTriangular, :_JuliaSelfAdjoint),
    (:_NativeSelfAdjoint, :(Union{_BorrowVector,_GemmJuliaVector})),
)
    @eval Base.:*(a::$A, b::$B) = _selfadjoint_operand(a)*_selfadjoint_operand(b)
end
for D in (:_ProductDense, :_GemmJuliaDestination), (A, B) in _SelfAdjointPairs
    @eval begin
        LinearAlgebra.mul!(
            c::$D,
            a::$A,
            b::$B,
            alpha::Number,
            beta::Number;
            team = nothing,
        ) = LinearAlgebra.mul!(
            c,
            _selfadjoint_operand(a),
            _selfadjoint_operand(b),
            alpha,
            beta;
            team,
        )
        LinearAlgebra.mul!(c::$D, a::$A, b::$B; team = nothing) =
            LinearAlgebra.mul!(c, _selfadjoint_operand(a), _selfadjoint_operand(b); team)
    end
end

# Matrix-vector `mul!` with an expanded self-adjoint operand.
for (D, A, B) in (
    (:(Union{FaerVector,FaerVectorView,_GemmJuliaDestination}), :_NativeSelfAdjoint, :(Union{_GemmNativeVector,_GemmJuliaVector})),
    (:(Union{FaerVector,FaerVectorView}), :_JuliaSelfAdjoint, :(Union{_GemmNativeVector,_GemmJuliaVector})),
    (:_GemmJuliaDestination, :_JuliaSelfAdjoint, :_GemmNativeVector),
)
    @eval begin
        LinearAlgebra.mul!(
            c::$D,
            a::$A,
            b::$B,
            alpha::Number,
            beta::Number;
            team = nothing,
        ) = LinearAlgebra.mul!(c, _selfadjoint_full(a), b, alpha, beta; team)
        LinearAlgebra.mul!(c::$D, a::$A, b::$B; team = nothing) =
            LinearAlgebra.mul!(c, _selfadjoint_full(a), b; team)
    end
end

# Banded wrappers: faer has no banded product kernel, and expanding the band
# into a dense native owner would turn an O(n²) product into an O(n³) gemm.
# The native operand is copied to Julia once (one native view copy),
# LinearAlgebra's banded kernel computes the product, and the result is a
# native matrix (native destinations receive it in one native assignment).
const _ProductBanded=Union{
    LinearAlgebra.Bidiagonal,
    LinearAlgebra.Tridiagonal,
    LinearAlgebra.SymTridiagonal,
}
const _BandedNative=Union{_BorrowMatrix,_BorrowTranspose}
_banded_julia(x::_BorrowTranspose) = Array(copy(x))
_banded_julia(x) = Array(x)
function _banded_store!(c, w::Matrix)
    lock(_matrix_lock) do
        p, d=_view_source(c)
        b=FaerBorrowedArray(w)
        q, e=_view_source(b)
        GC.@preserve b d e _native_call(RustCrate.view_assign_into, p, pointer(d), q, pointer(e))
        return c
    end
end
function _banded_mul!(c, a, b, alpha, beta)
    x=a isa _BandedNative ? _banded_julia(a) : a
    y=b isa _BandedNative ? _banded_julia(b) : b
    c isa _GemmJuliaDestination &&
        return (LinearAlgebra.mul!(c isa FaerBorrowedArray ? parent(c) : c, x, y, alpha, beta); c)
    w=iszero(beta) ? Matrix{eltype(c)}(undef, size(c)) : Matrix(c)
    LinearAlgebra.mul!(w, x, y, alpha, beta)
    return _banded_store!(c, w)
end
for (A, B) in ((:_ProductBanded, :_BandedNative), (:_BandedNative, :_ProductBanded))
    @eval begin
        function Base.:*(a::$A, b::$B)
            P=promote_type(eltype(a), eltype(b))
            P<:_NativeScalar || return invoke(*, Tuple{AbstractMatrix,AbstractMatrix}, a, b)
            x=a isa _BandedNative ? _banded_julia(a) : a
            y=b isa _BandedNative ? _banded_julia(b) : b
            return FaerMatrix(x*y)
        end
    end
    for D in (:_ProductDense, :_GemmJuliaDestination)
        @eval begin
            LinearAlgebra.mul!(c::$D, a::$A, b::$B, alpha::Number, beta::Number) =
                _banded_mul!(c, a, b, alpha, beta)
            LinearAlgebra.mul!(c::$D, a::$A, b::$B) = _banded_mul!(c, a, b, true, false)
        end
    end
end
function Base.:*(a::_ProductBanded, b::_BorrowVector)
    promote_type(eltype(a), eltype(b))<:_NativeScalar ||
        return invoke(*, Tuple{AbstractMatrix,AbstractVector}, a, b)
    return FaerVector(a*Vector(b))
end
