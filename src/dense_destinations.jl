# Native writes into dense destinations: native owners and views, borrowed
# Julia arrays (`faerview`) and plain Julia strided arrays of a native scalar.
# Rust's `views::destination` builds the faer `MatMut` from the descriptor for
# one call only; inputs whose bytes overlap the destination are snapshot there
# (`views::destination::operand`), every other input is read in place.

# Native operands are read through their own descriptors; Julia strided
# operands are borrowed read-only (repeated indices are fine for reads).
const _GemmNative=Union{FaerMatrix,FaerVector,FaerRow,FaerView,FaerBorrowedArray}
const _GemmNativeMatrix=Union{FaerMatrix,FaerRow,FaerMatrixView,FaerBorrowedMatrix}
const _GemmNativeVector=Union{FaerVector,FaerVectorView,FaerBorrowedVector}
const _GemmTransposed=Union{
    LinearAlgebra.Transpose{<:Any,<:_GemmNativeMatrix},
    LinearAlgebra.Adjoint{<:Any,<:_GemmNativeMatrix},
}
const _GemmLeft=Union{_GemmNativeMatrix,_GemmTransposed}
const _GemmJuliaMatrix=Union{
    StridedMatrix{<:_NativeScalar},
    LinearAlgebra.Transpose{<:Any,<:StridedMatrix{<:_NativeScalar}},
    LinearAlgebra.Adjoint{<:Any,<:StridedMatrix{<:_NativeScalar}},
}
const _GemmJuliaVector=StridedVector{<:_NativeScalar}
# Row-vector left operands of `FaerRow` destinations: `v'`/`transpose(v)` of
# native, borrowed or Julia strided vectors (a `FaerRow` is a native matrix).
const _GemmRowLeft=Union{
    LinearAlgebra.Transpose{<:Any,<:Union{_GemmNativeVector,_GemmJuliaVector}},
    LinearAlgebra.Adjoint{<:Any,<:Union{_GemmNativeVector,_GemmJuliaVector}},
}
const _GemmJuliaDestination=Union{StridedVecOrMat{<:_NativeScalar},FaerBorrowedArray}

_gemm_input(x::_GemmNative) = x
_gemm_input(x::StridedVecOrMat) = _julia_borrowable(x) ? FaerBorrowedArray(x) : nothing
_gemm_input(x::LinearAlgebra.Transpose) = _gemm_wrap(transpose, parent(x))
_gemm_input(x::LinearAlgebra.Adjoint) = _gemm_wrap(adjoint, parent(x))
_gemm_input(::Any) = nothing
function _gemm_wrap(f, p)
    q=_gemm_input(p)
    return isnothing(q) ? nothing : f(q)
end
# An operand of scalar `P` with a view descriptor; others are converted copies
# (converted natively from the descriptor when there is one).
function _gemm_operand(x, ::Type{P}) where {P}
    y=_gemm_input(x)
    if !isnothing(y)
        eltype(x)===P && return y
        _native_promotable(eltype(x), P) && return _native_convert(y, P)
    end
    m=size(x, 1)
    return FaerMatrix{P}(reshape(Array{P}(_gemm_materialize(x)), m, :))
end
# Native transposes and views are materialized natively before export.
_gemm_materialize(x::Union{_GemmTransposed,_GemmNative}) = Array(copy(x))
_gemm_materialize(x) = x
_injective_descriptor(d) = (d[3]!=0 || d[4]<=1) && (d[6]!=0 || d[7]<=1)
# The destination itself when it can be borrowed mutably, otherwise nothing.
function _gemm_destination(c::Union{FaerMatrix,FaerVector,FaerRow,FaerView})
    _, d=_view_source(c)
    return _injective_descriptor(d) && d[9]==0 ? c : nothing
end
# Read-only borrows (NPY buffers, possibly PROT_READ mappings) are never
# written, natively or through a Julia fallback.
_check_writable(c::FaerBorrowedArray) =
    _borrowed_writable(c) ||
    throw(ArgumentError("borrowed array is read-only; copy it to a FaerMatrix first"))
_check_writable(c) = true
function _gemm_destination(c::FaerBorrowedArray)
    _check_writable(c)
    return _faerview_valid(parent(c)) ? c : nothing
end
_gemm_destination(c::StridedVecOrMat{<:_NativeScalar}) =
    _faerview_valid(c) ? FaerBorrowedArray(c) : nothing
_gemm_destination(::Any) = nothing

function _view_gemm!(c, a, b, alpha::Ref{T}, beta::Ref{T}, alpha_zero, beta_zero, team) where {T}
    p, d=_view_source(c)
    q, e=_view_source(a)
    r, f=_view_source(b)
    GC.@preserve d e f alpha beta _native_call(
        RustCrate.view_gemm,
        p,
        pointer(d),
        q,
        pointer(e),
        r,
        pointer(f),
        Ptr{UInt8}(Base.unsafe_convert(Ptr{T}, alpha)),
        Ptr{UInt8}(Base.unsafe_convert(Ptr{T}, beta)),
        alpha_zero,
        beta_zero,
        team,
    )
    return c
end

# Dense `mul!(C, A, B, alpha, beta)` computed by faer directly in `C`.
#
# `C` may be a native owner or view, a borrowed Julia array or a plain Julia
# strided array of a native scalar; operands may be native, borrowed or Julia
# strided (with lazy transposes/adjoints). With one scalar type throughout no
# operand or result is staged: operands overlapping `C` (`mul!(C, C, B)`,
# views of the same owner or Julia memory) are snapshot natively and the rest
# are read in place. Mixed scalar types promote the operands; a destination that
# cannot hold the promoted type receives the converted result only after it has
# been completely computed. `beta == 0` never reads `C` and `alpha == 0` never
# reads `A` or `B`.
#
# `team` lends a caller-owned `FaerTeam` to faer's gemm for this call only;
# faer's cost model keeps products too small to pay for a split on one member.
function _dense_mul!(c, a, b, alpha, beta; team = nothing)
    lock(_matrix_lock) do
        isnothing(team) || _handle(team)
        _check_writable(c)
        size(a, 2)==size(b, 1) || throw(DimensionMismatch("inner dimensions must match"))
        size(c, 1)==size(a, 1) && size(c, 2)==size(b, 2) ||
            throw(DimensionMismatch("output dimensions must match"))
        T=eltype(c)
        P=promote_type(eltype(a), eltype(b), T)
        P<:_NativeScalar ||
            throw(ArgumentError("dense mul! requires native scalar types"))
        # Coefficients are never rounded silently (for example BigFloat).
        promote_type(P, typeof(alpha), typeof(beta))<:_NativeScalar ||
            throw(ArgumentError("dense mul! coefficients require native scalar types"))
        x=_gemm_operand(a, P)
        y=_gemm_operand(b, P)
        al, be=Ref{P}(P(alpha)), Ref{P}(P(beta))
        dest=P===T ? _gemm_destination(c) : nothing
        if isnothing(dest)
            dims=(size(c, 1), size(c, 2))
            work=iszero(beta) ? FaerMatrix{P}(undef, dims...) :
                 FaerMatrix{P}(reshape(Array{P}(_gemm_materialize(c)), dims))
            _view_gemm!(work, x, y, al, be, iszero(alpha), iszero(beta), team)
            # Convert completely before writing (InexactError leaves C intact).
            copyto!(c, reshape(Array{T}(Matrix(work)), size(c)))
        else
            _view_gemm!(dest, x, y, al, be, iszero(alpha), iszero(beta), team)
        end
        return c
    end
end

# Destinations: native owners/views, borrowed and Julia strided arrays. Each
# method needs one native or borrowed operand; Julia-only products stay BLAS.
# `FaerRow` destinations also take row-vector left operands (`mul!(r, v', B)`).
for (D, A, B) in (
    (:(Union{FaerMatrix,FaerRow,FaerMatrixView}), :_GemmLeft, :_GemmLeft),
    (:(Union{FaerMatrix,FaerRow,FaerMatrixView}), :_GemmLeft, :_GemmJuliaMatrix),
    (:(Union{FaerMatrix,FaerRow,FaerMatrixView}), :_GemmJuliaMatrix, :_GemmLeft),
    (:FaerRow, :_GemmRowLeft, :(Union{_GemmLeft,_GemmJuliaMatrix})),
    (:_GemmJuliaDestination, :_GemmLeft, :_GemmLeft),
    (:_GemmJuliaDestination, :_GemmLeft, :_GemmJuliaMatrix),
    (:_GemmJuliaDestination, :_GemmJuliaMatrix, :_GemmLeft),
    (:FaerBorrowedMatrix, :_GemmJuliaMatrix, :_GemmJuliaMatrix),
    (:(Union{FaerVectorView,_GemmJuliaDestination}), :_GemmLeft, :_GemmNativeVector),
    (:(Union{FaerVectorView,_GemmJuliaDestination}), :_GemmLeft, :_GemmJuliaVector),
    (:(Union{FaerVectorView,_GemmJuliaDestination}), :_GemmJuliaMatrix, :_GemmNativeVector),
    (:FaerBorrowedVector, :_GemmJuliaMatrix, :_GemmJuliaVector),
)
    @eval begin
        LinearAlgebra.mul!(
            c::$D,
            a::$A,
            b::$B,
            alpha::Number,
            beta::Number;
            team = nothing,
        ) = _dense_mul!(c, a, b, alpha, beta; team)
        LinearAlgebra.mul!(c::$D, a::$A, b::$B; team = nothing) =
            _dense_mul!(c, a, b, true, false; team)
    end
end

const _CopyNativeSource=Union{FaerMatrix,FaerVector,FaerRow,FaerView,FaerBorrowedArray}
# Same-shape, same-scalar native sources are written by one native view
# assignment directly into Julia memory; everything else keeps Base's
# linear-order copyto! semantics through a Julia array.
function _copyto_julia!(dest::AbstractArray, src::AbstractArray)
    lock(_matrix_lock) do
        _check_writable(dest)
        same=size(dest)==size(src) && eltype(dest)===eltype(src)
        target=same ? _gemm_destination(dest) : nothing
        input=same ? _gemm_input(src) : nothing
        if isnothing(target) || isnothing(input)
            julia=dest isa FaerBorrowedArray ? parent(dest) : dest
            copyto!(julia, src isa _CopyNativeSource ? Array(src) : src)
            return dest
        end
        p, d=_view_source(target)
        q, e=_view_source(input)
        GC.@preserve d e _native_call(RustCrate.view_assign_into, p, pointer(d), q, pointer(e))
        return dest
    end
end
# Other sources (sparse, ranges, CHOLMOD dense, ...) use Base's methods on the
# borrowed array.
const _CopyJuliaSource=Union{
    Array,
    SubArray{<:Any,<:Any,<:Array},
    LinearAlgebra.Transpose{<:Any,<:Union{Matrix,SubArray{<:Any,2,<:Array}}},
    LinearAlgebra.Adjoint{<:Any,<:Union{Matrix,SubArray{<:Any,2,<:Array}}},
}
Base.copyto!(
    dest::FaerBorrowedArray,
    src::Union{_CopyNativeSource,_GemmTransposed,_CopyJuliaSource},
) = _copyto_julia!(dest, src)
Base.copyto!(dest::StridedArray{T}, src::_CopyNativeSource) where {T<:_NativeScalar} =
    _copyto_julia!(dest, src)
