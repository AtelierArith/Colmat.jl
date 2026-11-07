abstract type AbstractFaerHouseholder{T} <: AbstractMatrix{T} end

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
In-place Julia arrays must use one of the four native scalar types.
Application to arrays does not form H as a dense matrix; `Matrix(H)` explicitly
does. Composing two reflections returns a dense native product.
`lmul!`/`rmul!` and three/five-argument `mul!(C, H, B[, α, β])` write native
owners and views, borrowed arrays (`faerview`) and injective Julia strided
arrays of the working scalar directly through faer (no staged result): a `B`
overlapping `C` is snapshot natively, `β == 0` copies `B` into `C` and reflects
there without reading `C`, `α == 0` never reads `B`, and `β ≠ 0` uses one
native product buffer. Coefficients convert to the operands' scalar (Float32
data with Float64 literals stays native). Mixed scalar types and repeated-index
destinations stage a result and complete conversion before writing. The same
rules hold for `FaerHouseholderSequence` and compact Hessenberg/tridiagonal Q.
Read-only borrowed destinations are rejected before any write. In-place calls
convert the reflection's stored coefficients/metadata to the destination type
before mutation. Adjoint/copy/conjugate/transpose return independent owners.
`lmul!/rmul!/mul!/ldiv!/rdiv!` accept caller-owned `scratch=` and `team=`;
see `scratch_requirements`. Staged (mixed-scalar) results and conversions allocate separately.
"""
mutable struct FaerHouseholder{T,H} <: AbstractFaerHouseholder{T}
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
function _householder_coerce(h::FaerHouseholder{T}, ::Type{T}) where {T<:_NativeScalar}
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
const _HouseholderMatrix=Union{_RotationMatrix,StridedMatrix{<:Number},FaerBorrowedMatrix}
const _HouseholderVector=Union{_BorrowVector,StridedVector{<:Number},FaerBorrowedVector}
const _HouseholderArray=Union{_HouseholderMatrix,_HouseholderVector}
function _householder_check(h, a, right)
    _handle(h)
    a isa _RotationArray && _rotation_open(a)
    _householder_workspace_query(h, 0, right)
    size(a, right ? 2 : 1)==size(h, 1) ||
        throw(DimensionMismatch("Householder application dimensions must match"))
    return nothing
end
_householder_native_apply!(h::FaerHouseholder, a, right) =
    _native_call(RustCrate.householder_apply, a, h, right)
function _householder_apply!(
    h::AbstractFaerHouseholder,
    a::Union{FaerMatrix,FaerVector,FaerRow},
    right;
    scratch = nothing,
    team = nothing,
)
    lock(_matrix_lock) do
        _householder_check(h, a, right)
        reflected=_householder_coerce(h, eltype(a))
        _householder_resource_apply!(reflected, a, right; scratch, team)
        return a
    end
end
# Native destination actions `dest = alpha*H*src + beta*dest` (`right`:
# `alpha*src*H + beta*dest`) for `H` already of `eltype(dest)`. `dest` is a
# mutable view destination (`_gemm_destination`), `src` a readable view input
# of the same scalar (`_gemm_input`) or `nothing` for in-place application.
# faer writes the destination's `MatMut` directly (Rust
# `views::action_into`): `beta == 0` copies `src` into `dest` and applies
# there (no temporary; `dest` is not read), `alpha == 0` never reads `src`, and
# otherwise one native product buffer precedes the axpby. A source overlapping
# `dest` is snapshot natively.
function _householder_into_native!(h::FaerHouseholder, p, d, q, e, right, al, be, az, bz; scratch, team)
    _native_call(RustCrate.householder_mul_into, p, d, q, e, h, right, al, be, az, bz, scratch, team)
end
function _householder_into!(h, dest, src, alpha, beta, right; scratch, team)
    T=eltype(dest)
    p, d=_view_source(dest)
    q, e=isnothing(src) ? (Ptr{UInt8}(C_NULL), Int[]) : _view_source(src)
    al, be=Ref{T}(alpha), Ref{T}(beta)
    GC.@preserve d e al be _householder_into_native!(
        h,
        p,
        pointer(d),
        q,
        isnothing(src) ? Ptr{Int}(C_NULL) : pointer(e),
        right,
        Ptr{UInt8}(Base.unsafe_convert(Ptr{T}, al)),
        Ptr{UInt8}(Base.unsafe_convert(Ptr{T}, be)),
        iszero(alpha),
        iszero(beta);
        scratch,
        team,
    )
    return dest
end
# Every reflector family is applied by faer directly in any destination it can
# borrow mutably (native owners and views, borrowed and Julia strided arrays).
function _householder_apply_into!(h, dest, right; scratch, team)
    req=_householder_resource_check(h, dest, eltype(dest), right; scratch, team)
    _householder_into!(h, dest, nothing, true, false, right; scratch, team)
    return req
end
function _householder_apply!(
    h::AbstractFaerHouseholder,
    a::FaerView,
    right;
    scratch = nothing,
    team = nothing,
)
    lock(_matrix_lock) do
        _householder_check(h, a, right)
        reflected=_householder_coerce(h, eltype(a))
        if !isnothing(_gemm_destination(a))
            _householder_apply_into!(reflected, a, right; scratch, team)
            return a
        end
        result=copy(a)
        _householder_apply!(h, result, right; scratch, team)
        return copyto!(a, result)
    end
end
function _householder_apply!(
    h::AbstractFaerHouseholder,
    a::Union{StridedVector,StridedMatrix,FaerBorrowedArray},
    right;
    scratch = nothing,
    team = nothing,
)
    lock(_matrix_lock) do
        _check_writable(a)
        _householder_check(h, a, right)
        eltype(a)<:_NativeScalar || throw(
            ArgumentError(
                "in-place Householder arrays require Float32, Float64, ComplexF32 or ComplexF64",
            ),
        )
        reflected=_householder_coerce(h, eltype(a))
        dest=_gemm_destination(a)
        if !isnothing(dest)
            _householder_apply_into!(reflected, dest, right; scratch, team)
            return a
        end
        julia=a isa FaerBorrowedArray ? parent(a) : a
        result=julia isa AbstractVector ? FaerVector(julia) : FaerMatrix(julia)
        _householder_apply!(h, result, right; scratch, team)
        copyto!(julia, result isa FaerVector ? Vector(result) : Matrix(result))
        return a
    end
end
LinearAlgebra.lmul!(
    h::AbstractFaerHouseholder,
    a::_HouseholderArray;
    scratch = nothing,
    team = nothing,
) = _householder_apply!(h, a, false; scratch, team)
LinearAlgebra.rmul!(
    a::_HouseholderMatrix,
    h::AbstractFaerHouseholder;
    scratch = nothing,
    team = nothing,
) = _householder_apply!(h, a, true; scratch, team)
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
Base.:*(h::AbstractFaerHouseholder, a::_HouseholderArray) =
    _householder_product(h, a, false)
Base.:*(a::_HouseholderMatrix, h::AbstractFaerHouseholder) =
    _householder_product(h, a, true)
function Base.:*(a::AbstractFaerHouseholder, b::AbstractFaerHouseholder)
    lock(_matrix_lock) do
        _handle(a)
        _handle(b)
        size(a)==size(b) || throw(DimensionMismatch("reflection dimensions must match"))
        return a*FaerMatrix(b)
    end
end
Base.:\(h::FaerHouseholder, a::_HouseholderArray) = h*a
function FaerMatrix(h::AbstractFaerHouseholder{T}) where {T}
    lock(_matrix_lock) do
        _handle(h)
        result=FaerMatrix{T}(Matrix{T}(LinearAlgebra.I, size(h)...))
        return _householder_apply!(h, result, false)
    end
end
Base.Matrix(h::AbstractFaerHouseholder) = Matrix(FaerMatrix(h))
Base.Matrix{T}(h::AbstractFaerHouseholder) where {T} = Matrix{T}(Matrix(h))

function _householder_matrix_source(a, ::Type{T}) where {T}
    a isa _BorrowMatrix && return _coerce(copy(a), T)
    a isa _BorrowVector && return _coerce(_copy_view_matrix(a), T)
    # Strided Julia arrays of the target type copy straight into native storage.
    a isa StridedVecOrMat{T} &&
        _julia_borrowable(a) &&
        return _copy_view_matrix(FaerBorrowedArray(a))
    return a isa AbstractVector ? FaerMatrix{T}(reshape(a, :, 1)) : FaerMatrix{T}(a)
end
# Working scalar of a Householder product: operands and destination only.
# Coefficients convert to it (as dense `mul!`), so Float32 data with Float64
# literals stays Float32 and native.
function _householder_mul_scalar(dest, h, a, alpha, beta)
    T=eltype(dest)
    T<:_NativeScalar || throw(
        ArgumentError("Householder mul! destinations require a native scalar type"),
    )
    P=promote_type(eltype(h), _reflector_type(eltype(a)), T)
    P<:_NativeScalar && promote_type(P, typeof(alpha), typeof(beta))<:_NativeScalar ||
        throw(ArgumentError("unsupported Householder product scalar type"))
    return P
end
function _householder_mul!(
    dest,
    h,
    a,
    alpha,
    beta,
    right;
    scratch = nothing,
    team = nothing,
)
    lock(_matrix_lock) do
        _check_writable(dest)
        _householder_check(h, a, right)
        _householder_check(h, dest, right)
        size(dest)==size(a) || throw(
            DimensionMismatch(
                "Householder product destination must match the source shape",
            ),
        )
        T=eltype(dest)
        P=_householder_mul_scalar(dest, h, a, alpha, beta)
        al, be=P(alpha), P(beta)
        _householder_resource_check(h, a, P, right; scratch, team)
        target=P===T ? _gemm_destination(dest) : nothing
        src=eltype(a)===P ? _gemm_input(a) : nothing
        if !isnothing(target) && !isnothing(src)
            _dense_rhs_validate(a)
            _dense_rhs_validate(dest)
            reflected=_householder_coerce(h, P)
            _householder_into!(reflected, target, src, al, be, right; scratch, team)
            return dest
        end
        dims=ndims(a)==1 ? (length(a), 1) : size(a)
        result=if iszero(al)
            _zeros(P, dims)
        else
            work=_householder_matrix_source(a, P)
            al*_householder_apply!(_householder_coerce(h, P), work, right; scratch, team)
        end
        !iszero(be) && (result=result+be*_householder_matrix_source(dest, P))
        # Convert the complete result before any destination writes, including
        # real destinations that cannot represent a complex product.
        buffer=ndims(dest)==1 ? vec(Matrix{T}(result)) : Matrix{T}(result)
        copyto!(dest isa FaerBorrowedArray ? parent(dest) : dest, buffer)
        return dest
    end
end
LinearAlgebra.mul!(
    dest::_HouseholderMatrix,
    h::AbstractFaerHouseholder,
    a::_HouseholderMatrix,
    alpha::Number,
    beta::Number;
    scratch = nothing,
    team = nothing,
) = _householder_mul!(dest, h, a, alpha, beta, false; scratch, team)
LinearAlgebra.mul!(
    dest::_HouseholderVector,
    h::AbstractFaerHouseholder,
    a::_HouseholderVector,
    alpha::Number,
    beta::Number;
    scratch = nothing,
    team = nothing,
) = _householder_mul!(dest, h, a, alpha, beta, false; scratch, team)
LinearAlgebra.mul!(
    dest::_HouseholderMatrix,
    a::_HouseholderMatrix,
    h::AbstractFaerHouseholder,
    alpha::Number,
    beta::Number;
    scratch = nothing,
    team = nothing,
) = _householder_mul!(dest, h, a, alpha, beta, true; scratch, team)
LinearAlgebra.mul!(
    dest::_HouseholderMatrix,
    h::AbstractFaerHouseholder,
    a::_HouseholderMatrix;
    scratch = nothing,
    team = nothing,
) = _householder_mul!(dest, h, a, 1, 0, false; scratch, team)
LinearAlgebra.mul!(
    dest::_HouseholderVector,
    h::AbstractFaerHouseholder,
    a::_HouseholderVector;
    scratch = nothing,
    team = nothing,
) = _householder_mul!(dest, h, a, 1, 0, false; scratch, team)
LinearAlgebra.mul!(
    dest::_HouseholderMatrix,
    a::_HouseholderMatrix,
    h::AbstractFaerHouseholder;
    scratch = nothing,
    team = nothing,
) = _householder_mul!(dest, h, a, 1, 0, true; scratch, team)

LinearAlgebra.ldiv!(
    h::AbstractFaerHouseholder,
    a::_HouseholderArray;
    scratch = nothing,
    team = nothing,
) = _householder_apply!(adjoint(h), a, false; scratch, team)
LinearAlgebra.ldiv!(
    dest::_HouseholderMatrix,
    h::AbstractFaerHouseholder,
    a::_HouseholderMatrix;
    scratch = nothing,
    team = nothing,
) = _householder_mul!(dest, adjoint(h), a, 1, 0, false; scratch, team)
LinearAlgebra.ldiv!(
    dest::_HouseholderVector,
    h::AbstractFaerHouseholder,
    a::_HouseholderVector;
    scratch = nothing,
    team = nothing,
) = _householder_mul!(dest, adjoint(h), a, 1, 0, false; scratch, team)
LinearAlgebra.rdiv!(
    a::_HouseholderMatrix,
    h::AbstractFaerHouseholder;
    scratch = nothing,
    team = nothing,
) = _householder_apply!(adjoint(h), a, true; scratch, team)
Base.:/(a::_HouseholderMatrix, h::AbstractFaerHouseholder) = a*adjoint(h)
