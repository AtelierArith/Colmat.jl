# Float64 keeps its established native owner; the other three primitive types
# use TypedMatrix/TypedFactor. No raw pointer crosses between those owner ABIs.
const _NativeScalar = Union{Float32,Float64,ComplexF32,ComplexF64}
const _ExtraScalar = Union{Float32,ComplexF32,ComplexF64}
# faer's fx128/cx128 and its generic field traits are intentionally not exposed:
# every native owner and factor dispatches over exactly these four types.
_unsupported_scalar(::Type{T}) where {T} = throw(
    ArgumentError(
        "unsupported scalar type $T: native owners support Float32, Float64, ComplexF32 and ComplexF64",
    ),
)
# Untyped constructors keep native inputs and convert other real/complex inputs.
_native_input_type(::Type{T}) where {T<:_NativeScalar} = T
_native_input_type(::Type{<:Real}) = Float64
_native_input_type(::Type{<:Complex}) = ComplexF64
_native_input_type(::Type{T}) where {T} = _unsupported_scalar(T)
# Scalar arithmetic follows promote_type and never rounds to a narrower native type.
function _native_promotion(::Type{S}, ::Type{T}) where {S,T}
    P=promote_type(S, T)
    P<:_NativeScalar || throw(
        ArgumentError(
            "unsupported promoted scalar type $P: native owners support Float32, Float64, ComplexF32 and ComplexF64",
        ),
    )
    return P
end
_tag(::Type{Float32}) = Int32(0)
_tag(::Type{Float64}) = Int32(1)
_tag(::Type{ComplexF32}) = Int32(2)
_tag(::Type{ComplexF64}) = Int32(3)
_realtype(::Type{T}) where {T} = typeof(real(zero(T)))
function _typed_result(::Type{T}, f, dims, args...) where {T<:_ExtraScalar}
    lock(_matrix_lock) do
        h=RustCrate.TypedMatrix()
        _native_call(f, h, args...)
        return FaerMatrix{T,typeof(h)}(h, dims)
    end
end
function _matrix_from_dense(data::Matrix{T}) where {T<:_ExtraScalar}
    GC.@preserve data return _typed_result(
        T,
        RustCrate.load,
        size(data),
        Ptr{UInt8}(pointer(data)),
        UInt(size(data, 1)),
        UInt(size(data, 2)),
        _tag(T),
    )
end
FaerMatrix{T}(a::AbstractMatrix) where {T<:_NativeScalar} =
    _matrix_from_dense(_checked_dense(T, a))
FaerMatrix{T}(::AbstractMatrix) where {T} = _unsupported_scalar(T)
FaerMatrix(a::AbstractMatrix{Float32}) = FaerMatrix{Float32}(a)
FaerMatrix(a::AbstractMatrix{ComplexF32}) = FaerMatrix{ComplexF32}(a)
FaerMatrix(a::AbstractMatrix{ComplexF64}) = FaerMatrix{ComplexF64}(a)
FaerMatrix(a::AbstractMatrix{<:Complex}) = FaerMatrix{ComplexF64}(a)
FaerMatrix(a::FaerMatrix) = copy(a)
_coerce(a::FaerMatrix{T}, ::Type{T}) where {T} = a
_coerce(a::FaerMatrix, ::Type{T}) where {T} =
    T<:_NativeScalar && _native_promotable(eltype(a), T) ? _native_convert(a, T) :
    FaerMatrix{T}(a)
function _zeros(::Type{T}, dims::Tuple{Int,Int}) where {T<:_NativeScalar}
    all(>=(0), dims) || throw(ArgumentError("matrix dimensions must be nonnegative"))
    T===Float64 &&
        return _matrix_result(RustCrate.make_zeros, dims, UInt(dims[1]), UInt(dims[2]))
    return _typed_result(T, RustCrate.zeroes, dims, UInt(dims[1]), UInt(dims[2]), _tag(T))
end
function Base.getindex(a::FaerMatrix{T}, i::Int, j::Int) where {T<:_ExtraScalar}
    @boundscheck checkbounds(a, i, j)
    out=Ref{T}()
    GC.@preserve out _native_call(
        RustCrate.element,
        a,
        UInt(i-1),
        UInt(j-1),
        Ptr{UInt8}(Base.unsafe_convert(Ptr{T}, out)),
    )
    return out[]
end
function Base.setindex!(a::FaerMatrix{T}, x, i::Int, j::Int) where {T<:_ExtraScalar}
    @boundscheck checkbounds(a, i, j)
    value=Ref{T}(x)
    GC.@preserve value _native_call(
        RustCrate.set_element,
        a,
        UInt(i-1),
        UInt(j-1),
        Ptr{UInt8}(Base.unsafe_convert(Ptr{T}, value)),
    )
    return a
end
function Base.Matrix(a::FaerMatrix{T}) where {T<:_ExtraScalar}
    out=Matrix{T}(undef, size(a))
    GC.@preserve out _native_call(
        RustCrate.to_buffer,
        a,
        Ptr{UInt8}(pointer(out)),
        UInt(length(out)),
    )
    return out
end
Base.Array(a::FaerMatrix{T}) where {T<:_ExtraScalar} = Matrix(a)
Base.Matrix{S}(a::FaerMatrix{T}) where {S,T<:_ExtraScalar} = convert(Matrix{S}, Matrix(a))
function Base.deepcopy_internal(a::FaerMatrix{T}, dict::IdDict) where {T<:_ExtraScalar}
    get!(dict, a) do ;
        copy(a)
    end
end
function Base.similar(a::FaerMatrix{T}, ::Type{S}, dims::Dims) where {T<:_ExtraScalar,S}
    return S<:_NativeScalar && length(dims)==2 ? _zeros(S, dims) : Array{S}(undef, dims)
end
Base.zero(a::FaerMatrix{T}) where {T<:_ExtraScalar} = _zeros(T, size(a))
function Base.fill!(a::FaerMatrix{T}, x) where {T<:_ExtraScalar}
    value=Ref{T}(x)
    GC.@preserve value _native_call(
        RustCrate.fill_value,
        a,
        Ptr{UInt8}(Base.unsafe_convert(Ptr{T}, value)),
    )
    return a
end
function Base.copyto!(dest::FaerMatrix{T}, src::FaerMatrix{T}) where {T<:_ExtraScalar}
    size(dest)==size(src) || throw(DimensionMismatch("matrix sizes must match"))
    lock(_matrix_lock) do
        _handle(dest)
        _handle(src)
        dest===src || _native_call(RustCrate.transform_from, dest, src, Int32(0))
    end
    return dest
end
Base.copyto!(dest::FaerMatrix{T}, src::FaerMatrix) where {T} =
    copyto!(dest, _coerce(src, T))
function _typed_binary(a::FaerMatrix{T}, b::FaerMatrix{T}, op) where {T<:_ExtraScalar}
    if op in (0, 1)
        size(a)==size(b) || throw(DimensionMismatch("matrix sizes must match"))
        dims=size(a)
    elseif op==2
        size(a, 2)==size(b, 1) || throw(DimensionMismatch("inner dimensions must match"))
        dims=(size(a, 1), size(b, 2))
    else
        dims=(
            Base.checked_mul(size(a, 1), size(b, 1)),
            Base.checked_mul(size(a, 2), size(b, 2)),
        )
    end
    return _typed_result(T, RustCrate.arithmetic_from, dims, a, b, Int32(op))
end
Base.:+(a::FaerMatrix{T}, b::FaerMatrix{T}) where {T<:_ExtraScalar} = _typed_binary(a, b, 0)
Base.:-(a::FaerMatrix{T}, b::FaerMatrix{T}) where {T<:_ExtraScalar} = _typed_binary(a, b, 1)
Base.:*(a::FaerMatrix{T}, b::FaerMatrix{T}) where {T<:_ExtraScalar} = _typed_binary(a, b, 2)
for op in (:+, :-, :*)
    @eval function Base.$op(a::FaerMatrix{T}, b::FaerMatrix{S}) where {T,S}
        P=promote_type(T, S)
        return Base.$op(_coerce(a, P), _coerce(b, P))
    end
end
Base.:+(a::FaerMatrix{T}) where {T<:_ExtraScalar} = copy(a)
Base.:-(a::FaerMatrix{T}) where {T<:_ExtraScalar} = (-one(_realtype(T)))*a
function Base.:*(x::Number, a::FaerMatrix{T}) where {T}
    P=_native_promotion(typeof(x), T)
    converted=_coerce(a, P)
    P===Float64 && return Float64(x)*converted
    # Real factors scale complex entries componentwise, as Julia's Real*Complex
    # does, so an infinite part never turns the other part into NaN.
    R=x isa Real ? _realtype(P) : P
    factor=Ref{R}(x)
    return GC.@preserve factor _typed_result(
        P,
        R===P ? RustCrate.scale_from : RustCrate.scale_real_from,
        size(a),
        converted,
        Ptr{UInt8}(Base.unsafe_convert(Ptr{R}, factor)),
    )
end
Base.:*(a::FaerMatrix{T}, x::Number) where {T} = x*a
Base.:/(a::FaerMatrix{T}, x::Number) where {T<:_ExtraScalar} = (one(T)/x)*a
Base.:/(a::FaerMatrix{T}, x::Real) where {T<:_ExtraScalar} = (one(_realtype(T))/x)*a
Base.:/(a::FaerMatrix{Float64}, x::Complex) = (1.0/x)*a
Base.:\(x::Number, a::FaerMatrix) = a/x
Base.conj(a::FaerMatrix{T}) where {T<:_ExtraScalar} =
    _typed_result(T, RustCrate.transform_from, size(a), a, Int32(3))
Base.conj(a::FaerMatrix{Float64}) = copy(a)
function Base.copy(a::LinearAlgebra.Transpose{T,<:FaerMatrix{T}}) where {T<:_ExtraScalar}
    return _typed_result(
        T,
        RustCrate.transform_from,
        reverse(size(parent(a))),
        parent(a),
        Int32(1),
    )
end
function Base.copy(a::LinearAlgebra.Adjoint{T,<:FaerMatrix{T}}) where {T<:_ExtraScalar}
    return _typed_result(
        T,
        RustCrate.transform_from,
        reverse(size(parent(a))),
        parent(a),
        Int32(2),
    )
end
const _AnyFaerTransposed=Union{
    LinearAlgebra.Transpose{T,<:FaerMatrix{T}},
    LinearAlgebra.Adjoint{T,<:FaerMatrix{T}},
} where {T}
# Products with lazy transposes and vectors read their operands in place
# (views.jl, mixed_ops.jl).
function LinearAlgebra.svd(
    a::FaerMatrix{T};
    full::Bool = false,
    native::Bool = false,
    scratch = nothing,
    team = nothing,
    alg = nothing,
) where {T<:_ExtraScalar}
    alg===nothing || return _svd_alg(alg) do
        LinearAlgebra.svd(a; full, native, scratch, team)
    end
    native && return _owned_spectral(a; full, scratch, team)
    _svd_default_path(scratch, team) || return _svd_workspace_result(a; full, scratch, team)
    m, n=size(a)
    k=min(m, n)
    R=_realtype(T)
    u=Matrix{T}(undef, m, full ? m : k)
    s=Vector{R}(undef, k)
    v=Matrix{T}(undef, n, full ? n : k)
    GC.@preserve u s v _native_call(
        RustCrate.svd_buffers,
        a,
        Ptr{UInt8}(pointer(u)),
        Ptr{UInt8}(pointer(s)),
        Ptr{UInt8}(pointer(v)),
        full,
        false,
    )
    return SVD(u, s, copy(adjoint(v)))
end
function LinearAlgebra.svdvals(
    a::FaerMatrix{T};
    scratch = nothing,
    team = nothing,
    alg = nothing,
) where {T<:_ExtraScalar}
    alg===nothing || return _svd_alg(()->LinearAlgebra.svdvals(a; scratch, team), alg)
    _svd_default_path(scratch, team) ||
        return _svd_workspace_result(a; values_only = true, scratch, team)
    s=Vector{_realtype(T)}(undef, min(size(a)...))
    GC.@preserve s _native_call(
        RustCrate.svd_buffers,
        a,
        Ptr{UInt8}(C_NULL),
        Ptr{UInt8}(pointer(s)),
        Ptr{UInt8}(C_NULL),
        false,
        true,
    )
    return s
end
faer_svd(a::FaerMatrix{T}) where {T<:_ExtraScalar} = LinearAlgebra.svd(a)
function _typed_scalar(a::FaerMatrix{T}, code) where {T<:_ExtraScalar}
    value=Ref{T}()
    GC.@preserve value _native_call(
        RustCrate.scalar_reduction,
        a,
        Ptr{UInt8}(Base.unsafe_convert(Ptr{T}, value)),
        Int32(code),
    )
    return value[]
end
function Base.sum(a::FaerMatrix{T}; dims = :, kw...) where {T<:_ExtraScalar}
    isempty(kw) || return sum(Matrix(a); dims, kw...)
    dims isa Colon && return _typed_scalar(a, 0)
    dims==1 &&
        return _typed_result(T, RustCrate.axis_sum_from, (1, size(a, 2)), a, Int32(1))
    dims==2 &&
        return _typed_result(T, RustCrate.axis_sum_from, (size(a, 1), 1), a, Int32(2))
    return _native_sum(a, dims)
end
function LinearAlgebra.tr(a::FaerMatrix{T}) where {T<:_ExtraScalar}
    LinearAlgebra.checksquare(a)
    return _typed_scalar(a, 2)
end
function LinearAlgebra.norm(a::FaerMatrix{T}, p::Real = 2) where {T<:_ExtraScalar}
    p in (1, 2, Inf) || return _native_norm(a, p)
    R=_realtype(T)
    value=Ref{R}()
    code=p==2 ? 0 : p==1 ? 1 : 2
    GC.@preserve value _native_call(
        RustCrate.norm_reduction,
        a,
        Ptr{UInt8}(Base.unsafe_convert(Ptr{R}, value)),
        Int32(code),
    )
    return value[]
end
Base.all(::typeof(isfinite), a::FaerMatrix{T}) where {T<:_ExtraScalar} =
    _native_call(RustCrate.predicate, a, Int32(0))
Base.any(::typeof(isnan), a::FaerMatrix{T}) where {T<:_ExtraScalar} =
    _native_call(RustCrate.predicate, a, Int32(1))
Base.isone(a::FaerMatrix{T}) where {T<:_ExtraScalar} =
    _native_call(RustCrate.predicate, a, Int32(2))
LinearAlgebra.issymmetric(a::FaerMatrix{T}) where {T<:_ExtraScalar} =
    _native_call(RustCrate.predicate, a, Int32(3))
LinearAlgebra.ishermitian(a::FaerMatrix{T}) where {T<:_ExtraScalar} =
    _native_call(RustCrate.predicate, a, Int32(4))

const _FactorCodes=(
    lu = 0,
    full_lu = 1,
    qr = 2,
    pivoted_qr = 3,
    cholesky = 4,
    ldlt = 5,
    bunchkaufman = 6,
    svd = 7,
)
function _factor(
    a::FaerMatrix{T},
    ::Val{K},
    code;
    check = true,
    uplo = :L,
    team = nothing,
    rook::Bool = false,
) where {T<:_ExtraScalar,K}
    if K in (:cholesky, :ldlt, :bunchkaufman)
        LinearAlgebra.checksquare(a)
    end
    lock(_matrix_lock) do
        h=RustCrate.TypedFactor()
        code=_construction_code(K, code, rook)
        info=Int(_initialize_dense_factor(h, a, code, uplo===:L, team))
        if check && info!=0
            K===:cholesky && throw(LinearAlgebra.PosDefException(info))
            throw(LinearAlgebra.SingularException(info))
        end
        return FaerFactorization{T,K,typeof(h)}(h, size(a), info, uplo, rook)
    end
end
LinearAlgebra.lu(
    a::FaerMatrix{T};
    check::Bool = true,
    allowsingular::Bool = false,
    team = nothing,
) where {T<:_ExtraScalar} = _factor(a, Val(:lu), 0; check = check&&!allowsingular, team)
LinearAlgebra.lu(
    a::FaerMatrix{T},
    ::LinearAlgebra.RowMaximum;
    kwargs...,
) where {T<:_ExtraScalar} = LinearAlgebra.lu(a; kwargs...)
full_piv_lu(a::FaerMatrix{T}; check::Bool = true, team = nothing) where {T<:_ExtraScalar} =
    _factor(a, Val(:full_lu), 1; check, team)
LinearAlgebra.qr(a::FaerMatrix{T}; team = nothing) where {T<:_ExtraScalar} =
    _factor(a, Val(:qr), 2; team)
LinearAlgebra.qr(
    a::FaerMatrix{T},
    ::LinearAlgebra.NoPivot;
    team = nothing,
) where {T<:_ExtraScalar} = LinearAlgebra.qr(a; team)
LinearAlgebra.qr(
    a::FaerMatrix{T},
    ::LinearAlgebra.ColumnNorm;
    team = nothing,
) where {T<:_ExtraScalar} = _factor(a, Val(:pivoted_qr), 3; team)
const _ExtraHermitian=LinearAlgebra.Hermitian{T,<:FaerMatrix{T}} where {T<:_ExtraScalar}
const _ExtraSymmetric=LinearAlgebra.Symmetric{Float32,<:FaerMatrix{Float32}}
const _ExtraSelfAdjoint=Union{_ExtraHermitian,_ExtraSymmetric}
_uplo(a::_ExtraSelfAdjoint) = a.uplo=='L' ? :L : :U
function LinearAlgebra.cholesky(
    a::FaerMatrix{T};
    check::Bool = true,
    team = nothing,
) where {T<:_ExtraScalar}
    LinearAlgebra.ishermitian(a) ||
        return _non_hermitian_cholesky(a, check, RustCrate.TypedFactor)
    return _factor(a, Val(:cholesky), 4; check, team)
end
LinearAlgebra.cholesky(a::_ExtraSelfAdjoint; check::Bool = true, team = nothing) =
    _factor(parent(a), Val(:cholesky), 4; check, team, uplo = _uplo(a))
function _require_hermitian(a::FaerMatrix)
    LinearAlgebra.ishermitian(a) && return nothing
    message="self-adjoint factorization requires Hermitian input"
    eltype(a)<:Complex &&
        LinearAlgebra.issymmetric(a) &&
        (message*="; complex symmetric (non-Hermitian) LDLᵀ is not provided by t4a-faer")
    throw(ArgumentError(message))
end
function LinearAlgebra.ldlt(
    a::FaerMatrix{T};
    check::Bool = true,
    team = nothing,
) where {T<:_ExtraScalar}
    _require_hermitian(a)
    return _factor(a, Val(:ldlt), 5; check, team)
end
LinearAlgebra.ldlt(a::_ExtraSelfAdjoint; check::Bool = true, team = nothing) =
    _factor(parent(a), Val(:ldlt), 5; check, team, uplo = _uplo(a))
function LinearAlgebra.bunchkaufman(
    a::FaerMatrix{T},
    rook::Bool = false;
    check::Bool = true,
    team = nothing,
) where {T<:_ExtraScalar}
    _require_hermitian(a)
    return _factor(a, Val(:bunchkaufman), 6; check, team, rook)
end
LinearAlgebra.bunchkaufman(
    a::_ExtraSelfAdjoint,
    rook::Bool = false;
    check::Bool = true,
    team = nothing,
) = _factor(parent(a), Val(:bunchkaufman), 6; check, team, uplo = _uplo(a), rook)
_factor_part(f::FaerFactorization{T}, code, dims; team = nothing) where {T<:_ExtraScalar} =
    _typed_result(T, RustCrate.factor_part_from, dims, f, Int32(code), team)
FaerMatrix(f::FaerFactorization{T}; team = nothing) where {T<:_ExtraScalar} =
    _factor_part(f, 0, size(f); team)
function Base.copy(f::FaerFactorization{T,K}) where {T<:_ExtraScalar,K}
    lock(_matrix_lock) do
        h=RustCrate.TypedFactor()
        _native_call(RustCrate.clone_factor, h, f)
        return FaerFactorization{T,K,typeof(h)}(h, size(f), f.info, f.uplo, f.rook)
    end
end
function Base.deepcopy_internal(
    f::FaerFactorization{T},
    dict::IdDict,
) where {T<:_ExtraScalar}
    get!(dict, f) do ;
        copy(f)
    end
end
function _typed_solve(
    f::FaerFactorization{T,K},
    b::FaerMatrix{S},
    op = 0;
    team = nothing,
    scratch = nothing,
) where {T,K,S}
    (isnothing(team) && isnothing(scratch)) || _dense_solve_validate(f)
    transposed=op in (1, 2)
    size(b, 1)==size(f, transposed ? 2 : 1) ||
        throw(DimensionMismatch("factor and RHS rows must match"))
    LinearAlgebra.issuccess(f) || throw(LinearAlgebra.SingularException(f.info))
    _dense_rhs_validate(b)
    P=promote_type(T, S)
    isnothing(team) && isnothing(scratch) && P!==T && _dense_solve_validate(f)
    if T===Float64
        if S<:Complex
            data=Matrix(b)
            r=_factor_solve(f, FaerMatrix{Float64}(real.(data)), transposed; scratch, team)
            i=_factor_solve(f, FaerMatrix{Float64}(imag.(data)), transposed; scratch, team)
            return r+im*i
        end
        return _factor_solve(f, _coerce(b, Float64), transposed; scratch, team)
    elseif P!==T
        g=_factor(
            _promoted_factor_matrix(f, P; team),
            Val(K),
            getproperty(_FactorCodes, K);
            uplo = f.uplo,
            team,
            rook = f.rook,
        )
        try
            # Keep source precision when applying the promoted saved factor.
            return _dense_workspace_native_result(
                g,
                _coerce(b, P),
                op,
                scratch,
                team;
                precision = Float64(eps(_realtype(T))),
            )
        finally
            close(g)
        end
    end
    (!isnothing(scratch) || (K in (:lu, :full_lu) && size(f, 1)!=size(f, 2))) &&
        return _dense_workspace_native_result(f, _coerce(b, T), op, scratch, team)
    isnothing(team) || return _dense_team_native_result(f, _coerce(b, T), op, team)
    return _typed_result(
        T,
        RustCrate.factor_solve_from,
        (size(f, transposed ? 1 : 2), size(b, 2)),
        f,
        _coerce(b, T),
        Int32(op),
    )
end
Base.:\(f::FaerFactorization, b::FaerMatrix) = _typed_solve(f, b)
Base.:\(f::FaerFactorization{Float64}, b::AbstractMatrix{<:Complex}) = f\FaerMatrix(b)
Base.:\(f::FaerFactorization{Float64}, b::AbstractVector{<:Complex}) =
    vec(Matrix(f\FaerMatrix(reshape(b, :, 1))))
const _ExtraAdjointFactor=LinearAlgebra.AdjointFactorization{
    T,
    <:FaerFactorization{T},
} where {T<:_ExtraScalar}
const _ExtraTransposeFactor=LinearAlgebra.TransposeFactorization{
    T,
    <:FaerFactorization{T},
} where {T<:_ExtraScalar}
const _ExtraTransformedFactor=Union{_ExtraAdjointFactor,_ExtraTransposeFactor}
_solve_op(::_ExtraAdjointFactor) = 2
_solve_op(::_ExtraTransposeFactor) = 1
Base.:\(f::_ExtraTransformedFactor, b::FaerMatrix) =
    _typed_solve(parent(f), b, _solve_op(f))
function Base.inv(f::FaerFactorization{T}; team = nothing) where {T<:_ExtraScalar}
    LinearAlgebra.checksquare(f)
    LinearAlgebra.issuccess(f) || throw(LinearAlgebra.SingularException(f.info))
    return _typed_result(T, RustCrate.inverse_from, size(f), f, team)
end
Base.inv(a::FaerMatrix{T}; team = nothing) where {T<:_ExtraScalar} =
    inv(LinearAlgebra.lu(a; team); team)
LinearAlgebra.factorize(a::FaerMatrix{T}; team = nothing) where {T<:_ExtraScalar} =
    size(a, 1)==size(a, 2) ? LinearAlgebra.lu(a; team) :
    LinearAlgebra.qr(a, LinearAlgebra.ColumnNorm(); team)
Base.:\(a::FaerMatrix, b::FaerMatrix) = LinearAlgebra.factorize(a)\b
Base.:\(a::FaerMatrix{T}, b::AbstractMatrix{<:Number}) where {T<:_ExtraScalar} =
    a\FaerMatrix(b)
Base.:\(a::FaerMatrix{T}, b::AbstractVector{<:Number}) where {T<:_ExtraScalar} =
    vec(Matrix(a\FaerMatrix(reshape(b, :, 1))))
Base.:\(a::FaerMatrix{Float64}, b::AbstractMatrix{<:Complex}) = a\FaerMatrix(b)
Base.:\(a::FaerMatrix{Float64}, b::AbstractVector{<:Complex}) =
    vec(Matrix(a\FaerMatrix(reshape(b, :, 1))))
Base.:\(a::_AnyFaerTransposed, b::FaerMatrix) = copy(a)\b
Base.:/(a::FaerMatrix, b::FaerMatrix) = copy(adjoint(copy(adjoint(b))\copy(adjoint(a))))
Base.:/(a::FaerMatrix, f::FaerFactorization{T}) where {T<:_ExtraScalar} =
    copy(adjoint(adjoint(f)\copy(adjoint(a))))

LinearAlgebra.isposdef(a::FaerMatrix{T}) where {T<:_ExtraScalar} =
    LinearAlgebra.ishermitian(a)&&LinearAlgebra.issuccess(
        LinearAlgebra.cholesky(a; check = false),
    )
LinearAlgebra.isposdef(a::_ExtraSelfAdjoint) =
    LinearAlgebra.issuccess(LinearAlgebra.cholesky(a; check = false))
faer_mul(a::FaerMatrix, b::FaerMatrix) = a*b
faer_solve(a::FaerMatrix{T}, b::AbstractMatrix{<:Real}) where {T<:_ExtraScalar} = a\b
faer_solve(a::FaerMatrix{T}, b::AbstractVector{<:Real}) where {T<:_ExtraScalar} = a\b
faer_solve(a::FaerMatrix, b::AbstractMatrix{<:Complex}) = a\b
faer_solve(a::FaerMatrix, b::AbstractVector{<:Complex}) = a\b

# Intersections with LinearAlgebra's real-factor/complex-array optimizations and
# with the Float64 compatibility methods are explicit, avoiding ambiguous calls.
Base.:\(f::FaerFactorization{T}, b::FaerMatrix{S}) where {T<:_ExtraScalar,S<:Number} =
    _typed_solve(f, b)
Base.:\(f::FaerFactorization{Float64}, b::FaerMatrix{<:Real}) = _typed_solve(f, b)
Base.:\(f::FaerFactorization{Float64}, b::FaerMatrix{<:Complex}) = _typed_solve(f, b)
Base.:\(a::FaerMatrix{T}, b::FaerMatrix{S}) where {T<:_ExtraScalar,S<:Number} =
    _promoted_matrix_solve(a, b)
Base.:\(a::FaerMatrix{Float64}, b::FaerMatrix{<:Real}) = LinearAlgebra.factorize(a)\b
Base.:\(a::FaerMatrix{Float64}, b::FaerMatrix{<:Complex}) = LinearAlgebra.factorize(a)\b
Base.:\(a::_FaerTransposedMatrix, b::FaerMatrix{<:Real}) = copy(a)\b
faer_mul(a::FaerMatrix{<:Real}, b::FaerMatrix{<:Real}) = a*b
faer_solve(a::FaerMatrix{Float32}, b::AbstractMatrix{<:Real}) = a\b
faer_solve(a::FaerMatrix{Float32}, b::AbstractVector{<:Real}) = a\b
for T in (Float32, Float64), N in (1, 2)
    @eval Base.:\(f::FaerFactorization{$T}, b::Array{Complex{$T},$N}) =
        _complex_array_solve(f, b)
    for TF in (LinearAlgebra.AdjointFactorization, LinearAlgebra.TransposeFactorization)
        op=TF===LinearAlgebra.AdjointFactorization ? 2 : 1
        @eval Base.:\(f::$TF{$T,<:FaerFactorization{$T}}, b::Array{Complex{$T},$N}) =
            _complex_array_solve(parent(f), b, $op)
    end
end
function _complex_array_solve(f, b, op = 0)
    data=ndims(b)==1 ? reshape(b, :, 1) : b
    result=_typed_solve(f, FaerMatrix(data), op)
    return ndims(b)==1 ? vec(Matrix(result)) : result
end
for T in (Float32, Float64)
    @eval Base.:\(
        f::FaerFactorization{$T},
        b::Union{Vector{Complex{$T}},Matrix{Complex{$T}}},
    ) = _complex_array_solve(f, b)
    for TF in (LinearAlgebra.AdjointFactorization, LinearAlgebra.TransposeFactorization)
        op=TF===LinearAlgebra.AdjointFactorization ? 2 : 1
        @eval Base.:\(
            f::$TF{$T,<:FaerFactorization{$T}},
            b::Union{Vector{Complex{$T}},Matrix{Complex{$T}}},
        ) = _complex_array_solve(parent(f), b, $op)
    end
end

for T in (Float32, ComplexF32, ComplexF64)
    @eval begin
        Base.:\(f::FaerFactorization{$T}, b::AbstractMatrix{<:Number}) = f\FaerMatrix(b)
        Base.:\(f::FaerFactorization{$T}, b::AbstractVector{<:Number}) =
            vec(Matrix(f\FaerMatrix(reshape(b, :, 1))))
        Base.:\(f::FaerFactorization{$T}, b::FaerMatrix{<:Number}) = _typed_solve(f, b)
    end
    for TF in (LinearAlgebra.AdjointFactorization, LinearAlgebra.TransposeFactorization)
        op=TF===LinearAlgebra.AdjointFactorization ? 2 : 1
        @eval begin
            Base.:\(f::$TF{$T,<:FaerFactorization{$T}}, b::AbstractMatrix{<:Number}) =
                f\FaerMatrix(b)
            Base.:\(f::$TF{$T,<:FaerFactorization{$T}}, b::AbstractVector{<:Number}) =
                vec(Matrix(f\FaerMatrix(reshape(b, :, 1))))
            Base.:\(f::$TF{$T,<:FaerFactorization{$T}}, b::FaerMatrix{<:Number}) =
                _typed_solve(parent(f), b, $op)
        end
    end
end
function _promoted_matrix_solve(a::FaerMatrix{T}, b::FaerMatrix{S}) where {T,S}
    P=promote_type(T, S)
    return LinearAlgebra.factorize(_coerce(a, P))\_coerce(b, P)
end
function _promoted_factor_matrix(
    f::FaerFactorization{T,K},
    ::Type{P};
    team = nothing,
) where {T,K,P}
    product(a, b) = isnothing(team) ? a*b : team_mul(team, a, b)
    if K in (:lu, :full_lu)
        a=product(_coerce(f.L, P), _coerce(f.U, P))
        return K===:lu ? FaerMatrix{P}(a[invperm(f.p), :]) :
               FaerMatrix{P}(a[invperm(f.p), invperm(f.q)])
    elseif K in (:qr, :pivoted_qr)
        a=product(_coerce(f.Qthin, P), _coerce(f.R, P))
        return K===:qr ? a : FaerMatrix{P}(a[:, invperm(f.p)])
    else
        l=_coerce(f.L, P)
        a=K===:cholesky ? product(l, copy(l')) :
          product(product(l, _coerce(f.D, P)), copy(l'))
        return K===:bunchkaufman ? FaerMatrix{P}(a[invperm(f.p), invperm(f.p)]) : a
    end
end

function LinearAlgebra.dot(a::FaerMatrix{T}, b::FaerMatrix{T}) where {T<:_ExtraScalar}
    size(a)==size(b) || return _native_dot(a, b)
    value=Ref{T}()
    GC.@preserve value _native_call(
        RustCrate.dot_buffer,
        a,
        b,
        Ptr{UInt8}(Base.unsafe_convert(Ptr{T}, value)),
    )
    return value[]
end
function LinearAlgebra.dot(a::FaerMatrix{T}, b::FaerMatrix{S}) where {T,S}
    P=promote_type(T, S)
    return LinearAlgebra.dot(_coerce(a, P), _coerce(b, P))
end
function _typed_eigen(
    a::FaerMatrix{T};
    b = nothing,
    vectors = true,
    hermitian = false,
    lower = true,
    other_lower = true,
    sortby = _eigensort,
) where {T<:_ExtraScalar}
    n=LinearAlgebra.checksquare(a)
    isnothing(b) ||
        size(b)==size(a) ||
        throw(DimensionMismatch("generalized eigen input dimensions must match"))
    R=_realtype(T)
    V=hermitian ? R : Complex{R}
    U=hermitian ? T : Complex{R}
    values=Vector{V}(undef, n)
    eigenvectors=Matrix{U}(undef, n, vectors ? n : 0)
    GC.@preserve values eigenvectors begin
        if isnothing(b)
            _native_call(
                RustCrate.eigen_typed_buffers,
                a,
                Ptr{UInt8}(pointer(values)),
                Ptr{UInt8}(pointer(eigenvectors)),
                vectors,
                hermitian,
                lower,
            )
        else
            eltype(b)===T || throw(
                ArgumentError(
                    "internal generalized eigen inputs must have matching scalar types",
                ),
            )
            _native_call(
                RustCrate.generalized_typed_buffers,
                a,
                b,
                Ptr{UInt8}(pointer(values)),
                Ptr{UInt8}(pointer(eigenvectors)),
                vectors,
                hermitian,
                lower,
                other_lower,
            )
        end
    end
    if !isnothing(sortby)
        order=sortperm(values; by = sortby)
        values=values[order]
        vectors && (eigenvectors=eigenvectors[:, order])
    end
    return vectors ? LinearAlgebra.Eigen(values, eigenvectors) : values
end
LinearAlgebra.eigen(
    a::FaerMatrix{T};
    side::Symbol = :right,
    sortby = _eigensort,
    scratch = nothing,
    team = nothing,
    permute::Bool = false,
    scale::Bool = false,
) where {T<:_ExtraScalar} =
    _balanced_check(permute, scale) ?
    _balanced_eigen(a, true; side, sortby, scratch, team) :
    side===:right && _general_evd_default_path(scratch, team) ? _typed_eigen(a; sortby) :
    _general_evd_workspace_result(a; sortby, scratch, team, side)
LinearAlgebra.eigvals(
    a::FaerMatrix{T};
    sortby = _eigensort,
    scratch = nothing,
    team = nothing,
    permute::Bool = false,
    scale::Bool = false,
) where {T<:_ExtraScalar} =
    _balanced_check(permute, scale) ? _balanced_eigen(a, false; sortby, scratch, team) :
    _general_evd_default_path(scratch, team) ? _typed_eigen(a; vectors = false, sortby) :
    _general_evd_workspace_result(a; vectors = false, sortby, scratch, team)
LinearAlgebra.eigen(
    a::_ExtraSelfAdjoint;
    native::Bool = false,
    sortby = _eigensort,
    scratch = nothing,
    team = nothing,
) =
    native ?
    _owned_spectral(parent(a); eigen = true, lower = _uplo(a)===:L, sortby, scratch, team) :
    _evd_default_path(scratch, team) ?
    _typed_eigen(parent(a); hermitian = true, lower = _uplo(a)===:L, sortby) :
    _evd_workspace_result(a; sortby, scratch, team)
LinearAlgebra.eigvals(
    a::_ExtraSelfAdjoint;
    sortby = _eigensort,
    scratch = nothing,
    team = nothing,
) =
    _evd_default_path(scratch, team) ?
    _typed_eigen(
        parent(a);
        vectors = false,
        hermitian = true,
        lower = _uplo(a)===:L,
        sortby,
    ) : _evd_workspace_result(a; vectors = false, sortby, scratch, team)
function LinearAlgebra.eigen(
    a::FaerMatrix{T},
    b::FaerMatrix{T};
    homogeneous::Bool = false,
    side::Symbol = :right,
    sortby = homogeneous ? nothing : _eigensort,
    scratch = nothing,
    team = nothing,
) where {T<:_ExtraScalar}
    return _general_gevd_workspace_result(a, b; sortby, scratch, team, homogeneous, side)
end
LinearAlgebra.eigvals(
    a::FaerMatrix{T},
    b::FaerMatrix{T};
    homogeneous::Bool = false,
    sortby = homogeneous ? nothing : _eigensort,
    scratch = nothing,
    team = nothing,
) where {T<:_ExtraScalar} = _general_gevd_workspace_result(
    a,
    b;
    vectors = false,
    sortby,
    scratch,
    team,
    homogeneous,
)
function LinearAlgebra.eigen(a::FaerMatrix{T}, b::FaerMatrix{S}; kwargs...) where {T,S}
    P=promote_type(T, S)
    return LinearAlgebra.eigen(_coerce(a, P), _coerce(b, P); kwargs...)
end
function LinearAlgebra.eigvals(a::FaerMatrix{T}, b::FaerMatrix{S}; kwargs...) where {T,S}
    P=promote_type(T, S)
    return LinearAlgebra.eigvals(_coerce(a, P), _coerce(b, P); kwargs...)
end
const _AnyFaerSelfAdjoint=Union{_FaerSelfAdjoint,_ExtraSelfAdjoint}
function LinearAlgebra.eigen(
    a::_ExtraSelfAdjoint,
    b::_ExtraSelfAdjoint;
    sortby = _eigensort,
    scratch = nothing,
    team = nothing,
)
    return _selfadj_gevd_workspace_result(a, b; sortby, scratch, team)
end
function LinearAlgebra.eigvals(
    a::_ExtraSelfAdjoint,
    b::_ExtraSelfAdjoint;
    sortby = _eigensort,
    scratch = nothing,
    team = nothing,
)
    return _selfadj_gevd_workspace_result(a, b; vectors = false, sortby, scratch, team)
end
function LinearAlgebra.eigen(a::_AnyFaerSelfAdjoint, b::_AnyFaerSelfAdjoint; kwargs...)
    P=promote_type(eltype(a), eltype(b))
    return LinearAlgebra.eigen(
        LinearAlgebra.Hermitian(_coerce(parent(a), P), _uplo(a)),
        LinearAlgebra.Hermitian(_coerce(parent(b), P), _uplo(b));
        kwargs...,
    )
end
function LinearAlgebra.eigvals(a::_AnyFaerSelfAdjoint, b::_AnyFaerSelfAdjoint; kwargs...)
    P=promote_type(eltype(a), eltype(b))
    return LinearAlgebra.eigvals(
        LinearAlgebra.Hermitian(_coerce(parent(a), P), _uplo(a)),
        LinearAlgebra.Hermitian(_coerce(parent(b), P), _uplo(b));
        kwargs...,
    )
end
function pfaffian(
    a::FaerMatrix{T};
    scratch = nothing,
    team = nothing,
) where {T<:_ExtraScalar}
    (!isnothing(scratch) || !isnothing(team)) && return _pfaffian_options(a, scratch, team)
    value=Ref{T}()
    GC.@preserve value _native_call(
        RustCrate.pfaffian_buffer,
        a,
        Ptr{UInt8}(Base.unsafe_convert(Ptr{T}, value)),
    )
    return value[]
end

const _ExtraTriangular=Union{
    LinearAlgebra.LowerTriangular{T,<:FaerMatrix{T}},
    LinearAlgebra.UpperTriangular{T,<:FaerMatrix{T}},
    LinearAlgebra.UnitLowerTriangular{T,<:FaerMatrix{T}},
    LinearAlgebra.UnitUpperTriangular{T,<:FaerMatrix{T}},
} where {T<:_ExtraScalar}
function _triangular_wrap(a, b)
    return _islower(a) ?
           (
        _isunit(a) ? LinearAlgebra.UnitLowerTriangular(b) :
        LinearAlgebra.LowerTriangular(b)
    ) :
           (
        _isunit(a) ? LinearAlgebra.UnitUpperTriangular(b) :
        LinearAlgebra.UpperTriangular(b)
    )
end
Base.inv(a::_ExtraTriangular) = _typed_result(
    eltype(a),
    RustCrate.triangular_inverse_typed,
    size(a),
    parent(a),
    _islower(a),
    _isunit(a),
)
function LinearAlgebra.logabsdet(f::FaerFactorization{T}) where {T<:_ExtraScalar}
    R=_realtype(T)
    value=Ref{R}()
    phase=Ref{T}()
    GC.@preserve value phase _native_call(
        RustCrate.determinant_parts,
        f,
        Ptr{UInt8}(Base.unsafe_convert(Ptr{R}, value)),
        Ptr{UInt8}(Base.unsafe_convert(Ptr{T}, phase)),
    )
    return (value[], phase[])
end
function LinearAlgebra.det(f::FaerFactorization{T}) where {T<:_ExtraScalar}
    value, phase=LinearAlgebra.logabsdet(f)
    return exp(value)*phase
end
function LinearAlgebra.logdet(f::FaerFactorization{T}) where {T<:_ExtraScalar}
    value, phase=LinearAlgebra.logabsdet(f)
    T<:Real &&
        phase<0 &&
        throw(DomainError(phase, "real logdet requires a nonnegative determinant"))
    return iszero(phase) ? T(-Inf) : value+log(phase)
end
LinearAlgebra.logabsdet(a::FaerMatrix{T}) where {T<:_ExtraScalar} =
    LinearAlgebra.logabsdet(LinearAlgebra.lu(a; check = false))
LinearAlgebra.logdet(a::FaerMatrix{T}) where {T<:_ExtraScalar} =
    LinearAlgebra.logdet(LinearAlgebra.lu(a; check = false))
function Base.sum(::typeof(abs2), a::FaerMatrix{T}) where {T<:_ExtraScalar}
    R=_realtype(T)
    value=Ref{R}()
    GC.@preserve value _native_call(
        RustCrate.norm_reduction,
        a,
        Ptr{UInt8}(Base.unsafe_convert(Ptr{R}, value)),
        Int32(3),
    )
    return value[]
end
Base.sum(::typeof(abs), a::FaerMatrix{T}) where {T<:_ExtraScalar} = LinearAlgebra.norm(a, 1)
