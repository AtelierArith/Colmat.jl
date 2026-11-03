"""
    FaerFactorization

A reusable t4a-faer factorization, returned by `lu`, `qr`, `cholesky`, `ldlt`,
`bunchkaufman` and `full_piv_lu` on `FaerMatrix`. Numerical factors are owned in
Rust and independent of the input. `F \\ B` reuses them; factor properties return
independent FaerMatrix objects. `close(F)` invalidates only this factorization.
"""
mutable struct FaerFactorization{T,K,H} <: LinearAlgebra.Factorization{T}
    handle::Union{Nothing,H}
    dims::Tuple{Int,Int}
    info::Int
    uplo::Symbol
end

_native_argument(f::FaerFactorization) = _handle(f)
_is_native_owner(::FaerFactorization) = true
function _handle(f::FaerFactorization)
    h = getfield(f, :handle)
    isnothing(h) && throw(ArgumentError("FaerFactorization has been closed"))
    return h
end
Base.size(f::FaerFactorization) = getfield(f, :dims)
Base.size(f::FaerFactorization, d::Integer) =
    d < 1 ? throw(ArgumentError("dimension must be positive")) : d <= 2 ? size(f)[d] : 1
Base.isopen(f::FaerFactorization) =
    lock(() -> !isnothing(getfield(f, :handle)), _matrix_lock)
function Base.close(f::FaerFactorization)
    lock(_matrix_lock) do
        f.handle = nothing
    end
    return nothing
end
LinearAlgebra.issuccess(f::FaerFactorization) = getfield(f, :info) == 0

function _factor(a::FaerMatrix{Float64}, ::Val{K}, code; check = true, uplo = :L) where {K}
    if K in (:cholesky, :ldlt, :bunchkaufman)
        size(a, 1) == size(a, 2) ||
            throw(DimensionMismatch("factorization requires a square matrix"))
    end
    lock(_matrix_lock) do
        h = RustCrate.OwnedFactor()
        info = Int(_native_call(RustCrate.initialize, h, a, Int32(code), uplo === :L))
        if check && info != 0
            K === :cholesky && throw(LinearAlgebra.PosDefException(info))
            throw(LinearAlgebra.SingularException(info))
        end
        return FaerFactorization{Float64,K,typeof(h)}(h, size(a), info, uplo)
    end
end

"""Factor a Rust-owned matrix using t4a-faer's partial row-pivot LU."""
LinearAlgebra.lu(a::FaerMatrix{Float64}; check::Bool = true, allowsingular::Bool = false) =
    _factor(a, Val(:lu), 0; check = check && !allowsingular)
LinearAlgebra.lu(a::FaerMatrix{Float64}, ::LinearAlgebra.RowMaximum; kwargs...) =
    LinearAlgebra.lu(a; kwargs...)

"""
    full_piv_lu(A::FaerMatrix{Float64}; check=true)

Compute t4a-faer's full row-and-column-pivot LU. The permutation vectors satisfy
`Matrix(A)[F.p,F.q] ≈ Matrix(F.L*F.U)`.
"""
full_piv_lu(a::FaerMatrix{Float64}; check::Bool = true) =
    _factor(a, Val(:full_lu), 1; check)

"""Compute a reusable Rust-owned QR factorization, optionally with column pivoting."""
LinearAlgebra.qr(a::FaerMatrix{Float64}) = _factor(a, Val(:qr), 2)
LinearAlgebra.qr(a::FaerMatrix{Float64}, ::LinearAlgebra.NoPivot) = LinearAlgebra.qr(a)
LinearAlgebra.qr(a::FaerMatrix{Float64}, ::LinearAlgebra.ColumnNorm) =
    _factor(a, Val(:pivoted_qr), 3)

const _FaerSelfAdjoint = Union{
    LinearAlgebra.Symmetric{Float64,<:FaerMatrix{Float64}},
    LinearAlgebra.Hermitian{Float64,<:FaerMatrix{Float64}},
}
_uplo(a::_FaerSelfAdjoint) = a.uplo == 'L' ? :L : :U
function LinearAlgebra.cholesky(a::FaerMatrix{Float64}; check::Bool = true)
    LinearAlgebra.issymmetric(a) || throw(LinearAlgebra.PosDefException(-1))
    return _factor(a, Val(:cholesky), 4; check)
end
LinearAlgebra.cholesky(a::_FaerSelfAdjoint; check::Bool = true) =
    _factor(parent(a), Val(:cholesky), 4; check, uplo = _uplo(a))
function LinearAlgebra.ldlt(a::FaerMatrix{Float64}; check::Bool = true)
    LinearAlgebra.issymmetric(a) || throw(ArgumentError("LDLT requires self-adjoint input"))
    return _factor(a, Val(:ldlt), 5; check)
end
LinearAlgebra.ldlt(a::_FaerSelfAdjoint; check::Bool = true) =
    _factor(parent(a), Val(:ldlt), 5; check, uplo = _uplo(a))
function LinearAlgebra.bunchkaufman(a::FaerMatrix{Float64}; check::Bool = true)
    LinearAlgebra.issymmetric(a) ||
        throw(ArgumentError("block LDLT requires self-adjoint input"))
    return _factor(a, Val(:bunchkaufman), 6; check)
end
LinearAlgebra.bunchkaufman(a::_FaerSelfAdjoint; check::Bool = true) =
    _factor(parent(a), Val(:bunchkaufman), 6; check, uplo = _uplo(a))

function _permutation(f::FaerFactorization{T,K}, column::Bool = false) where {T,K}
    n = size(f, column || K === :pivoted_qr ? 2 : 1)
    p = Vector{UInt}(undef, n)
    GC.@preserve p _native_call(RustCrate.permutation, f, pointer(p), UInt(n), column)
    return Int.(p)
end
_factor_part(f::FaerFactorization{Float64}, code, dims) =
    _matrix_result(RustCrate.factor_part, dims, f, Int32(code))
function Base.getproperty(f::FaerFactorization{T,K}, name::Symbol) where {T,K}
    m, n = size(f)
    k = min(m, n)
    if name === :packed && K in (:lu, :full_lu, :qr, :pivoted_qr, :cholesky)
        return _factor_part(f, 7, size(f))
    elseif name === :Qbasis && K in (:qr, :pivoted_qr)
        return _factor_part(f, 8, (m, k))
    elseif name === :Qcoeff && K in (:qr, :pivoted_qr)
        rows=Int(_native_call(RustCrate.qr_blocksize, f))
        return _factor_part(f, 9, (rows, k))
    elseif name === :L && K in (:lu, :full_lu, :cholesky, :ldlt, :bunchkaufman)
        return _factor_part(f, 1, (m, k))
    elseif name === :U
        K === :cholesky && return copy(adjoint(f.L))
        K in (:lu, :full_lu) && return _factor_part(f, 2, (k, n))
    elseif name === :Q && K in (:qr, :pivoted_qr)
        return _factor_part(f, 3, (m, m))
    elseif name === :Qthin && K in (:qr, :pivoted_qr)
        return _factor_part(f, 6, (m, k))
    elseif name === :R && K in (:qr, :pivoted_qr)
        return _factor_part(f, 4, (k, n))
    elseif name === :D && K in (:ldlt, :bunchkaufman)
        return _factor_part(f, 5, (n, n))
    elseif name === :p && K in (:lu, :full_lu, :bunchkaufman, :pivoted_qr)
        return _permutation(f)
    elseif name === :q && K === :full_lu
        return _permutation(f, true)
    elseif name === :P && K in (:lu, :full_lu, :bunchkaufman)
        return FaerMatrix(Matrix{T}(LinearAlgebra.I, m, m)[f.p, :])
    end
    return getfield(f, name)
end
function Base.propertynames(f::FaerFactorization{T,K}, private::Bool = false) where {T,K}
    props =
        K in (:lu, :full_lu) ? (:L, :U, :p, :P) :
        K in (:qr, :pivoted_qr) ? (:Q, :Qthin, :R) : K === :cholesky ? (:L, :U) : (:L, :D)
    K in (:lu, :full_lu, :qr, :pivoted_qr, :cholesky) && (props=(props..., :packed))
    K in (:qr, :pivoted_qr) && (props=(props..., :Qbasis, :Qcoeff))
    K === :full_lu && (props = (props..., :q))
    K in (:pivoted_qr, :bunchkaufman) && (props = (props..., :p))
    return private ? (props..., :handle, :dims, :info, :uplo) : (props..., :info, :uplo)
end

"""Reconstruct a new Rust-owned matrix from a reusable faer factorization."""
FaerMatrix(f::FaerFactorization{Float64}) = _factor_part(f, 0, size(f))
Base.Matrix(f::FaerFactorization) = Matrix(FaerMatrix(f))
Base.Array(f::FaerFactorization) = Matrix(f)
function Base.copy(f::FaerFactorization{Float64,K}) where {K}
    lock(_matrix_lock) do
        h = RustCrate.OwnedFactor()
        _native_call(RustCrate.clone_factor, h, f)
        return FaerFactorization{Float64,K,typeof(h)}(h, size(f), f.info, f.uplo)
    end
end
function Base.deepcopy_internal(f::FaerFactorization{Float64}, dict::IdDict)
    get!(dict, f) do
        copy(f)
    end
end

function _factor_solve(
    f::FaerFactorization{Float64},
    b::FaerMatrix{Float64},
    transposed = false,
)
    size(b, 1) == size(f, transposed ? 2 : 1) ||
        throw(DimensionMismatch("factor and RHS rows do not match"))
    LinearAlgebra.issuccess(f) || throw(LinearAlgebra.SingularException(f.info))
    return _matrix_result(
        RustCrate.factor_solve,
        (size(f, transposed ? 1 : 2), size(b, 2)),
        f,
        b,
        transposed,
    )
end
Base.:\(f::FaerFactorization{Float64}, b::FaerMatrix{Float64}) = _factor_solve(f, b)
Base.:\(f::FaerFactorization{Float64}, b::AbstractMatrix{<:Real}) =
    f \ FaerMatrix{Float64}(b)
Base.:\(f::FaerFactorization{Float64}, b::AbstractVector{<:Real}) =
    vec(Matrix(f \ FaerMatrix(reshape(b, :, 1))))
const _FaerTransposedFactor = Union{
    LinearAlgebra.TransposeFactorization{Float64,<:FaerFactorization{Float64}},
    LinearAlgebra.AdjointFactorization{Float64,<:FaerFactorization{Float64}},
}
Base.:\(f::_FaerTransposedFactor, b::FaerMatrix{Float64}) =
    _factor_solve(parent(f), b, true)
Base.:\(f::_FaerTransposedFactor, b::AbstractMatrix{<:Real}) = f \ FaerMatrix{Float64}(b)
Base.:\(f::_FaerTransposedFactor, b::AbstractVector{<:Real}) =
    vec(Matrix(f \ FaerMatrix(reshape(b, :, 1))))

function LinearAlgebra.ldiv!(
    dest::FaerMatrix{Float64},
    f::Union{FaerFactorization{Float64},_FaerTransposedFactor},
    b::FaerMatrix{Float64},
)
    size(dest) == (size(f, 2), size(b, 2)) ||
        throw(DimensionMismatch("output size does not match solve"))
    return copyto!(dest, f \ b)
end
function LinearAlgebra.ldiv!(
    f::Union{FaerFactorization{Float64},_FaerTransposedFactor},
    b::FaerMatrix{Float64},
)
    size(f, 1) == size(f, 2) ||
        throw(DimensionMismatch("two-argument ldiv! requires a square factor"))
    return LinearAlgebra.ldiv!(b, f, b)
end
function _ldiv_array!(dest, f, b)
    result = f \ b
    size(dest) == size(result) ||
        throw(DimensionMismatch("output size does not match solve"))
    copyto!(
        dest,
        dest isa FaerMatrix && result isa FaerMatrix ? result :
        result isa FaerMatrix ? Matrix(result) : result,
    )
    return dest
end

LinearAlgebra.ldiv!(
    dest::AbstractMatrix,
    f::FaerFactorization{Float64},
    b::AbstractMatrix,
) = _ldiv_array!(dest, f, b)
LinearAlgebra.ldiv!(
    dest::AbstractVector,
    f::FaerFactorization{Float64},
    b::AbstractVector,
) = _ldiv_array!(dest, f, b)
LinearAlgebra.ldiv!(
    dest::FaerMatrix{Float64},
    f::FaerFactorization{Float64},
    b::FaerMatrix{Float64},
) = _ldiv_array!(dest, f, b)

function Base.inv(f::FaerFactorization{Float64})
    size(f, 1)==size(f, 2) || throw(DimensionMismatch("inverse requires a square factor"))
    LinearAlgebra.issuccess(f) || throw(LinearAlgebra.SingularException(f.info))
    return _matrix_result(RustCrate.factor_inverse, size(f), f)
end
Base.inv(a::FaerMatrix{Float64}) = inv(LinearAlgebra.lu(a))
LinearAlgebra.factorize(a::FaerMatrix{Float64}) =
    size(a, 1)==size(a, 2) ? LinearAlgebra.lu(a) :
    LinearAlgebra.qr(a, LinearAlgebra.ColumnNorm())

function LinearAlgebra.logabsdet(f::FaerFactorization{Float64})
    out=Vector{Float64}(undef, 2)
    GC.@preserve out _native_call(RustCrate.logabsdet_buffer, f, pointer(out))
    return (out[1], out[2])
end
function LinearAlgebra.det(f::FaerFactorization{Float64})
    value, sign=LinearAlgebra.logabsdet(f)
    return sign*exp(value)
end
function LinearAlgebra.logdet(f::FaerFactorization{Float64})
    value, sign=LinearAlgebra.logabsdet(f)
    sign<0 && throw(DomainError(sign, "real logdet requires a nonnegative determinant"))
    return value
end
LinearAlgebra.logabsdet(a::FaerMatrix{Float64}) =
    LinearAlgebra.logabsdet(LinearAlgebra.lu(a; check = false))
LinearAlgebra.logdet(a::FaerMatrix{Float64}) =
    LinearAlgebra.logdet(LinearAlgebra.lu(a; check = false))
LinearAlgebra.isposdef(a::FaerMatrix{Float64}) =
    LinearAlgebra.issymmetric(a) &&
    LinearAlgebra.issuccess(LinearAlgebra.cholesky(a; check = false))
LinearAlgebra.isposdef(a::_FaerSelfAdjoint) =
    LinearAlgebra.issuccess(LinearAlgebra.cholesky(a; check = false))
