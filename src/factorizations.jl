"""
    FaerFactorization

A reusable t4a-faer factorization, returned by `lu`, `qr`, `cholesky`, `ldlt`,
`bunchkaufman` and `full_piv_lu` on `FaerMatrix`. Numerical factors are owned in
Rust and independent of the input. `F \\ B` reuses them; factor properties return
independent FaerMatrix objects. `close(F)` invalidates only this factorization.

Construction accepts `team=FaerTeam(...)` to borrow a reusable native team.
Factors own their data and remain usable after the team or input closes.
`ldiv!` and `rdiv!` accept a solve-time `team=` independently of construction.
Property extraction and ordinary `F \\ B` remain sequential.
"""
mutable struct FaerFactorization{T,K,H} <: LinearAlgebra.Factorization{T}
    handle::Union{Nothing,H}
    dims::Tuple{Int,Int}
    info::Int
    uplo::Symbol
    rook::Bool
end
FaerFactorization{T,K,H}(handle, dims, info, uplo) where {T,K,H} =
    FaerFactorization{T,K,H}(handle, dims, info, uplo, false)

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

function _initialize_dense_factor(h, a, code, lower, team)
    isnothing(team) || team isa FaerTeam || throw(ArgumentError("team must be a FaerTeam"))
    isnothing(team) || _handle(team)
    args=(h, a, Int32(code), lower, UInt(size(a, 1)), UInt(size(a, 2)), _tag(eltype(a)))
    return isnothing(team) ? _native_call(RustCrate.initialize, args...) :
           _native_call(RustCrate.initialize_team, args..., team)
end

function _factor(
    a::FaerMatrix{Float64},
    ::Val{K},
    code;
    check = true,
    uplo = :L,
    team = nothing,
    rook::Bool = false,
) where {K}
    if K in (:cholesky, :ldlt, :bunchkaufman)
        size(a, 1) == size(a, 2) ||
            throw(DimensionMismatch("factorization requires a square matrix"))
    end
    lock(_matrix_lock) do
        h = RustCrate.OwnedFactor()
        code = _construction_code(K, code, rook)
        info = Int(_initialize_dense_factor(h, a, code, uplo === :L, team))
        if check && info != 0
            K === :cholesky && throw(LinearAlgebra.PosDefException(info))
            throw(LinearAlgebra.SingularException(info))
        end
        return FaerFactorization{Float64,K,typeof(h)}(h, size(a), info, uplo, rook)
    end
end
# Native construction code 8 selects faer's rook strategy for block LDLᴴ.
_construction_code(K, code, rook) = K === :bunchkaufman && rook ? 8 : code

# Julia's `cholesky(A; check=false)` returns an unsuccessful factor with
# `info == -1` for non-Hermitian input. No numerical factor is stored.
function _non_hermitian_cholesky(a::FaerMatrix{T}, check::Bool, handle) where {T}
    LinearAlgebra.checksquare(a)
    check && throw(LinearAlgebra.PosDefException(-1))
    lock(_matrix_lock) do
        h = handle()
        return FaerFactorization{T,:cholesky,typeof(h)}(h, size(a), -1, :L, false)
    end
end

"""Factor a Rust-owned matrix using t4a-faer's partial row-pivot LU."""
LinearAlgebra.lu(
    a::FaerMatrix{Float64};
    check::Bool = true,
    allowsingular::Bool = false,
    team = nothing,
) = _factor(a, Val(:lu), 0; check = check && !allowsingular, team)
LinearAlgebra.lu(a::FaerMatrix{Float64}, ::LinearAlgebra.RowMaximum; kwargs...) =
    LinearAlgebra.lu(a; kwargs...)

"""
    full_piv_lu(A::FaerMatrix; check=true, team=nothing)

Compute t4a-faer's full row-and-column-pivot LU. The permutation vectors satisfy
`Matrix(A)[F.p,F.q] ≈ Matrix(F.L*F.U)`.
"""
full_piv_lu(a::FaerMatrix{Float64}; check::Bool = true, team = nothing) =
    _factor(a, Val(:full_lu), 1; check, team)

"""Compute a reusable Rust-owned QR factorization, optionally with column pivoting."""
LinearAlgebra.qr(a::FaerMatrix{Float64}; team = nothing) = _factor(a, Val(:qr), 2; team)
LinearAlgebra.qr(a::FaerMatrix{Float64}, ::LinearAlgebra.NoPivot; team = nothing) =
    LinearAlgebra.qr(a; team)
LinearAlgebra.qr(a::FaerMatrix{Float64}, ::LinearAlgebra.ColumnNorm; team = nothing) =
    _factor(a, Val(:pivoted_qr), 3; team)

const _FaerSelfAdjoint = Union{
    LinearAlgebra.Symmetric{Float64,<:FaerMatrix{Float64}},
    LinearAlgebra.Hermitian{Float64,<:FaerMatrix{Float64}},
}
_uplo(a::_FaerSelfAdjoint) = a.uplo == 'L' ? :L : :U
function LinearAlgebra.cholesky(a::FaerMatrix{Float64}; check::Bool = true, team = nothing)
    LinearAlgebra.issymmetric(a) ||
        return _non_hermitian_cholesky(a, check, RustCrate.OwnedFactor)
    return _factor(a, Val(:cholesky), 4; check, team)
end
LinearAlgebra.cholesky(a::_FaerSelfAdjoint; check::Bool = true, team = nothing) =
    _factor(parent(a), Val(:cholesky), 4; check, team, uplo = _uplo(a))
function LinearAlgebra.ldlt(a::FaerMatrix{Float64}; check::Bool = true, team = nothing)
    LinearAlgebra.issymmetric(a) || throw(ArgumentError("LDLT requires self-adjoint input"))
    return _factor(a, Val(:ldlt), 5; check, team)
end
LinearAlgebra.ldlt(a::_FaerSelfAdjoint; check::Bool = true, team = nothing) =
    _factor(parent(a), Val(:ldlt), 5; check, team, uplo = _uplo(a))
"""
    bunchkaufman(A::FaerMatrix, rook=false; check=true, team=nothing)

Compute t4a-faer's pivoted block LDLᴴ of Hermitian `A` (or a Hermitian/real
Symmetric wrapper). `rook=true` selects faer's rook pivoting; `false` keeps
faer's default partial-diagonal Bunch–Kaufman. `check=false` returns a factor
with `info > 0` for a singular block diagonal. Properties: `L`, `D`, `p`, `P`,
`packed`, `rook`, `uplo`; `inertia(F)` counts eigenvalue signs.
"""
function LinearAlgebra.bunchkaufman(
    a::FaerMatrix{Float64},
    rook::Bool = false;
    check::Bool = true,
    team = nothing,
)
    LinearAlgebra.issymmetric(a) ||
        throw(ArgumentError("block LDLT requires self-adjoint input"))
    return _factor(a, Val(:bunchkaufman), 6; check, team, rook)
end
LinearAlgebra.bunchkaufman(
    a::_FaerSelfAdjoint,
    rook::Bool = false;
    check::Bool = true,
    team = nothing,
) = _factor(parent(a), Val(:bunchkaufman), 6; check, team, uplo = _uplo(a), rook)

function _permutation(f::FaerFactorization{T,K}, column::Bool = false) where {T,K}
    n = size(f, column || K === :pivoted_qr ? 2 : 1)
    p = Vector{UInt}(undef, n)
    GC.@preserve p _native_call(RustCrate.permutation, f, pointer(p), UInt(n), column)
    return Int.(p)
end
# Reconstruction (code 0) and explicit Q (3, 6) borrow `team`; other parts copy.
_factor_part(f::FaerFactorization{Float64}, code, dims; team = nothing) =
    _matrix_result(RustCrate.factor_part, dims, f, Int32(code), team)
function Base.getproperty(f::FaerFactorization{T,K}, name::Symbol) where {T,K}
    m, n = size(f)
    k = min(m, n)
    if name === :packed &&
       K in (:lu, :full_lu, :qr, :pivoted_qr, :cholesky, :ldlt, :bunchkaufman)
        return _factor_part(f, 7, size(f))
    elseif name === :Qbasis && K in (:qr, :pivoted_qr)
        return _factor_part(f, 8, (m, k))
    elseif name === :Qcoeff && K in (:qr, :pivoted_qr)
        rows=Int(_native_call(RustCrate.qr_blocksize, f))
        return _factor_part(f, 9, (rows, k))
    elseif name === :L && K in (:lu, :full_lu, :cholesky, :ldlt, :bunchkaufman)
        return _factor_part(f, 1, (m, k))
    elseif name === :UL && K === :cholesky
        return getfield(f, :uplo) === :U ? f.U : f.L
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
"""
    qr_q(F; thin=false, team=nothing)

Form the explicit orthogonal/unitary `Q` of a dense QR factor (`qr(A)`,
`qr(A, ColumnNorm())` or the in-place `qr!(A; scratch)`) as a new `FaerMatrix`:
the square `m×m` `F.Q`, or with `thin=true` the `m×min(m,n)` `F.Qthin`. Property
access cannot take keywords; `qr_q` lends a caller-owned `FaerTeam` to faer's
blocked Householder application that builds `Q`. The factor is preserved.
"""
function qr_q end
qr_q(
    f::Union{FaerFactorization{<:Any,:qr},FaerFactorization{<:Any,:pivoted_qr}};
    thin::Bool = false,
    team = nothing,
) = _factor_part(f, thin ? 6 : 3, (size(f, 1), thin ? minimum(size(f)) : size(f, 1)); team)
function Base.propertynames(f::FaerFactorization{T,K}, private::Bool = false) where {T,K}
    props =
        K in (:lu, :full_lu) ? (:L, :U, :p, :P) :
        K in (:qr, :pivoted_qr) ? (:Q, :Qthin, :R) : K === :cholesky ? (:L, :U, :UL) :
        (:L, :D)
    K in (:lu, :full_lu, :qr, :pivoted_qr, :cholesky, :ldlt, :bunchkaufman) &&
        (props=(props..., :packed))
    K in (:qr, :pivoted_qr) && (props=(props..., :Qbasis, :Qcoeff))
    K === :full_lu && (props = (props..., :q))
    K in (:pivoted_qr, :bunchkaufman) && (props = (props..., :p))
    K === :bunchkaufman && (props = (props..., :rook))
    return private ? (props..., :handle, :dims, :info, :uplo, :rook) :
           (props..., :info, :uplo)
end

"""
Reconstruct a new Rust-owned matrix from a reusable faer factorization.
`team=` lends a caller-owned `FaerTeam` to faer's reconstruction kernel.
"""
FaerMatrix(f::FaerFactorization{Float64}; team = nothing) =
    _factor_part(f, 0, size(f); team)
Base.Matrix(f::FaerFactorization; team = nothing) =
    _convert_dense_reconstruction(Matrix, f; team)
Base.Array(f::FaerFactorization; team = nothing) = Matrix(f; team)
function Base.copy(f::FaerFactorization{Float64,K}) where {K}
    lock(_matrix_lock) do
        h = RustCrate.OwnedFactor()
        _native_call(RustCrate.clone_factor, h, f)
        return FaerFactorization{Float64,K,typeof(h)}(h, size(f), f.info, f.uplo, f.rook)
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
    transposed = false;
    team = nothing,
    scratch = nothing,
)
    size(b, 1) == size(f, transposed ? 2 : 1) ||
        throw(DimensionMismatch("factor and RHS rows do not match"))
    LinearAlgebra.issuccess(f) || throw(LinearAlgebra.SingularException(f.info))
    _dense_rhs_validate(b)
    (!isnothing(scratch) || (size(f, 1)!=size(f, 2) && _dense_kind(f) in (0, 1))) &&
        return _dense_workspace_native_result(f, b, transposed ? 1 : 0, scratch, team)
    if !isnothing(team)
        return _dense_team_native_result(f, b, transposed ? 1 : 0, team)
    end
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

# `team` lends a caller-owned `FaerTeam` to faer's factor inverse kernel.
function Base.inv(f::FaerFactorization{Float64}; team = nothing)
    size(f, 1)==size(f, 2) || throw(DimensionMismatch("inverse requires a square factor"))
    LinearAlgebra.issuccess(f) || throw(LinearAlgebra.SingularException(f.info))
    return _matrix_result(RustCrate.factor_inverse, size(f), f, team)
end
Base.inv(a::FaerMatrix{Float64}; team = nothing) = inv(LinearAlgebra.lu(a; team); team)
LinearAlgebra.factorize(a::FaerMatrix{Float64}; team = nothing) =
    size(a, 1)==size(a, 2) ? LinearAlgebra.lu(a; team) :
    LinearAlgebra.qr(a, LinearAlgebra.ColumnNorm(); team)

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
