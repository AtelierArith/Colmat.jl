# Direct in-place dense LU / QR / Cholesky: `lu!(A; scratch)`, `qr!(A; scratch)`
# and `cholesky!(A; scratch)` factor A's own native storage with faer's
# low-level `*_in_place` kernels, as Julia's LAPACK-backed `lu!` does.

"""
    FaerInPlaceFactorization

Factorization returned by `lu!(A; scratch)`, `qr!(A[, pivot]; scratch)` and
`cholesky!(A; scratch)` for a `FaerMatrix` `A`. Following Julia's `lu!`, the
packed factor lives in A itself: `F.factors === A`. LU stores U on/above the
diagonal and the strict unit-lower L below it; QR stores R on/above the
diagonal and faer's Householder basis below it, with the block Householder
coefficients (`F.Qcoeff`, not LAPACK's τ) kept in a Julia-owned buffer of F;
Cholesky stores L (or U for an upper `Hermitian`/`Symmetric` wrapper) and
zeroes the other triangle. The LU row permutation is kept in Julia vectors.

`full_piv_lu!(A; scratch)` stores LU as `lu!` does and keeps both permutations
(`F.p`, `F.q`). `ldlt!(A; scratch)` stores faer's unit-lower L below the
diagonal and D on it; `bunchkaufman!(A, rook; scratch)` stores L below the
diagonal and the block diagonal B's diagonal on it, with B's off-diagonal and
the symmetric permutation in Julia-owned buffers of F. For an upper
`Hermitian`/`Symmetric` wrapper both store the adjoint in the upper triangle;
the other triangle is zeroed. A singular `check=false` full LU or Bunch–Kaufman
factor keeps `info > 0`; an LDLᴴ zero pivot leaves A partially factored.

`qr!(A, ColumnNorm(); scratch)` runs faer's column-pivoted `qr_in_place`
(`FaerInPlaceFactorization{T,:pivoted_qr}`): A holds the packed QR of `A*P` as
unpivoted QR does, and the column permutation (`F.p`, `F.P` with
`A*P == Q*R`) lives in Julia vectors. The factor, `F.p` and `F.Qcoeff` are
bit-identical to the retained `qr(A, ColumnNorm())`. `rank(F; atol, rtol)`,
`ldiv!(F, B, rcond)` and `ldiv!(X, F, B; atol, rtol)` run the retained factor's
rank rule and rank-truncated minimum-norm kernel; the plain `\\`/`ldiv!`
solve does not detect rank deficiency.

F aliases A: later writes to A change F, and closing, resizing or truncating A
invalidates F (its operations then throw `ArgumentError`). `close(F)` only
invalidates F. Solves (`\\`, `ldiv!` with optional `scratch=`/`team=`, transpose
and adjoint for square factors, tall QR least squares), right solves of square
factors (`/`, `rdiv!(B, F)` with optional `scratch=`/`team=`, also transposed
and adjoint, on B's own storage), the destructive `LinearAlgebra.inv!(F)`
(which overwrites A with its inverse and closes F), `L/U/p/P/R/Q/Qthin/
Qbasis/Qcoeff/packed`, `q` (full LU), `D` (LDLᴴ, Bunch–Kaufman),
`regularization_count` (Cholesky, LDLᴴ), `rook` and
`inertia` (Bunch–Kaufman), `reconstruct(F)`/`Matrix(F)`/`FaerMatrix(F)` and their
typed forms, `inv`, `pinv` (QR), and `det/logabsdet/logdet` (QR from R's diagonal and the reflection count,
natively) read A's storage. `ldiv!(F, b)` also overwrites Julia strided arrays
and `FaerVector`s of a square factor (one native copy solved in place and
written back); a real factor solves a complex `b` as two real solves.
`copy(F)` copies A and F's buffers into a new direct factor, and
`FaerLinearOperator(F)` (also `cg(A, b; M = F)`) wraps such a copy as an
inverse-action operator. `F \\ b` of a rectangular LU/full LU, a wide QR or a
transposed rectangular factor returns the minimum-norm least-squares solution
through the pseudoinverse (`pinv(F)` for QR, `pinv(reconstruct(F))` for LU),
as the retained factor does. A right-hand side of a wider scalar type promotes
like the retained factor: a real factor splits a complex `b` of its precision
into two real solves, and otherwise `reconstruct(F)` is refactored once in the
promoted type (the in-place storage is left unchanged).
"""
mutable struct FaerInPlaceFactorization{T,K,M<:FaerMatrix{T}} <:
               LinearAlgebra.Factorization{T}
    factors::M
    perm::Vector{UInt}
    perm_inv::Vector{UInt}
    coeff::Matrix{T}
    dims::Tuple{Int,Int}
    epoch::UInt
    info::Int
    uplo::Symbol
    rook::Bool
    open::Bool
    # faer's `dynamic_regularization_count` (LLᴴ/LDLᴴ with regularization).
    regularization_count::Int
end
FaerInPlaceFactorization{T,K,M}(args::Vararg{Any,10}) where {T,K,M} =
    FaerInPlaceFactorization{T,K,M}(args..., 0)

# Native kind codes (the retained-factor construction codes).
_direct_kind(::Val{:lu}, rook = false) = Int32(0)
_direct_kind(::Val{:full_lu}, rook = false) = Int32(1)
_direct_kind(::Val{:qr}, rook = false) = Int32(2)
_direct_kind(::Val{:pivoted_qr}, rook = false) = Int32(3)
_direct_kind(::Val{:cholesky}, rook = false) = Int32(4)
_direct_kind(::Val{:ldlt}, rook = false) = Int32(5)
_direct_kind(::Val{:bunchkaufman}, rook = false) = rook ? Int32(8) : Int32(6)
_direct_kind(f::FaerInPlaceFactorization{T,K}) where {T,K} =
    _direct_kind(Val(K), getfield(f, :rook))

# Every native call goes through this check: A must be open, unresized and of
# the recorded shape, so a stale factor never reads reallocated storage.
function _direct_source(f::FaerInPlaceFactorization)
    getfield(f, :open) || throw(ArgumentError("FaerInPlaceFactorization has been closed"))
    a=getfield(f, :factors)
    isopen(a) &&
        a.epoch==getfield(f, :epoch) &&
        size(a)==getfield(f, :dims) ||
        throw(
            ArgumentError(
                "the matrix holding this in-place factorization was closed, resized or truncated",
            ),
        )
    _handle(a)
    return a
end

function _direct_requirements(kind, op, m, n, k, ::Type{T}, team) where {T}
    isnothing(team) || team isa FaerTeam || throw(ArgumentError("team must be a FaerTeam"))
    isnothing(team) || _handle(team)
    info=zeros(UInt, 3)
    _native_call(
        RustCrate.direct_factor_requirements,
        Int32(kind),
        Int32(op),
        UInt(m),
        UInt(n),
        UInt(k),
        _tag(T),
        team,
        info,
    )
    return (bytes = Int(info[1]), alignment = Int(info[2]), block = Int(info[3]))
end
function _direct_scratch_check(scratch, req, name)
    scratch isa FaerScratch || throw(ArgumentError("scratch must be a FaerScratch"))
    cap=scratch_capacity(scratch)
    cap.bytes>=req.bytes && cap.alignment>=req.alignment || throw(
        ArgumentError("$name scratch capacity/alignment is insufficient; call reserve!"),
    )
    return scratch
end
_coeff_pointer(c::Matrix) = Ptr{UInt8}(isempty(c) ? C_NULL : pointer(c))
# A zero-column coefficient buffer still needs a valid, non-null pointer.
_qr_coeff_pointer(c::Matrix) = Ptr{UInt8}(pointer(c))
_direct_coeff_pointer(kind, c::Matrix) = kind in (2, 3) ? _qr_coeff_pointer(c) : _coeff_pointer(c)

function _direct_factor!(
    a::FaerMatrix{T},
    ::Val{K},
    lower::Bool,
    scratch,
    team;
    rook::Bool = false,
    options = _NoSelfAdjointOptions,
) where {T,K}
    lock(_matrix_lock) do
        _handle(a)
        isnothing(team) ||
            team isa FaerTeam ||
            throw(ArgumentError("team must be a FaerTeam"))
        isnothing(team) || _handle(team)
        m, n=size(a)
        K in (:cholesky, :ldlt, :bunchkaufman) &&
            m!=n &&
            throw(DimensionMismatch("factorization requires a square matrix"))
        kind=_direct_kind(Val(K), rook)
        req=_with_selfadjoint_options(options) do
            return _direct_requirements(kind, 0, m, n, 0, T, team)
        end
        _direct_scratch_check(scratch, req, "factorization")
        # LU: row permutation; full LU: [row; column]; Bunch–Kaufman: symmetric.
        plen=K===:lu ? m : K===:full_lu ? m+n : K in (:bunchkaufman, :pivoted_qr) ? n : 0
        perm=Vector{UInt}(undef, plen)
        perm_inv=Vector{UInt}(undef, plen)
        # QR block coefficients, or Bunch–Kaufman's block off-diagonal column.
        coeff=Matrix{T}(
            undef,
            K in (:qr, :pivoted_qr) ? req.block : K===:bunchkaufman ? n : 0,
            K in (:qr, :pivoted_qr) ? min(m, n) : K===:bunchkaufman ? 1 : 0,
        )
        info, count=_with_selfadjoint_options(options) do
            info=GC.@preserve coeff Int(
                _native_call(
                    RustCrate.direct_factor,
                    a,
                    kind,
                    lower,
                    UInt(m),
                    UInt(n),
                    _tag(T),
                    perm,
                    perm_inv,
                    _direct_coeff_pointer(kind, coeff),
                    UInt(size(coeff, 1)),
                    scratch,
                    team,
                ),
            )
            return info, _regularization_count(K, info)
        end
        return FaerInPlaceFactorization{T,K,typeof(a)}(
            a,
            perm,
            perm_inv,
            coeff,
            (m, n),
            a.epoch,
            info,
            lower ? :L : :U,
            rook,
            true,
            count,
        )
    end
end
function _direct_non_hermitian(a::FaerMatrix{T}, check::Bool) where {T}
    LinearAlgebra.checksquare(a)
    check && throw(LinearAlgebra.PosDefException(-1))
    return FaerInPlaceFactorization{T,:cholesky,typeof(a)}(
        a,
        UInt[],
        UInt[],
        Matrix{T}(undef, 0, 0),
        size(a),
        a.epoch,
        -1,
        :L,
        false,
        true,
    )
end

function _direct_lu!(a::FaerMatrix, check, allowsingular, scratch, team)
    f=_direct_factor!(a, Val(:lu), true, scratch, team)
    check && !allowsingular && f.info!=0 && throw(LinearAlgebra.SingularException(f.info))
    return f
end
_direct_qr!(a::FaerMatrix, scratch, team) = _direct_factor!(a, Val(:qr), true, scratch, team)
_direct_pivoted_qr!(a::FaerMatrix, scratch, team) =
    _direct_factor!(a, Val(:pivoted_qr), true, scratch, team)
function _direct_cholesky!(
    a::FaerMatrix,
    lower::Bool,
    hermitian::Bool,
    check,
    scratch,
    team,
    options = _NoSelfAdjointOptions,
)
    lock(_matrix_lock) do
        _handle(a)
        hermitian || return _direct_non_hermitian(a, check)
        f=_direct_factor!(a, Val(:cholesky), lower, scratch, team; options)
        check && f.info!=0 && throw(LinearAlgebra.PosDefException(f.info))
        return f
    end
end
function _direct_full_piv_lu!(a::FaerMatrix, check, scratch, team)
    f=_direct_factor!(a, Val(:full_lu), true, scratch, team)
    check && f.info!=0 && throw(LinearAlgebra.SingularException(f.info))
    return f
end
# LDLᴴ / Bunch–Kaufman of the `lower` (or upper) triangle; the caller has
# checked that plain input is Hermitian.
function _direct_self_adjoint!(
    a::FaerMatrix,
    K::Symbol,
    lower::Bool,
    rook,
    check,
    scratch,
    team,
    options = _NoSelfAdjointOptions,
)
    f=_direct_factor!(a, Val(K), lower, scratch, team; rook = rook===true, options)
    check && f.info!=0 && throw(LinearAlgebra.SingularException(f.info))
    return f
end

"""
    scratch_requirements(lu!, A::FaerMatrix; team=nothing)
    scratch_requirements(qr!, A::FaerMatrix[, NoPivot() | ColumnNorm()]; team=nothing)
    scratch_requirements(cholesky!, A; team=nothing)
    scratch_requirements(full_piv_lu!, A::FaerMatrix; team=nothing)
    scratch_requirements(ldlt!, A; team=nothing)
    scratch_requirements(bunchkaufman!, A[, rook=false]; team=nothing, pivoting=nothing)

Query faer's exact workspace for the direct in-place factorization of A (a
`FaerMatrix`, or for the self-adjoint factorizations also a `Hermitian`/real
`Symmetric` wrapper of one). Pass the reserved `FaerScratch` as `scratch=` to
`lu!`, `qr!`, `cholesky!`, `full_piv_lu!`, `ldlt!` or `bunchkaufman!` to select
the direct path; see [`FaerInPlaceFactorization`](@ref). The query does not read
A's coefficients. Use the same team (and `rook`) for query and computation.
"""
function scratch_requirements(::typeof(LinearAlgebra.lu!), a::FaerMatrix; team = nothing)
    lock(_matrix_lock) do
        _handle(a)
        req=_direct_requirements(0, 0, size(a)..., 0, eltype(a), team)
        return (bytes = req.bytes, alignment = req.alignment)
    end
end
function scratch_requirements(::typeof(LinearAlgebra.qr!), a::FaerMatrix; team = nothing)
    lock(_matrix_lock) do
        _handle(a)
        req=_direct_requirements(2, 0, size(a)..., 0, eltype(a), team)
        return (bytes = req.bytes, alignment = req.alignment)
    end
end
scratch_requirements(
    ::typeof(LinearAlgebra.qr!),
    a::FaerMatrix,
    ::LinearAlgebra.NoPivot;
    team = nothing,
) = scratch_requirements(LinearAlgebra.qr!, a; team)
function scratch_requirements(
    ::typeof(LinearAlgebra.qr!),
    a::FaerMatrix,
    ::LinearAlgebra.ColumnNorm;
    team = nothing,
)
    lock(_matrix_lock) do
        _handle(a)
        req=_direct_requirements(3, 0, size(a)..., 0, eltype(a), team)
        return (bytes = req.bytes, alignment = req.alignment)
    end
end
const _DirectCholeskyInput = Union{
    FaerMatrix,
    LinearAlgebra.Hermitian{<:Any,<:FaerMatrix},
    LinearAlgebra.Symmetric{<:Any,<:FaerMatrix},
}
function scratch_requirements(
    ::typeof(LinearAlgebra.cholesky!),
    a::_DirectCholeskyInput;
    team = nothing,
)
    lock(_matrix_lock) do
        source=a isa FaerMatrix ? a : parent(a)
        _handle(source)
        n=LinearAlgebra.checksquare(source)
        req=_direct_requirements(4, 0, n, n, 0, eltype(source), team)
        return (bytes = req.bytes, alignment = req.alignment)
    end
end
function _direct_square_requirements(kind, a; team)
    lock(_matrix_lock) do
        source=a isa FaerMatrix ? a : parent(a)
        _handle(source)
        n=LinearAlgebra.checksquare(source)
        req=_direct_requirements(kind, 0, n, n, 0, eltype(source), team)
        return (bytes = req.bytes, alignment = req.alignment)
    end
end
function scratch_requirements(::typeof(full_piv_lu!), a::FaerMatrix; team = nothing)
    lock(_matrix_lock) do
        _handle(a)
        req=_direct_requirements(1, 0, size(a)..., 0, eltype(a), team)
        return (bytes = req.bytes, alignment = req.alignment)
    end
end
scratch_requirements(
    ::typeof(LinearAlgebra.ldlt!),
    a::_DirectCholeskyInput;
    team = nothing,
) = _direct_square_requirements(5, a; team)
function scratch_requirements(
    ::typeof(LinearAlgebra.bunchkaufman!),
    a::_DirectCholeskyInput,
    rook::Bool = false;
    team = nothing,
    pivoting = nothing,
)
    # faer's query takes the whole `LbltParams`: the strategy (`:full` keeps
    # per-thread pivot candidates) and the `with_cholesky_tuning` block size.
    options, rook=_lblt_options(rook, pivoting)
    return _with_selfadjoint_options(options) do
        _direct_square_requirements(_direct_kind(Val(:bunchkaufman), rook), a; team)
    end
end
for (fn, signature, args) in (
    (:(LinearAlgebra.lu!), :(a::FaerMatrix,), :(a,)),
    (:full_piv_lu!, :(a::FaerMatrix,), :(a,)),
    (:(LinearAlgebra.ldlt!), :(a::_DirectCholeskyInput,), :(a,)),
    (:(LinearAlgebra.bunchkaufman!), :(a::_DirectCholeskyInput,), :(a,)),
    (:(LinearAlgebra.bunchkaufman!), :(a::_DirectCholeskyInput, rook::Bool), :(a, rook)),
    (:(LinearAlgebra.qr!), :(a::FaerMatrix,), :(a,)),
    (:(LinearAlgebra.qr!), :(a::FaerMatrix, p::LinearAlgebra.NoPivot), :(a, p)),
    (:(LinearAlgebra.qr!), :(a::FaerMatrix, p::LinearAlgebra.ColumnNorm), :(a, p)),
    (:(LinearAlgebra.cholesky!), :(a::_DirectCholeskyInput,), :(a,)),
)
    @eval function FaerScratch(::typeof($fn), $(signature.args...); team = nothing, kwargs...)
        req=scratch_requirements($fn, $(args.args...); team, kwargs...)
        return FaerScratch(req.bytes; alignment = req.alignment)
    end
    @eval function reserve!(
        work::FaerScratch,
        ::typeof($fn),
        $(signature.args...);
        team = nothing,
        kwargs...,
    )
        req=scratch_requirements($fn, $(args.args...); team, kwargs...)
        return reserve!(work, req.bytes; alignment = req.alignment)
    end
end

Base.size(f::FaerInPlaceFactorization) = getfield(f, :dims)
Base.size(f::FaerInPlaceFactorization, d::Integer) =
    d < 1 ? throw(ArgumentError("dimension must be positive")) : d <= 2 ? size(f)[d] : 1
LinearAlgebra.issuccess(f::FaerInPlaceFactorization) = getfield(f, :info)==0
Base.isopen(f::FaerInPlaceFactorization) = lock(_matrix_lock) do
    getfield(f, :open) || return false
    a=getfield(f, :factors)
    return isopen(a) && a.epoch==getfield(f, :epoch) && size(a)==getfield(f, :dims)
end
function Base.close(f::FaerInPlaceFactorization)
    lock(_matrix_lock) do
        f.open=false
    end
    return nothing
end

function _direct_part(f::FaerInPlaceFactorization{T}, code, dims; team = nothing) where {T}
    lock(_matrix_lock) do
        a=_direct_source(f)
        coeff=getfield(f, :coeff)
        args=(
            a,
            _direct_kind(f),
            Int32(code),
            getfield(f, :uplo)===:L,
            UInt.(size(f))...,
            _tag(T),
            getfield(f, :perm),
            getfield(f, :perm_inv),
            _direct_coeff_pointer(_direct_kind(f), coeff),
            UInt(size(coeff, 1)),
            team,
        )
        GC.@preserve coeff begin
            T===Float64 ? _matrix_result(RustCrate.direct_part, dims, args...) :
            _typed_result(T, RustCrate.direct_part, dims, args...)
        end
    end
end
# The row (or symmetric) permutation; `column=true` gives full LU's columns.
function _direct_permutation(
    f::FaerInPlaceFactorization{T,K},
    column::Bool = false,
) where {T,K}
    lock(_matrix_lock) do
        _direct_source(f)
        p=getfield(f, :perm)
        m=size(f, 1)
        range=K===:full_lu ? (column ? ((m+1):length(p)) : (1:m)) : eachindex(p)
        return Int.(view(p, range)) .+ 1
    end
end
function Base.getproperty(f::FaerInPlaceFactorization{T,K}, name::Symbol) where {T,K}
    m, n=size(f)
    k=min(m, n)
    lu=K===:lu || K===:full_lu
    qrk=K===:qr || K===:pivoted_qr
    if name===:factors
        lock(() -> _direct_source(f), _matrix_lock)
        return getfield(f, :factors)
    elseif name===:packed
        K===:cholesky && return _direct_part(f, 1, (n, n))
        K in (:ldlt, :bunchkaufman) && return _direct_part(f, 7, (n, n))
        return lock(() -> copy(_direct_source(f)), _matrix_lock)
    elseif lu && name===:L
        return _direct_part(f, 1, (m, k))
    elseif lu && name===:U
        return _direct_part(f, 2, (k, n))
    elseif (lu || K in (:bunchkaufman, :pivoted_qr)) && name===:p
        return _direct_permutation(f)
    elseif K===:full_lu && name===:q
        return _direct_permutation(f, true)
    elseif (lu || K===:bunchkaufman) && name===:P
        return FaerMatrix(Matrix{T}(LinearAlgebra.I, m, m)[f.p, :])
    elseif K===:pivoted_qr && name===:P
        # `A*P == Q*R`, as `QRPivoted`.
        return FaerMatrix(Matrix{T}(LinearAlgebra.I, n, n)[:, f.p])
    elseif K in (:ldlt, :bunchkaufman) && name===:L
        return _direct_part(f, 1, (n, n))
    elseif K in (:ldlt, :bunchkaufman) && name===:D
        return _direct_part(f, 5, (n, n))
    elseif qrk && name===:Q
        return _direct_part(f, 3, (m, m))
    elseif qrk && name===:Qthin
        return _direct_part(f, 6, (m, k))
    elseif qrk && name===:R
        return _direct_part(f, 4, (k, n))
    elseif qrk && name===:Qbasis
        return _direct_part(f, 8, (m, k))
    elseif qrk && name===:Qcoeff
        return _direct_part(f, 9, size(getfield(f, :coeff)))
    elseif K===:cholesky && name===:L
        return _direct_part(f, 1, (n, n))
    elseif K===:cholesky && name===:U
        return _direct_part(f, 2, (n, n))
    elseif K===:cholesky && name===:UL
        return getfield(f, :uplo)===:U ? f.U : f.L
    end
    return getfield(f, name)
end
const _DirectQr{T} =
    Union{FaerInPlaceFactorization{T,:qr},FaerInPlaceFactorization{T,:pivoted_qr}}
qr_q(f::_DirectQr; thin::Bool = false, team = nothing) =
    _direct_part(
        f,
        thin ? 6 : 3,
        (size(f, 1), thin ? minimum(size(f)) : size(f, 1));
        team,
    )
function Base.propertynames(f::FaerInPlaceFactorization{T,K}, private::Bool = false) where {T,K}
    props =
        K===:lu ? (:L, :U, :p, :P) :
        K===:full_lu ? (:L, :U, :p, :q, :P) :
        K===:qr ? (:Q, :Qthin, :R, :Qbasis, :Qcoeff) :
        K===:pivoted_qr ? (:Q, :Qthin, :R, :Qbasis, :Qcoeff, :p, :P) :
        K===:ldlt ? (:L, :D) :
        K===:bunchkaufman ? (:L, :D, :p, :P, :rook) : (:L, :U, :UL)
    K in (:cholesky, :ldlt) && (props=(props..., :regularization_count))
    props=(:factors, props..., :packed, :info, :uplo)
    return private ? (props..., :perm, :perm_inv, :coeff, :dims, :epoch, :open) : props
end

function _cholesky_failure(f::FaerInPlaceFactorization{T,K}) where {T,K}
    info=getfield(f, :info)
    K===:cholesky ? LinearAlgebra.PosDefException(info) :
    LinearAlgebra.SingularException(info)
end
FaerMatrix(f::FaerInPlaceFactorization; team = nothing) = _direct_part(f, 0, size(f); team)
FaerMatrix{T}(f::FaerInPlaceFactorization; team = nothing) where {T<:_NativeScalar} =
    _typed_dense_reconstruction(T, f; team)
reconstruct(f::FaerInPlaceFactorization; team = nothing) = FaerMatrix(f; team)
Base.Matrix(f::FaerInPlaceFactorization; team = nothing) =
    _convert_dense_reconstruction(Matrix, f; team)
Base.Matrix{T}(f::FaerInPlaceFactorization; team = nothing) where {T} =
    _convert_dense_reconstruction(Matrix{T}, f; team)
Base.Array(f::FaerInPlaceFactorization; team = nothing) = Matrix(f; team)
function Base.inv(f::FaerInPlaceFactorization{T,K}; team = nothing) where {T,K}
    isnothing(team) || _handle(team)
    size(f, 1)==size(f, 2) || throw(DimensionMismatch("inverse requires a square factor"))
    LinearAlgebra.issuccess(f) || throw(_cholesky_failure(f))
    if K in (:qr, :pivoted_qr)
        d=Matrix(_direct_part(f, 11, (size(f, 1), 1)))
        i=findfirst(iszero, d)
        isnothing(i) || throw(LinearAlgebra.SingularException(i[1]))
    end
    return _direct_part(f, 10, size(f); team)
end
# Natively: the diagonal log-measure of #148 over one pivot per factor of
# det(A) (U's diagonal, L's twice, D, or B's block determinants) times the
# permutation signs. A singular LU or Bunch–Kaufman factor gives (-Inf, 0).
function LinearAlgebra.logabsdet(f::FaerInPlaceFactorization{T,K}) where {T,K}
    LinearAlgebra.checksquare(f)
    LinearAlgebra.issuccess(f) ||
        K in (:lu, :full_lu, :bunchkaufman) ||
        throw(_cholesky_failure(f))
    out=Matrix(_direct_part(f, 12, (2, 1)))
    return (real(out[1]), T<:Real ? real(out[2]) : out[2])
end
function LinearAlgebra.det(f::FaerInPlaceFactorization)
    value, phase=LinearAlgebra.logabsdet(f)
    return phase*exp(value)
end
LinearAlgebra.logdet(f::FaerInPlaceFactorization) =
    _logdet_from(LinearAlgebra.logabsdet(f)...)
LinearAlgebra.inertia(
    f::FaerInPlaceFactorization{T,:bunchkaufman};
    atol::Real = 0,
    rtol::Real = 0,
) where {T} = _bunchkaufman_inertia(f, T, atol, rtol)
LinearAlgebra.isposdef(f::FaerInPlaceFactorization{T,:cholesky}) where {T} =
    LinearAlgebra.issuccess(f)

function Base.copy(f::FaerInPlaceFactorization{T,K}) where {T,K}
    lock(_matrix_lock) do
        a=copy(_direct_source(f))
        return FaerInPlaceFactorization{T,K,typeof(a)}(
            a,
            copy(getfield(f, :perm)),
            copy(getfield(f, :perm_inv)),
            copy(getfield(f, :coeff)),
            size(f),
            a.epoch,
            getfield(f, :info),
            getfield(f, :uplo),
            getfield(f, :rook),
            true,
            getfield(f, :regularization_count),
        )
    end
end
Base.deepcopy_internal(f::FaerInPlaceFactorization, dict::IdDict) =
    get!(() -> copy(f), dict, f)

const _DirectTransposed = Union{
    LinearAlgebra.TransposeFactorization{<:Any,<:FaerInPlaceFactorization},
    LinearAlgebra.AdjointFactorization{<:Any,<:FaerInPlaceFactorization},
}
_direct_op(::FaerInPlaceFactorization) = 1
_direct_op(::LinearAlgebra.TransposeFactorization) = 2
_direct_op(::LinearAlgebra.AdjointFactorization) = 3
_direct_parent(f::FaerInPlaceFactorization) = f
_direct_parent(f::_DirectTransposed) = parent(f)
const _DirectSolveFactor = Union{FaerInPlaceFactorization,_DirectTransposed}

function _direct_solve_rows(f::FaerInPlaceFactorization{T,K}, op) where {T,K}
    m, n=size(f)
    if K in (:qr, :pivoted_qr) && op==1
        m>=n || throw(
            DimensionMismatch(
                "in-place QR solves require a tall or square factor; use F \\ b for the pseudoinverse solution",
            ),
        )
        return m
    end
    m==n || throw(
        DimensionMismatch(
            "in-place solves require a square factor (or a tall QR least-squares solve)",
        ),
    )
    return n
end

"""
    scratch_requirements(ldiv!, F::FaerInPlaceFactorization, B::FaerMatrix; team=nothing)

Query the workspace of `ldiv!(F, B; scratch, team)` (also for `transpose(F)` and
`adjoint(F)`). The solve overwrites B in place without native allocation of its
own; tall QR writes the least-squares solution into `B[1:n, :]`.
"""
function scratch_requirements(
    ::typeof(LinearAlgebra.ldiv!),
    g::_DirectSolveFactor,
    b::FaerMatrix;
    team = nothing,
)
    lock(_matrix_lock) do
        f=_direct_parent(g)
        _direct_source(f)
        _handle(b)
        op=_direct_op(g)
        _direct_solve_rows(f, op)
        req=_direct_requirements(_direct_kind(f), op, size(f)..., size(b, 2), eltype(f), team)
        return (bytes = req.bytes, alignment = req.alignment)
    end
end
function FaerScratch(
    ::typeof(LinearAlgebra.ldiv!),
    g::_DirectSolveFactor,
    b::FaerMatrix;
    team = nothing,
)
    req=scratch_requirements(LinearAlgebra.ldiv!, g, b; team)
    return FaerScratch(req.bytes; alignment = req.alignment)
end
function reserve!(
    work::FaerScratch,
    ::typeof(LinearAlgebra.ldiv!),
    g::_DirectSolveFactor,
    b::FaerMatrix;
    team = nothing,
)
    req=scratch_requirements(LinearAlgebra.ldiv!, g, b; team)
    return reserve!(work, req.bytes; alignment = req.alignment)
end

# Right solves `X op(A) = B` run natively on the transposed view of B
# (operation codes 4–6); they need a square factor.
function _direct_rdiv_shape(f::FaerInPlaceFactorization, b)
    size(f, 1)==size(f, 2) ||
        throw(DimensionMismatch("in-place right solves require a square factor"))
    size(b, 2)==size(f, 1) ||
        throw(DimensionMismatch("factor and right-hand side columns do not match"))
    return size(b, 1)
end

"""
    scratch_requirements(rdiv!, B::FaerMatrix, F::FaerInPlaceFactorization; team=nothing)

Query the workspace of `rdiv!(B, F; scratch, team)` (also for `transpose(F)` and
`adjoint(F)`, square factors only). The right solve `X op(A) = B` runs faer's
in-place left solve on the transposed view of B's own storage (`X A = B` as
`Aᵀ Xᵀ = Bᵀ`; `X Aᴴ = B` as the conjugate-only solve `conj(A) Xᵀ = Bᵀ`), so B is
overwritten with X without native allocation of its own.
"""
function scratch_requirements(
    ::typeof(LinearAlgebra.rdiv!),
    b::FaerMatrix,
    g::_DirectSolveFactor;
    team = nothing,
)
    lock(_matrix_lock) do
        f=_direct_parent(g)
        _direct_source(f)
        _handle(b)
        columns=_direct_rdiv_shape(f, b)
        req=_direct_requirements(
            _direct_kind(f),
            _direct_op(g)+3,
            size(f)...,
            columns,
            eltype(f),
            team,
        )
        return (bytes = req.bytes, alignment = req.alignment)
    end
end
function FaerScratch(
    ::typeof(LinearAlgebra.rdiv!),
    b::FaerMatrix,
    g::_DirectSolveFactor;
    team = nothing,
)
    req=scratch_requirements(LinearAlgebra.rdiv!, b, g; team)
    return FaerScratch(req.bytes; alignment = req.alignment)
end
function reserve!(
    work::FaerScratch,
    ::typeof(LinearAlgebra.rdiv!),
    b::FaerMatrix,
    g::_DirectSolveFactor;
    team = nothing,
)
    req=scratch_requirements(LinearAlgebra.rdiv!, b, g; team)
    return reserve!(work, req.bytes; alignment = req.alignment)
end

function _direct_ldiv!(g::_DirectSolveFactor, b::FaerMatrix, scratch, team; right = false)
    lock(_matrix_lock) do
        f=_direct_parent(g)
        T=eltype(f)
        a=_direct_source(f)
        _handle(b)
        eltype(b)===T || throw(
            ArgumentError("in-place factor solves require a right-hand side of eltype $T"),
        )
        _handle(b)===_handle(a) && throw(
            ArgumentError(
                "the right-hand side must not share storage with the in-place factorization",
            ),
        )
        isnothing(team) ||
            team isa FaerTeam ||
            throw(ArgumentError("team must be a FaerTeam"))
        isnothing(team) || _handle(team)
        LinearAlgebra.issuccess(f) || throw(_cholesky_failure(f))
        op=_direct_op(g)
        if right
            columns=_direct_rdiv_shape(f, b)
            op+=3
        else
            rows=_direct_solve_rows(f, op)
            size(b, 1)==rows || throw(DimensionMismatch("factor and RHS rows do not match"))
            columns=size(b, 2)
        end
        kind=_direct_kind(f)
        req=_direct_requirements(kind, op, size(f)..., columns, T, team)
        temporary=isnothing(scratch)
        work=temporary ? FaerScratch(req.bytes; alignment = req.alignment) : scratch
        try
            _direct_scratch_check(work, req, "solve")
            coeff=getfield(f, :coeff)
            GC.@preserve coeff _native_call(
                RustCrate.direct_solve,
                a,
                b,
                kind,
                Int32(op),
                getfield(f, :uplo)===:L,
                UInt.(size(f))...,
                _tag(T),
                getfield(f, :perm),
                getfield(f, :perm_inv),
                _direct_coeff_pointer(kind, coeff),
                UInt(size(coeff, 1)),
                work,
                team,
            )
        finally
            temporary && close(work)
        end
        return b
    end
end
LinearAlgebra.ldiv!(
    f::FaerInPlaceFactorization,
    b::FaerMatrix;
    scratch = nothing,
    team = nothing,
) = _direct_ldiv!(f, b, scratch, team)
for W in (:TransposeFactorization, :AdjointFactorization)
    @eval LinearAlgebra.ldiv!(
        f::LinearAlgebra.$W{<:Any,<:FaerInPlaceFactorization},
        b::FaerMatrix;
        scratch = nothing,
        team = nothing,
    ) = _direct_ldiv!(f, b, scratch, team)
end

# ldiv! on Julia arrays and native vectors overwrites b with the solution of
# a square factor: one native copy is solved in place and written back. A real
# factor solves a complex b as two real solves (real and imaginary parts),
# like LinearAlgebra. Rectangular factors use `F \ b` (the solution has a
# different number of rows).
function _direct_ldiv_array!(g::_DirectSolveFactor, b::AbstractVecOrMat; scratch, team)
    f=_direct_parent(g)
    T=eltype(f)
    lock(_matrix_lock) do
        _direct_source(f)
        rows=_direct_solve_rows(f, _direct_op(g))
        rows==size(f, 2) || throw(
            ArgumentError("in-place ldiv! needs a square factor; use F \\ b for least squares"),
        )
        size(b, 1)==rows || throw(DimensionMismatch("factor and RHS rows do not match"))
    end
    if eltype(b)<:Complex && T<:Real
        parts=(real.(b), imag.(b))
        for p in parts
            _direct_ldiv_array!(g, p; scratch, team)
        end
        b.=complex.(parts[1], parts[2])
        return b
    end
    eltype(b)===T ||
        throw(ArgumentError("ldiv! right-hand side must have the factor's scalar type (or its complex type)"))
    x=FaerMatrix{T}(b isa AbstractVector ? reshape(collect(b), :, 1) : b)
    try
        _direct_ldiv!(g, x, scratch, team)
        copyto!(b, b isa AbstractVector ? vec(Matrix(x)) : Matrix(x))
    finally
        close(x)
    end
    return b
end
const _DirectArrayRhs=Union{StridedVecOrMat{<:_NativeScalar},FaerVector}
LinearAlgebra.ldiv!(
    f::FaerInPlaceFactorization,
    b::_DirectArrayRhs;
    scratch = nothing,
    team = nothing,
) = _direct_ldiv_array!(f, b; scratch, team)
for W in (:TransposeFactorization, :AdjointFactorization)
    @eval LinearAlgebra.ldiv!(
        f::LinearAlgebra.$W{<:Any,<:FaerInPlaceFactorization},
        b::_DirectArrayRhs;
        scratch = nothing,
        team = nothing,
    ) = _direct_ldiv_array!(f, b; scratch, team)
end
# Whether the in-place kernels solve this shape: square factors, and tall QR
# least squares without a transpose.
function _direct_native_shape(f::FaerInPlaceFactorization{T,K}, op) where {T,K}
    m, n=size(f)
    return m==n || (K in (:qr, :pivoted_qr) && op==1 && m>n)
end
# Rectangular LU/full LU, wide QR and transposed rectangular solves: the
# minimum-norm least-squares solution through the pseudoinverse, as the
# retained factor's SVD/projection path returns. QR uses its native `pinv(F)`;
# LU reconstructs A natively first.
function _direct_pinv_solve(g::_DirectSolveFactor, b::AbstractVecOrMat)
    f=_direct_parent(g)
    T=eltype(f)
    K=_direct_kind_symbol(f)
    P=K in (:qr, :pivoted_qr) ? LinearAlgebra.pinv(f) : LinearAlgebra.pinv(reconstruct(f))
    op=_direct_op(g)
    Q=op==1 ? P : op==2 ? transpose(P) : adjoint(P)
    size(b, 1)==size(Q, 2) || throw(DimensionMismatch("factor and RHS rows do not match"))
    x=Q*FaerMatrix{T}(b isa AbstractVector ? reshape(collect(b), :, 1) : b)
    return b isa AbstractVector ? vec(Matrix(x)) : x
end
_direct_kind_symbol(::FaerInPlaceFactorization{T,K}) where {T,K} = K
# Mixed scalar types follow the retained factor's policy: the factor is
# reconstructed natively in the promoted type and refactored once as a
# retained factor of the same kind (the in-place storage is left untouched).
function _direct_promoted(f::FaerInPlaceFactorization{T,K}, ::Type{P}) where {T,K,P}
    a=FaerMatrix{P}(reconstruct(f))
    K===:lu && return LinearAlgebra.lu(a)
    K===:full_lu && return full_piv_lu(a)
    K===:qr && return LinearAlgebra.qr(a)
    K===:pivoted_qr && return LinearAlgebra.qr(a, LinearAlgebra.ColumnNorm())
    h=LinearAlgebra.Hermitian(a, getfield(f, :uplo))
    K===:cholesky && return LinearAlgebra.cholesky(h)
    K===:ldlt && return LinearAlgebra.ldlt(h)
    return LinearAlgebra.bunchkaufman(h, getfield(f, :rook)===true)
end
function _direct_backslash(g::_DirectSolveFactor, b::AbstractVecOrMat)
    f=_direct_parent(g)
    T=eltype(f)
    op=_direct_op(g)
    P=promote_type(T, eltype(b))
    if P!==T
        P<:_NativeScalar ||
            throw(ArgumentError("direct factor solves require native scalar types"))
        # Same precision, complex RHS of a real factor: two real solves.
        T<:Real && P===Complex{T} && return _direct_backslash_complex(g, b)
        G=_direct_promoted(f, P)
        H=op==1 ? G : op==2 ? transpose(G) : adjoint(G)
        return H\b
    end
    native=lock(() -> (_direct_source(f); _direct_native_shape(f, op)), _matrix_lock)
    native || return _direct_pinv_solve(g, b)
    rows=lock(() -> (_direct_source(f); _direct_solve_rows(f, op)), _matrix_lock)
    size(b, 1)==rows || throw(DimensionMismatch("factor and RHS rows do not match"))
    x=FaerMatrix{T}(b isa AbstractVector ? reshape(collect(b), :, 1) : b)
    _direct_ldiv!(g, x, nothing, nothing)
    n=size(f, 2)
    rows==n || truncate!(x, n, size(x, 2))
    return b isa AbstractVector ? vec(Matrix(x)) : x
end
# A real factor solves a complex RHS as two real solves, like LinearAlgebra.
_direct_backslash_complex(f, b) =
    _direct_backslash(f, real.(b)) .+ im .* _direct_backslash(f, imag.(b))
Base.:\(f::FaerInPlaceFactorization, b::AbstractVecOrMat) = _direct_backslash(f, b)
Base.:\(
    f::FaerInPlaceFactorization{T},
    b::VecOrMat{Complex{T}},
) where {T<:Union{Float32,Float64}} = _direct_backslash_complex(f, b)
for W in (:TransposeFactorization, :AdjointFactorization)
    @eval Base.:\(
        f::LinearAlgebra.$W{<:Any,<:FaerInPlaceFactorization},
        b::AbstractVecOrMat,
    ) = _direct_backslash(f, b)
    @eval Base.:\(
        f::LinearAlgebra.$W{T,<:FaerInPlaceFactorization},
        b::VecOrMat{Complex{T}},
    ) where {T<:Union{Float32,Float64}} = _direct_backslash_complex(f, b)
end

LinearAlgebra.rdiv!(
    b::FaerMatrix,
    f::FaerInPlaceFactorization;
    scratch = nothing,
    team = nothing,
) = _direct_ldiv!(f, b, scratch, team; right = true)
for W in (:TransposeFactorization, :AdjointFactorization)
    @eval LinearAlgebra.rdiv!(
        b::FaerMatrix,
        f::LinearAlgebra.$W{<:Any,<:FaerInPlaceFactorization};
        scratch = nothing,
        team = nothing,
    ) = _direct_ldiv!(f, b, scratch, team; right = true)
end

# `B / F` solves one native copy of B in place.
function _direct_slash(b::AbstractMatrix, g::_DirectSolveFactor)
    f=_direct_parent(g)
    lock(() -> (_direct_source(f); _direct_rdiv_shape(f, b)), _matrix_lock)
    x=FaerMatrix{eltype(f)}(b)
    return _direct_ldiv!(g, x, nothing, nothing; right = true)
end
# A real factor solves a complex RHS as two real solves, like `\`.
_direct_slash_complex(b, f) =
    Matrix(_direct_slash(real.(b), f)) .+ im .* Matrix(_direct_slash(imag.(b), f))
Base.:/(b::AbstractMatrix, f::FaerInPlaceFactorization) = _direct_slash(b, f)
for W in (:TransposeFactorization, :AdjointFactorization)
    @eval Base.:/(
        b::AbstractMatrix,
        f::LinearAlgebra.$W{<:Any,<:FaerInPlaceFactorization},
    ) = _direct_slash(b, f)
end
# The complex-RHS signatures of LinearAlgebra's real-factor `/` methods.
for B in (
        :(Union{
            Matrix{Complex{T}},
            LinearAlgebra.Adjoint{Complex{T},Vector{Complex{T}}},
            LinearAlgebra.Transpose{Complex{T},Vector{Complex{T}}},
        }),
        :(Matrix{Complex{T}}),
        :(LinearAlgebra.Adjoint{Complex{T},Vector{Complex{T}}}),
        :(LinearAlgebra.Transpose{Complex{T},Vector{Complex{T}}}),
    ),
    F in (
        :(FaerInPlaceFactorization{T}),
        :(LinearAlgebra.TransposeFactorization{T,<:FaerInPlaceFactorization}),
        :(LinearAlgebra.AdjointFactorization{T,<:FaerInPlaceFactorization}),
    )

    @eval Base.:/(b::$B, f::$F) where {T<:Union{Float32,Float64}} =
        _direct_slash_complex(b, f)
end

# Destructive inverse (native operation code 7): faer's inverse kernels read
# the factor while writing their output, so the packed factor is copied into
# the caller scratch and A itself receives inv(A).
function _direct_inverse_requirements(f::FaerInPlaceFactorization, team)
    _direct_source(f)
    size(f, 1)==size(f, 2) || throw(DimensionMismatch("inverse requires a square factor"))
    return _direct_requirements(_direct_kind(f), 7, size(f)..., 0, eltype(f), team)
end
"""
    scratch_requirements(inv!, F::FaerInPlaceFactorization; team=nothing)
    scratch_requirements(inv!, A::FaerMatrix; team=nothing)

Query the workspace of `LinearAlgebra.inv!(F; scratch, team)`: a copy of the
packed factor (plus Bunch–Kaufman's off-diagonal) and faer's inverse
workspace. For a square `FaerMatrix` A it also covers the direct LU of
`inv!(A)`.
"""
function scratch_requirements(
    ::typeof(LinearAlgebra.inv!),
    f::FaerInPlaceFactorization;
    team = nothing,
)
    lock(_matrix_lock) do
        req=_direct_inverse_requirements(f, team)
        return (bytes = req.bytes, alignment = req.alignment)
    end
end
function scratch_requirements(::typeof(LinearAlgebra.inv!), a::FaerMatrix; team = nothing)
    lock(_matrix_lock) do
        _handle(a)
        n=LinearAlgebra.checksquare(a)
        factor=_direct_requirements(0, 0, n, n, 0, eltype(a), team)
        inverse=_direct_requirements(0, 7, n, n, 0, eltype(a), team)
        return (
            bytes = max(factor.bytes, inverse.bytes),
            alignment = max(factor.alignment, inverse.alignment),
        )
    end
end
for X in (:FaerInPlaceFactorization, :FaerMatrix)
    @eval function FaerScratch(fun::typeof(LinearAlgebra.inv!), x::$X; team = nothing)
        req=scratch_requirements(fun, x; team)
        return FaerScratch(req.bytes; alignment = req.alignment)
    end
    @eval function reserve!(
        work::FaerScratch,
        fun::typeof(LinearAlgebra.inv!),
        x::$X;
        team = nothing,
    )
        req=scratch_requirements(fun, x; team)
        return reserve!(work, req.bytes; alignment = req.alignment)
    end
end

"""
    LinearAlgebra.inv!(F::FaerInPlaceFactorization; scratch=nothing, team=nothing)

Overwrite `F.factors` (the matrix A that `lu!`, `full_piv_lu!`, `qr!`,
`cholesky!`, `ldlt!` or `bunchkaufman!` factored in place) with `inv(A)` and
return it, like Julia's `inv!(::LU)` and `inv!(::Cholesky)`; for the
self-adjoint kinds both triangles hold the Hermitian inverse. F no longer holds
a factor afterwards, so it is closed. Singular or failed factors throw before
any write. The kernel uses only the caller scratch
([`scratch_requirements`](@ref)`(inv!, F)`), with no native allocation of its
own; `team` lends a caller-owned [`FaerTeam`](@ref).

    LinearAlgebra.inv!(A::FaerMatrix; scratch=nothing, team=nothing)

Invert a square `FaerMatrix` in place through its direct LU, as
`inv!(lu!(A; scratch); scratch)` with one workspace
(`scratch_requirements(inv!, A)`). A singular A throws `SingularException`
after the factorization has overwritten A.
"""
function LinearAlgebra.inv!(
    f::FaerInPlaceFactorization{T,K};
    scratch = nothing,
    team = nothing,
) where {T,K}
    lock(_matrix_lock) do
        a=_direct_source(f)
        isnothing(team) ||
            team isa FaerTeam ||
            throw(ArgumentError("team must be a FaerTeam"))
        isnothing(team) || _handle(team)
        req=_direct_inverse_requirements(f, team)
        LinearAlgebra.issuccess(f) || throw(_cholesky_failure(f))
        temporary=isnothing(scratch)
        work=temporary ? FaerScratch(req.bytes; alignment = req.alignment) : scratch
        try
            _direct_scratch_check(work, req, "inverse")
            kind=_direct_kind(f)
            coeff=getfield(f, :coeff)
            # QR returns the index of a zero R pivot, found before any write.
            code=GC.@preserve coeff _native_call(
                RustCrate.direct_inverse,
                a,
                kind,
                getfield(f, :uplo)===:L,
                UInt.(size(f))...,
                _tag(T),
                getfield(f, :perm),
                getfield(f, :perm_inv),
                _direct_coeff_pointer(kind, coeff),
                UInt(size(coeff, 1)),
                work,
                team,
            )
            iszero(code) || throw(LinearAlgebra.SingularException(Int(code)))
        finally
            temporary && close(work)
        end
        f.open=false
        return a
    end
end
function LinearAlgebra.inv!(a::FaerMatrix; scratch = nothing, team = nothing)
    LinearAlgebra.checksquare(a)
    temporary=isnothing(scratch)
    work=temporary ? FaerScratch(LinearAlgebra.inv!, a; team) : scratch
    try
        f=_direct_lu!(a, true, false, work, team)
        return LinearAlgebra.inv!(f; scratch = work, team)
    finally
        temporary && close(work)
    end
end

"""
    FaerLinearOperator(F::FaerInPlaceFactorization)

Inverse-action operator of a square nonsingular direct factor (`lu!`,
`full_piv_lu!`, `qr!` with or without `ColumnNorm()`, `cholesky!`, `ldlt!` or
`bunchkaufman!` with `scratch=`), usable as a preconditioner: `cg(A, b;
M = F)` converts it automatically. The operator owns an independent copy of
the factor (`copy(F)`), so later writes to `A` do not change it. Each forward
or adjoint application solves the callback's array in place with that copy
(`ldiv!(F, Y)`/`ldiv!(F', Y)`).
"""
function FaerLinearOperator(f::FaerInPlaceFactorization{T,K}) where {T,K}
    g=lock(_matrix_lock) do
        _direct_source(f)
        LinearAlgebra.checksquare(f)
        LinearAlgebra.issuccess(f) || throw(
            K in (:cholesky, :ldlt) ? _cholesky_failure(f) :
            LinearAlgebra.SingularException(Int(getfield(f, :info))),
        )
        copy(f)
    end
    n=size(g, 1)
    forward=(Y, X)->(copyto!(Y, X); LinearAlgebra.ldiv!(g, Y); Y)
    adjoint=(Y, X)->(copyto!(Y, X); LinearAlgebra.ldiv!(g', Y); Y)
    return FaerLinearOperator(forward, adjoint, n, n; T)
end
