# Direct in-place dense LU / QR / Cholesky: `lu!(A; scratch)`, `qr!(A; scratch)`
# and `cholesky!(A; scratch)` factor A's own native storage with faer's
# low-level `*_in_place` kernels, as Julia's LAPACK-backed `lu!` does.

"""
    FaerInPlaceFactorization

Factorization returned by `lu!(A; scratch)`, `qr!(A[, NoPivot()]; scratch)` and
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

F aliases A: later writes to A change F, and closing, resizing or truncating A
invalidates F (its operations then throw `ArgumentError`). `close(F)` only
invalidates F. Solves (`\\`, `ldiv!` with optional `scratch=`/`team=`, transpose
and adjoint for square factors, tall QR least squares), `L/U/p/P/R/Q/Qthin/
Qbasis/Qcoeff/packed`, `q` (full LU), `D` (LDLᴴ, Bunch–Kaufman), `rook` and
`inertia` (Bunch–Kaufman), `reconstruct(F)`/`Matrix(F)`/`FaerMatrix(F)` and their
typed forms, `inv`, and `det/logabsdet/logdet` (all but QR, natively) read A's
storage. `copy(F)` copies A and F's buffers into a new direct
factor. Other retained-factor operations (`rdiv!`, linear-operator and
preconditioner use, pseudoinverse, mixed-scalar promotion, rectangular-LU or
wide-QR solves) require the default `lu!(A)`/`qr!(A)`/`cholesky!(A)` result.
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
end

# Native kind codes (the retained-factor construction codes).
_direct_kind(::Val{:lu}, rook = false) = Int32(0)
_direct_kind(::Val{:full_lu}, rook = false) = Int32(1)
_direct_kind(::Val{:qr}, rook = false) = Int32(2)
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
_direct_coeff_pointer(kind, c::Matrix) = kind==2 ? _qr_coeff_pointer(c) : _coeff_pointer(c)

function _direct_factor!(
    a::FaerMatrix{T},
    ::Val{K},
    lower::Bool,
    scratch,
    team;
    rook::Bool = false,
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
        req=_direct_requirements(kind, 0, m, n, 0, T, team)
        _direct_scratch_check(scratch, req, "factorization")
        # LU: row permutation; full LU: [row; column]; Bunch–Kaufman: symmetric.
        plen=K===:lu ? m : K===:full_lu ? m+n : K===:bunchkaufman ? n : 0
        perm=Vector{UInt}(undef, plen)
        perm_inv=Vector{UInt}(undef, plen)
        # QR block coefficients, or Bunch–Kaufman's block off-diagonal column.
        coeff=Matrix{T}(
            undef,
            K===:qr ? req.block : K===:bunchkaufman ? n : 0,
            K===:qr ? min(m, n) : K===:bunchkaufman ? 1 : 0,
        )
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
function _direct_cholesky!(a::FaerMatrix, lower::Bool, hermitian::Bool, check, scratch, team)
    lock(_matrix_lock) do
        _handle(a)
        hermitian || return _direct_non_hermitian(a, check)
        f=_direct_factor!(a, Val(:cholesky), lower, scratch, team)
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
)
    f=_direct_factor!(a, Val(K), lower, scratch, team; rook = rook===true)
    check && f.info!=0 && throw(LinearAlgebra.SingularException(f.info))
    return f
end

"""
    scratch_requirements(lu!, A::FaerMatrix; team=nothing)
    scratch_requirements(qr!, A::FaerMatrix[, NoPivot()]; team=nothing)
    scratch_requirements(cholesky!, A; team=nothing)
    scratch_requirements(full_piv_lu!, A::FaerMatrix; team=nothing)
    scratch_requirements(ldlt!, A; team=nothing)
    scratch_requirements(bunchkaufman!, A[, rook=false]; team=nothing)

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
scratch_requirements(
    ::typeof(LinearAlgebra.bunchkaufman!),
    a::_DirectCholeskyInput,
    rook::Bool = false;
    team = nothing,
) = _direct_square_requirements(_direct_kind(Val(:bunchkaufman), rook), a; team)
for (fn, signature, args) in (
    (:(LinearAlgebra.lu!), :(a::FaerMatrix,), :(a,)),
    (:full_piv_lu!, :(a::FaerMatrix,), :(a,)),
    (:(LinearAlgebra.ldlt!), :(a::_DirectCholeskyInput,), :(a,)),
    (:(LinearAlgebra.bunchkaufman!), :(a::_DirectCholeskyInput,), :(a,)),
    (:(LinearAlgebra.bunchkaufman!), :(a::_DirectCholeskyInput, rook::Bool), :(a, rook)),
    (:(LinearAlgebra.qr!), :(a::FaerMatrix,), :(a,)),
    (:(LinearAlgebra.qr!), :(a::FaerMatrix, p::LinearAlgebra.NoPivot), :(a, p)),
    (:(LinearAlgebra.cholesky!), :(a::_DirectCholeskyInput,), :(a,)),
)
    @eval function FaerScratch(::typeof($fn), $(signature.args...); team = nothing)
        req=scratch_requirements($fn, $(args.args...); team)
        return FaerScratch(req.bytes; alignment = req.alignment)
    end
    @eval function reserve!(
        work::FaerScratch,
        ::typeof($fn),
        $(signature.args...);
        team = nothing,
    )
        req=scratch_requirements($fn, $(args.args...); team)
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
    elseif (lu || K===:bunchkaufman) && name===:p
        return _direct_permutation(f)
    elseif K===:full_lu && name===:q
        return _direct_permutation(f, true)
    elseif (lu || K===:bunchkaufman) && name===:P
        return FaerMatrix(Matrix{T}(LinearAlgebra.I, m, m)[f.p, :])
    elseif K in (:ldlt, :bunchkaufman) && name===:L
        return _direct_part(f, 1, (n, n))
    elseif K in (:ldlt, :bunchkaufman) && name===:D
        return _direct_part(f, 5, (n, n))
    elseif K===:qr && name===:Q
        return _direct_part(f, 3, (m, m))
    elseif K===:qr && name===:Qthin
        return _direct_part(f, 6, (m, k))
    elseif K===:qr && name===:R
        return _direct_part(f, 4, (k, n))
    elseif K===:qr && name===:Qbasis
        return _direct_part(f, 8, (m, k))
    elseif K===:qr && name===:Qcoeff
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
qr_q(f::FaerInPlaceFactorization{T,:qr}; thin::Bool = false, team = nothing) where {T} =
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
        K===:ldlt ? (:L, :D) :
        K===:bunchkaufman ? (:L, :D, :p, :P, :rook) : (:L, :U, :UL)
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
    if K===:qr
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
    K===:qr && throw(
        ArgumentError(
            "logabsdet/det are not provided for in-place QR; use the default qr! result",
        ),
    )
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
    if K===:qr && op==1
        m>=n || throw(
            DimensionMismatch(
                "in-place QR solves require a tall or square factor; wide QR uses the default qr! result",
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

function _direct_ldiv!(g::_DirectSolveFactor, b::FaerMatrix, scratch, team)
    lock(_matrix_lock) do
        f=_direct_parent(g)
        T=eltype(f)
        a=_direct_source(f)
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
        rows=_direct_solve_rows(f, op)
        size(b, 1)==rows || throw(DimensionMismatch("factor and RHS rows do not match"))
        kind=_direct_kind(f)
        req=_direct_requirements(kind, op, size(f)..., size(b, 2), T, team)
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

function _direct_backslash(g::_DirectSolveFactor, b::AbstractVecOrMat)
    f=_direct_parent(g)
    T=eltype(f)
    op=_direct_op(g)
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
