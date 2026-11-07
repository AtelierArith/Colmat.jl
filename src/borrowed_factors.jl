# Borrowed Julia arrays (`faerview`) as dense factorization and spectral
# inputs. faer reads the borrowed `MatRef` in place through the view
# descriptor; the factor still owns its numerical storage (the copy faer makes
# into the factor is the result), but no intermediate Julia or native snapshot
# of the input is made. Results have the same types as for `FaerMatrix`.

const _BorrowedFactorInput=FaerBorrowedMatrix{<:_NativeScalar}
const _BorrowedSelfAdjoint=Union{
    LinearAlgebra.Hermitian{T,<:FaerBorrowedMatrix{T}},
    LinearAlgebra.Symmetric{T,<:FaerBorrowedMatrix{T}},
} where {T<:Union{Float32,Float64}}
const _BorrowedHermitian=LinearAlgebra.Hermitian{T,<:FaerBorrowedMatrix{T}} where {
    T<:Union{ComplexF32,ComplexF64},
}
const _BorrowedSelfAdjointInput=Union{_BorrowedSelfAdjoint,_BorrowedHermitian}
_uplo(a::_BorrowedSelfAdjointInput) = a.uplo=='L' ? :L : :U
_borrowed_factor_handle(::Type{Float64}) = RustCrate.OwnedFactor()
_borrowed_factor_handle(::Type{<:_ExtraScalar}) = RustCrate.TypedFactor()

function _borrowed_factor(
    a::FaerBorrowedMatrix{T},
    ::Val{K},
    code;
    check = true,
    uplo = :L,
    team = nothing,
    rook::Bool = false,
) where {T,K}
    K in (:cholesky, :ldlt, :bunchkaufman) && LinearAlgebra.checksquare(a)
    isnothing(team) || team isa FaerTeam || throw(ArgumentError("team must be a FaerTeam"))
    lock(_matrix_lock) do
        isnothing(team) || _handle(team)
        h=_borrowed_factor_handle(T)
        p, d=_view_source(a)
        info=GC.@preserve d Int(
            _native_call(
                RustCrate.initialize_view,
                h,
                p,
                pointer(d),
                Int32(_construction_code(K, code, rook)),
                uplo===:L,
                UInt(size(a, 1)),
                UInt(size(a, 2)),
                _tag(T),
                team,
            ),
        )
        if check && info!=0
            K===:cholesky && throw(LinearAlgebra.PosDefException(info))
            throw(LinearAlgebra.SingularException(info))
        end
        return FaerFactorization{T,K,typeof(h)}(h, size(a), info, uplo, rook)
    end
end
# Real inputs use the symmetric test of the native Float64 methods, complex
# inputs the Hermitian one.
_borrowed_selfadjoint(a::FaerBorrowedMatrix{<:Real}) = LinearAlgebra.issymmetric(a)
_borrowed_selfadjoint(a::FaerBorrowedMatrix) = LinearAlgebra.ishermitian(a)
function _borrowed_require_selfadjoint(a)
    _borrowed_selfadjoint(a) && return nothing
    message="self-adjoint factorization requires Hermitian input"
    eltype(a)<:Complex &&
        LinearAlgebra.issymmetric(a) &&
        (message*="; complex symmetric (non-Hermitian) LDLᵀ is not provided by t4a-faer")
    throw(ArgumentError(message))
end

LinearAlgebra.lu(
    a::_BorrowedFactorInput;
    check::Bool = true,
    allowsingular::Bool = false,
    team = nothing,
) = _borrowed_factor(a, Val(:lu), 0; check = check&&!allowsingular, team)
LinearAlgebra.lu(a::_BorrowedFactorInput, ::LinearAlgebra.RowMaximum; kwargs...) =
    LinearAlgebra.lu(a; kwargs...)
full_piv_lu(a::_BorrowedFactorInput; check::Bool = true, team = nothing) =
    _borrowed_factor(a, Val(:full_lu), 1; check, team)
LinearAlgebra.qr(a::_BorrowedFactorInput; team = nothing) =
    _borrowed_factor(a, Val(:qr), 2; team)
LinearAlgebra.qr(a::_BorrowedFactorInput, ::LinearAlgebra.NoPivot; team = nothing) =
    LinearAlgebra.qr(a; team)
LinearAlgebra.qr(a::_BorrowedFactorInput, ::LinearAlgebra.ColumnNorm; team = nothing) =
    _borrowed_factor(a, Val(:pivoted_qr), 3; team)
function LinearAlgebra.cholesky(a::_BorrowedFactorInput; check::Bool = true, team = nothing)
    T=eltype(a)
    if !_borrowed_selfadjoint(a)
        # As for native matrices: an unsuccessful factor with `info == -1`.
        LinearAlgebra.checksquare(a)
        check && throw(LinearAlgebra.PosDefException(-1))
        h=_borrowed_factor_handle(T)
        return FaerFactorization{T,:cholesky,typeof(h)}(h, size(a), -1, :L, false)
    end
    return _borrowed_factor(a, Val(:cholesky), 4; check, team)
end
LinearAlgebra.cholesky(a::_BorrowedSelfAdjointInput; check::Bool = true, team = nothing) =
    _borrowed_factor(parent(a), Val(:cholesky), 4; check, team, uplo = _uplo(a))
function LinearAlgebra.ldlt(a::_BorrowedFactorInput; check::Bool = true, team = nothing)
    _borrowed_require_selfadjoint(a)
    return _borrowed_factor(a, Val(:ldlt), 5; check, team)
end
LinearAlgebra.ldlt(a::_BorrowedSelfAdjointInput; check::Bool = true, team = nothing) =
    _borrowed_factor(parent(a), Val(:ldlt), 5; check, team, uplo = _uplo(a))
function LinearAlgebra.bunchkaufman(
    a::_BorrowedFactorInput,
    rook::Bool = false;
    check::Bool = true,
    team = nothing,
)
    _borrowed_require_selfadjoint(a)
    return _borrowed_factor(a, Val(:bunchkaufman), 6; check, team, rook)
end
LinearAlgebra.bunchkaufman(
    a::_BorrowedSelfAdjointInput,
    rook::Bool = false;
    check::Bool = true,
    team = nothing,
) = _borrowed_factor(parent(a), Val(:bunchkaufman), 6; check, team, uplo = _uplo(a), rook)

# Options without a borrowed-input driver (scratch, team, alg, native results,
# left/both sides, balancing, active tuning) take one native copy of the input
# and use the `FaerMatrix` method.
function _borrowed_copy_call(f, a)
    x=copy(a)
    try
        return f(x)
    finally
        close(x)
    end
end
_borrowed_wrap(a::LinearAlgebra.Hermitian, x) = LinearAlgebra.Hermitian(x, Symbol(a.uplo))
_borrowed_wrap(a::LinearAlgebra.Symmetric, x) = LinearAlgebra.Symmetric(x, Symbol(a.uplo))
_borrowed_copy_call(f, a::_BorrowedSelfAdjointInput) =
    _borrowed_copy_call(x->f(_borrowed_wrap(a, x)), parent(a))

function _borrowed_svd(a::FaerBorrowedMatrix{T}, full, values_only) where {T}
    m, n=size(a)
    k=min(m, n)
    u=Matrix{T}(undef, m, values_only ? 0 : full ? m : k)
    s=Vector{_realtype(T)}(undef, k)
    v=Matrix{T}(undef, n, values_only ? 0 : full ? n : k)
    lock(_matrix_lock) do
        p, d=_view_source(a)
        GC.@preserve d u s v _native_call(
            RustCrate.view_svd_buffers,
            p,
            pointer(d),
            Ptr{UInt8}(values_only ? C_NULL : pointer(u)),
            Ptr{UInt8}(pointer(s)),
            Ptr{UInt8}(values_only ? C_NULL : pointer(v)),
            full,
            values_only,
        )
    end
    return values_only ? s : LinearAlgebra.SVD(u, s, copy(adjoint(v)))
end
function LinearAlgebra.svd(a::_BorrowedFactorInput; full::Bool = false, kwargs...)
    isempty(kwargs) && _svd_default_path(nothing, nothing) && return _borrowed_svd(a, full, false)
    return _borrowed_copy_call(x->LinearAlgebra.svd(x; full, kwargs...), a)
end
function LinearAlgebra.svdvals(a::_BorrowedFactorInput; kwargs...)
    isempty(kwargs) && _svd_default_path(nothing, nothing) && return _borrowed_svd(a, false, true)
    return _borrowed_copy_call(x->LinearAlgebra.svdvals(x; kwargs...), a)
end

function _borrowed_eigen(a::FaerBorrowedMatrix{T}, vectors, hermitian, lower, sortby) where {T}
    n=LinearAlgebra.checksquare(a)
    R=_realtype(T)
    values=Vector{hermitian ? R : Complex{R}}(undef, n)
    eigenvectors=Matrix{hermitian ? T : Complex{R}}(undef, n, vectors ? n : 0)
    lock(_matrix_lock) do
        p, d=_view_source(a)
        GC.@preserve d values eigenvectors _native_call(
            RustCrate.view_eigen_buffers,
            p,
            pointer(d),
            Ptr{UInt8}(pointer(values)),
            Ptr{UInt8}(pointer(eigenvectors)),
            vectors,
            hermitian,
            lower,
        )
    end
    if !isnothing(sortby)
        order=sortperm(values; by = sortby)
        values=values[order]
        vectors && (eigenvectors=eigenvectors[:, order])
    end
    return vectors ? LinearAlgebra.Eigen(values, eigenvectors) : values
end
# As LinearAlgebra does for Julia arrays, a Hermitian unwrapped borrow uses the
# self-adjoint driver (real eigenvalues; `permute`/`scale` do not apply) unless
# a left/both `side` is requested. Returns the remaining keywords, or nothing.
function _borrowed_hermitian_keywords(a, kwargs)
    k=Base.structdiff(values(kwargs), NamedTuple{(:permute, :scale)})
    return !haskey(k, :side) && LinearAlgebra.ishermitian(a) ? k : nothing
end
function LinearAlgebra.eigen(a::_BorrowedFactorInput; sortby = _eigensort, kwargs...)
    h=_borrowed_hermitian_keywords(a, kwargs)
    isnothing(h) || return LinearAlgebra.eigen(LinearAlgebra.Hermitian(a); sortby, h...)
    isempty(kwargs) &&
        _evd_default_path(nothing, nothing) &&
        return _borrowed_eigen(a, true, false, true, sortby)
    return _borrowed_copy_call(x->LinearAlgebra.eigen(x; sortby, kwargs...), a)
end
function LinearAlgebra.eigvals(a::_BorrowedFactorInput; sortby = _eigensort, kwargs...)
    h=_borrowed_hermitian_keywords(a, kwargs)
    isnothing(h) || return LinearAlgebra.eigvals(LinearAlgebra.Hermitian(a); sortby, h...)
    isempty(kwargs) &&
        _evd_default_path(nothing, nothing) &&
        return _borrowed_eigen(a, false, false, true, sortby)
    return _borrowed_copy_call(x->LinearAlgebra.eigvals(x; sortby, kwargs...), a)
end
function LinearAlgebra.eigen(a::_BorrowedSelfAdjointInput; sortby = _eigensort, kwargs...)
    isempty(kwargs) &&
        _evd_default_path(nothing, nothing) &&
        return _borrowed_eigen(parent(a), true, true, _uplo(a)===:L, sortby)
    return _borrowed_copy_call(x->LinearAlgebra.eigen(x; sortby, kwargs...), a)
end
function LinearAlgebra.eigvals(a::_BorrowedSelfAdjointInput; sortby = _eigensort, kwargs...)
    isempty(kwargs) &&
        _evd_default_path(nothing, nothing) &&
        return _borrowed_eigen(parent(a), false, true, _uplo(a)===:L, sortby)
    return _borrowed_copy_call(x->LinearAlgebra.eigvals(x; sortby, kwargs...), a)
end
