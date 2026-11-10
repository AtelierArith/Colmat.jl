"""
    lu!(A::FaerMatrix; check=true, allowsingular=false, team=nothing, scratch=nothing)
    qr!(A::FaerMatrix[, pivot]; team=nothing[, scratch=nothing])
    cholesky!(A::FaerMatrix; check=true, team=nothing, scratch=nothing)

Without `scratch`, factor using native faer, then replace A's entries with
packed factors after a successful computation. LU stores U on/above the
diagonal and the strict L below it. QR stores R on/above the diagonal and
faer's Householder basis below; its block coefficients remain owned by the
returned factorization. This QR layout is not LAPACK's tau format. Cholesky
stores L (or U for an upper wrapper). F.packed returns independent
lower-oriented packed storage.

These methods allocate an independent factorization and stage the writeback;
F stays usable if A changes or closes, and existing same-shape views observe
the new entries. Errors before writeback preserve A. Failed check=false
Cholesky leaves A unchanged and returns the unsuccessful factorization.

With `scratch=work::FaerScratch` (sized by `scratch_requirements(lu!, A)`,
`scratch_requirements(qr!, A[, pivot])` or `scratch_requirements(cholesky!, A)`),
LU, QR (unpivoted or `ColumnNorm()`) and Cholesky instead run faer's
`lu_in_place`, `qr_in_place` (`no_pivoting` or `col_pivoting`) or
`cholesky_in_place` directly on A's native storage and return a
[`FaerInPlaceFactorization`](@ref) that aliases A (`F.factors === A`), as
Julia's `lu!` does. Colmat itself allocates no native memory there (only faer's
internal matrix products of blocked shapes may); the LU and column-pivoted QR permutations and the QR
coefficient buffers of F are allocated in Julia. Shape, scalar, scratch, team and
finiteness checks precede any write, but after a numerical failure (singular
`check=true` LU, non-positive-definite Cholesky) A holds partially factored
data, as with LAPACK.
"""
function _factor_writeback!(a::FaerMatrix, f::FaerFactorization; upper=false)
    if getfield(f, :info) != 0 &&
        (f isa FaerFactorization{<:Any,:cholesky} || f isa FaerFactorization{<:Any,:ldlt})
        return f
    end
    packed = f.packed
    copyto!(a, upper ? copy(adjoint(packed)) : packed)
    return f
end
function LinearAlgebra.lu!(
    a::FaerMatrix;
    check::Bool=true,
    allowsingular::Bool=false,
    team=nothing,
    scratch=nothing,
)
    isnothing(scratch) || return _direct_lu!(a, check, allowsingular, scratch, team)
    lock(_matrix_lock) do
        return _factor_writeback!(a, LinearAlgebra.lu(a; check, allowsingular, team))
    end
end
function LinearAlgebra.lu!(a::FaerMatrix, ::LinearAlgebra.RowMaximum; kwargs...)
    return LinearAlgebra.lu!(a; kwargs...)
end
function LinearAlgebra.qr!(a::FaerMatrix; team=nothing, scratch=nothing)
    return LinearAlgebra.qr!(a, LinearAlgebra.NoPivot(); team, scratch)
end
function LinearAlgebra.qr!(
    a::FaerMatrix, pivot::LinearAlgebra.NoPivot; team=nothing, scratch=nothing
)
    isnothing(scratch) || return _direct_qr!(a, scratch, team)
    lock(_matrix_lock) do
        return _factor_writeback!(a, LinearAlgebra.qr(a, pivot; team))
    end
end
function LinearAlgebra.qr!(
    a::FaerMatrix, pivot::LinearAlgebra.ColumnNorm; team=nothing, scratch=nothing
)
    isnothing(scratch) || return _direct_pivoted_qr!(a, scratch, team)
    lock(_matrix_lock) do
        return _factor_writeback!(a, LinearAlgebra.qr(a, pivot; team))
    end
end
function LinearAlgebra.cholesky!(
    a::FaerMatrix; check::Bool=true, team=nothing, scratch=nothing, kwargs...
)
    options = _cholesky_kw(size(a, 1); kwargs...)
    isnothing(scratch) || return _direct_cholesky!(
        a, true, LinearAlgebra.ishermitian(a), check, scratch, team, options
    )
    lock(_matrix_lock) do
        return _factor_writeback!(a, LinearAlgebra.cholesky(a; check, team, kwargs...))
    end
end
for W in (:Hermitian, :Symmetric)
    bound = W === :Symmetric ? Real : Any
    @eval function LinearAlgebra.cholesky!(
        a::LinearAlgebra.$W{T,<:FaerMatrix};
        check::Bool=true,
        team=nothing,
        scratch=nothing,
        kwargs...,
    ) where {T<:$bound}
        options = _cholesky_kw(size(a, 1); kwargs...)
        isnothing(scratch) || return _direct_cholesky!(
            parent(a), a.uplo == 'L', true, check, scratch, team, options
        )
        lock(_matrix_lock) do
            return _factor_writeback!(
                parent(a),
                LinearAlgebra.cholesky(a; check, team, kwargs...);
                upper=a.uplo == 'U',
            )
        end
    end
end
# `inplace=true`: the returned CholeskyPivoted's factors are A itself. With
# faer's automatic threshold (tol < 0) faer factors A's storage directly on the
# caller's scratch (`pivoted_llt_into`); an explicit tol >= 0 runs the panel
# algorithm (faer's in-place kernel has no absolute cutoff), which builds its
# own L before L is written into A. The lower triangle is read.
function _pivoted_inplace!(
    a;
    tol::Real=0.0,
    check::Bool=true,
    block_size::Integer=128,
    scratch=nothing,
    team=nothing,
)
    lock(_matrix_lock) do
        a isa Union{FaerMatrix,FaerMatrixView} ||
            a.uplo == 'L' ||
            throw(
                ArgumentError(
                    "inplace=true reads the lower triangle; wrap A as Hermitian(A, :L)"
                ),
            )
        source = a isa Union{FaerMatrix,FaerMatrixView} ? a : parent(a)
        _iterative_resource_check(scratch, team)
        _, n, ftol, block = _pivoted_input_plan(a, tol, block_size, team)
        T = eltype(source)
        if ftol < 0
            req = _pivoted_construction_requirement(n, T, ftol, block, team)
            p = zeros(UInt, n)
            rank = _pivoted_with_scratch(req, scratch) do work
                q, d = _view_source(source)
                GC.@preserve d Int(
                    _native_call(
                        RustCrate.pivoted_llt_into,
                        q,
                        pointer(d),
                        p,
                        UInt(block),
                        work,
                        team,
                    ),
                )
            end
            piv = Int.(p) .+ 1
        else
            f = LinearAlgebra.cholesky(
                a, LinearAlgebra.RowMaximum(); tol, check=false, block_size, scratch, team
            )
            try
                copyto!(source, f.factors)
            finally
                close(f.factors)
            end
            rank, piv = f.rank, f.p
        end
        check && rank < n && throw(LinearAlgebra.RankDeficientException(rank))
        return LinearAlgebra.CholeskyPivoted(source, 'L', piv, rank, tol, rank == n ? 0 : 1)
    end
end
function _pivoted_writeback!(a; inplace::Bool=false, kwargs...)
    inplace && return _pivoted_inplace!(a; kwargs...)
    lock(_matrix_lock) do
        source = a isa Union{FaerMatrix,FaerMatrixView} ? a : parent(a)
        f = LinearAlgebra.cholesky(a, LinearAlgebra.RowMaximum(); kwargs...)
        upper = !(a isa Union{FaerMatrix,FaerMatrixView}) && a.uplo == 'U'
        copyto!(source, upper ? copy(adjoint(f.factors)) : f.factors)
        return f
    end
end
function LinearAlgebra.cholesky!(a::FaerMatrix, ::LinearAlgebra.RowMaximum; kwargs...)
    return _pivoted_writeback!(a; kwargs...)
end
for W in (:Hermitian, :Symmetric)
    bound = W === :Symmetric ? Real : Any
    @eval LinearAlgebra.cholesky!(
        a::LinearAlgebra.$W{T,<:FaerMatrix}, ::LinearAlgebra.RowMaximum; kwargs...
    ) where {T<:$bound} = _pivoted_writeback!(a; kwargs...)
end

"""
    full_piv_lu!(A::FaerMatrix; check=true, team=nothing, scratch=nothing)
    ldlt!(A; check=true, team=nothing, scratch=nothing)
    bunchkaufman!(A, rook=false; check=true, team=nothing, scratch=nothing)

Without `scratch`, compute native full-pivot LU, LDLᴴ or Bunch–Kaufman, then
replace A by packed storage (lower-oriented, or its adjoint for an upper
wrapper). F.p/F.q retain the permutations. Returned factors are independent
of A.

With `scratch=work::FaerScratch` (sized by `scratch_requirements(full_piv_lu!, A)`,
`scratch_requirements(ldlt!, A)` or `scratch_requirements(bunchkaufman!, A[, rook])`),
faer's `lu::full_pivoting::factor::lu_in_place`, `cholesky::ldlt::factor::cholesky_in_place`
or `cholesky::lblt::factor::cholesky_in_place` (with the rook or default
partial-diagonal pivoting `LbltParams`) runs directly on A's native storage
and returns a [`FaerInPlaceFactorization`](@ref) aliasing A, as for `lu!`.

`cholesky!` and `ldlt!` accept the regularization keywords of `cholesky`/`ldlt`
and `bunchkaufman!` accepts `pivoting`, on both paths (pass the same `pivoting`
to `scratch_requirements(bunchkaufman!, A, rook; pivoting)`). Direct factors
report `F.regularization_count` like retained ones; parameters set by
[`with_cholesky_tuning`](@ref) apply to both paths and their queries.
"""
function full_piv_lu!(a::FaerMatrix; check::Bool=true, team=nothing, scratch=nothing)
    isnothing(scratch) || return _direct_full_piv_lu!(a, check, scratch, team)
    lock(_matrix_lock) do
        return _factor_writeback!(a, full_piv_lu(a; check, team))
    end
end
for fn in (:ldlt, :bunchkaufman)
    bang = Symbol(fn, :!)
    # `bunchkaufman!(A, rook=false)` keeps Julia's positional rook argument.
    pivot = fn === :bunchkaufman ? (Expr(:kw, :(rook::Bool), false),) : ()
    forward = fn === :bunchkaufman ? (:rook,) : ()
    # Regularization (`ldlt!`) or `pivoting` (`bunchkaufman!`) keywords.
    options = if fn === :bunchkaufman
        :(_lblt_kw(rook; kwargs...))
    else
        :((_ldlt_kw(size(a, 1); kwargs...), false))
    end
    @eval function LinearAlgebra.$bang(
        a::FaerMatrix,
        $(pivot...);
        check::Bool=true,
        team=nothing,
        scratch=nothing,
        kwargs...,
    )
        options, direct_rook = $options
        if !isnothing(scratch)
            # Like the default path: plain input must be Hermitian (checked
            # before any write).
            _require_hermitian(a)
            return _direct_self_adjoint!(
                a, $(QuoteNode(fn)), true, direct_rook, check, scratch, team, options
            )
        end
        lock(_matrix_lock) do
            return _factor_writeback!(
                a, LinearAlgebra.$fn(a, $(forward...); check, team, kwargs...)
            )
        end
    end
    for W in (:Hermitian, :Symmetric)
        bound = W === :Symmetric ? Real : Any
        @eval function LinearAlgebra.$bang(
            a::LinearAlgebra.$W{T,<:FaerMatrix},
            $(pivot...);
            check::Bool=true,
            team=nothing,
            scratch=nothing,
            kwargs...,
        ) where {T<:$bound}
            options, direct_rook = $options
            isnothing(scratch) || return _direct_self_adjoint!(
                parent(a),
                $(QuoteNode(fn)),
                a.uplo == 'L',
                direct_rook,
                check,
                scratch,
                team,
                options,
            )
            lock(_matrix_lock) do
                return _factor_writeback!(
                    parent(a),
                    LinearAlgebra.$fn(a, $(forward...); check, team, kwargs...);
                    upper=a.uplo == 'U',
                )
            end
        end
    end
end
