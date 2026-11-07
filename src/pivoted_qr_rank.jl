"""
    rank(F::FaerFactorization{T,:pivoted_qr}; atol=0, rtol=min(m,n)*eps(real(T))*iszero(atol))

Numerical rank of a column-pivoted QR factorization, with the rule of
`rank(::QRPivoted)`: the first index whose stored `|R[i,i]|` is at most
`max(atol, rtol*|R[1,1]|)`. The diagonal is read from the native factor without
reconstructing or copying `R`. Negative or NaN tolerances throw `ArgumentError`.
"""
function LinearAlgebra.rank(
    f::FaerFactorization{T,:pivoted_qr};
    atol::Real = 0,
    rtol::Real = min(size(f)...)*eps(real(float(T)))*iszero(atol),
) where {T}
    at, rt=_spectral_tolerances(atol, rtol)
    return Int(_native_call(RustCrate.pivoted_qr_rank, f, at, rt))
end

# Every truncated solve picks `r` with `rank` in the source factor precision and
# then runs the one native kernel, so positional `rcond` and keyword tolerances
# with the same threshold produce identical solutions.
function _pivoted_qr_native(f::FaerFactorization{T}, rhs::FaerMatrix{T}, r, team) where {T}
    dims=(size(f, 2), size(rhs, 2))
    fun=RustCrate.pivoted_qr_truncated_solve
    result=T===Float64 ? _matrix_result(fun, dims, f, rhs, UInt(r), team) :
           _typed_result(T, fun, dims, f, rhs, UInt(r), team)
    try
        return Matrix(result)
    finally
        close(result)
    end
end
function _pivoted_qr_truncated(f::FaerFactorization{T}, rhs::FaerMatrix{S}, r, team) where {T,S}
    P=promote_type(T, S)
    if T<:Real && P===complex(T)
        # A real factor solves the real and imaginary parts separately.
        data=Matrix(rhs)
        parts=map((real, imag)) do part
            b=FaerMatrix{T}(part.(data))
            try
                _pivoted_qr_native(f, b, r, team)
            finally
                close(b)
            end
        end
        return complex.(parts...)
    end
    b=_coerce(rhs, P)
    try
        P===T && return _pivoted_qr_native(f, b, r, team)
        # The rank was chosen in source precision; only the solve is promoted.
        g=_dense_workspace_promote(f, P; team)
        try
            return _pivoted_qr_native(g, b, r, team)
        finally
            close(g)
        end
    finally
        b===rhs || close(b)
    end
end
_pivoted_qr_stage(b::FaerMatrix, rows) = _copy_view_matrix(view(b, 1:rows, :))
_pivoted_qr_stage(b::FaerVector, rows) = _copy_view_matrix(view(b, 1:rows))
_pivoted_qr_stage(b::AbstractVecOrMat, rows) = FaerMatrix(reshape(b[1:rows, :], rows, :))
function _pivoted_qr_check(f, team)
    _handle(f)
    isnothing(team) || team isa FaerTeam || throw(ArgumentError("team must be a FaerTeam"))
    isnothing(team) || _handle(team)
    return nothing
end
"""
    ldiv!(F::FaerFactorization{T,:pivoted_qr}, B, rcond; team=nothing) -> (B, rank)

Rank-truncated minimum-norm least-squares solve, matching
`ldiv!(::QRPivoted, B, rcond)`: `B` needs at least `max(m,n)` rows, its first
`m` rows are the right-hand side and its first `n` rows receive the solution.
The rank is `rank(F; atol=0, rtol=rcond)` on the stored R diagonal (LAPACK
`xgelsy` estimates it incrementally, which can differ near the threshold); the
leading `rank` rows of R are completed orthogonally so the solution has minimum
norm. The result equals `ldiv!(X, F, B[1:m, :]; atol=0, rtol=rcond)` exactly.
`B` may be a native or Julia vector or matrix; a real factor accepts a complex
`B`. A non-finite right-hand side throws and leaves `B` unchanged. `team` runs
the native kernels on a caller-owned `FaerTeam`.
"""
function LinearAlgebra.ldiv!(
    f::FaerFactorization{T,:pivoted_qr},
    b::AbstractVecOrMat,
    rcond::Real;
    team::Union{Nothing,FaerTeam} = nothing,
) where {T}
    lock(_matrix_lock) do
        m, n=size(f)
        size(b, 1)>=max(m, n) || throw(
            DimensionMismatch(
                "B has leading dimension $(size(b, 1)) but needs at least $(max(m, n))",
            ),
        )
        rcond>=0 || throw(ArgumentError("rcond must be nonnegative"))
        _pivoted_qr_check(f, team)
        r=LinearAlgebra.rank(f; atol = 0, rtol = rcond)
        rhs=_pivoted_qr_stage(b, m)
        x=try
            _pivoted_qr_truncated(f, rhs, r, team)
        finally
            close(rhs)
        end
        ndims(b)==1 ? copyto!(view(b, 1:n), vec(x)) : copyto!(view(b, 1:n, :), x)
        return b, r
    end
end

const _PivotedQrSolve = Union{
    FaerFactorization{T,:pivoted_qr},
    LinearAlgebra.TransposeFactorization{T,<:FaerFactorization{T,:pivoted_qr}},
    LinearAlgebra.AdjointFactorization{T,<:FaerFactorization{T,:pivoted_qr}},
} where {T}
function _pivoted_qr_tolerance_ldiv!(dest, f, b, atol, rtol, scratch, team)
    lock(_matrix_lock) do
        f isa FaerFactorization || throw(
            ArgumentError("rank-truncated QR solves support only untransformed factors"),
        )
        isnothing(scratch) || throw(
            ArgumentError(
                "rank-truncated QR solve allocates its native workspace; omit scratch",
            ),
        )
        _pivoted_qr_check(f, team)
        at=isnothing(atol) ? 0 : atol
        rt=isnothing(rtol) ? _default_spectral_rtol(f, at) : rtol
        _spectral_tolerances(at, rt)
        for a in (dest, b)
            _spectral_rhs_open(a)
            _dense_rhs_validate(a)
        end
        Base.require_one_based_indexing(dest, b)
        size(b, 1)==size(f, 1) ||
            throw(DimensionMismatch("factor and RHS dimensions do not match"))
        size(dest)==(ndims(b)==1 ? (size(f, 2),) : (size(f, 2), size(b, 2))) ||
            throw(DimensionMismatch("factor solve destination shape mismatch"))
        r=LinearAlgebra.rank(f; atol = at, rtol = rt)
        rhs=FaerMatrix(ndims(b)==1 ? reshape(b, :, 1) : b)
        x=try
            _pivoted_qr_truncated(f, rhs, r, team)
        finally
            close(rhs)
        end
        copyto!(dest, ndims(dest)==1 ? vec(x) : x)
        return dest
    end
end
"""
    ldiv!(X, F::FaerFactorization{T,:pivoted_qr}, B; atol=nothing, rtol=nothing, team=nothing)
    ldiv!(F::FaerFactorization{T,:pivoted_qr}, B; atol=nothing, rtol=nothing, team=nothing)

With `atol` or `rtol`, solve the least-squares problem truncated to
`r = rank(F; atol, rtol)` and write the minimum-norm solution of the truncated
system to `X` (`size(X) == (n, size(B, 2))`, `size(B, 1) == m`). The defaults follow
`pinv` and the sparse-QR minimum-norm solve: `atol=nothing` means zero and
`rtol=nothing` means `min(m,n)*eps(real(T))` when `atol` is zero, else zero.
Negative or NaN tolerances throw `ArgumentError`. `r == 0` writes zeros.

This is the same native kernel as `ldiv!(F, B, rcond)`: `atol=0, rtol=rcond` gives
identical values. The rank always uses the source factor precision; a real factor
with a complex RHS solves the real and imaginary parts, and other promotion reuses
the existing promoted refactorization. `team=` borrows a native team. The RHS is
staged before writeback, so `X` may alias `B`, and every validation error,
including a non-finite RHS, leaves `X` unchanged. `scratch` and tolerances on
transposed/adjoint factors throw `ArgumentError`. The two-argument form requires
a square factor. Without tolerances the ordinary dense solve runs unchanged.
"""
function LinearAlgebra.ldiv!(
    dest::AbstractMatrix,
    f::_PivotedQrSolve,
    b::AbstractMatrix;
    scratch = nothing,
    team = nothing,
    atol = nothing,
    rtol = nothing,
)
    isnothing(atol) && isnothing(rtol) && return _dense_ldiv!(dest, f, b; scratch, team)
    return _pivoted_qr_tolerance_ldiv!(dest, f, b, atol, rtol, scratch, team)
end
function LinearAlgebra.ldiv!(
    dest::AbstractVector,
    f::_PivotedQrSolve,
    b::AbstractVector;
    scratch = nothing,
    team = nothing,
    atol = nothing,
    rtol = nothing,
)
    isnothing(atol) && isnothing(rtol) && return _dense_ldiv!(dest, f, b; scratch, team)
    return _pivoted_qr_tolerance_ldiv!(dest, f, b, atol, rtol, scratch, team)
end
function LinearAlgebra.ldiv!(
    f::_PivotedQrSolve,
    b::AbstractVecOrMat;
    scratch = nothing,
    team = nothing,
    atol = nothing,
    rtol = nothing,
)
    size(f, 1)==size(f, 2) || throw(
        DimensionMismatch("two-argument ldiv! requires square factors; use ldiv!(X,F,B)"),
    )
    isnothing(atol) && isnothing(rtol) && return _dense_ldiv!(b, f, b; scratch, team)
    return _pivoted_qr_tolerance_ldiv!(b, f, b, atol, rtol, scratch, team)
end
