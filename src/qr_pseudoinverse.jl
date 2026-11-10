# Moore–Penrose inverses of saved dense QR factors. With A·P = Q·R, the
# singular values of R are those of A, so pinv(A) = P·pinv(R)·Qᴴ truncates at
# exactly the cutoff `pinv(A; atol, rtol)` uses.

"""
    scratch_requirements(pinv, F::FaerFactorization{T,:qr}; team=nothing)

Query native workspace for `pinv(F; scratch, team)` on a saved dense QR, with or
without column pivoting: the SVD of saved R (singular values, thin singular
vectors and faer's SVD workspace), as for
`scratch_requirements(FaerLinearOperator, F; atol, rtol, team)` with tolerances.
The requirement does not depend on the tolerances. The truncated singular
factors, the Householder application and the independent result matrix
allocate outside this buffer.
"""
function scratch_requirements(::typeof(LinearAlgebra.pinv), f::_DenseQrFactor; team=nothing)
    lock(_matrix_lock) do
        _handle(f)
        _construction_options(nothing, team)
        return _qr_pseudoinverse_requirement(f, team)
    end
end
function FaerScratch(fun::typeof(LinearAlgebra.pinv), f::_DenseQrFactor; kwargs...)
    req = scratch_requirements(fun, f; kwargs...)
    return FaerScratch(req.bytes; alignment=req.alignment)
end
function reserve!(
    s::FaerScratch, fun::typeof(LinearAlgebra.pinv), f::_DenseQrFactor; kwargs...
)
    req = scratch_requirements(fun, f; kwargs...)
    return reserve!(s, req.bytes; alignment=req.alignment)
end

"""
    pinv(F::FaerFactorization{T,:qr}; atol=0, rtol=min(size(F)...)*eps(real(T)), team=nothing, scratch=nothing)
    pinv(F::FaerInPlaceFactorization{T,:qr}; atol=0, rtol, team=nothing)
    pinv(F::FaerInPlaceFactorization{T,:pivoted_qr}; atol=0, rtol, team=nothing)

Return an independent native dense Moore–Penrose inverse from a saved dense QR
(`qr(A)`, `qr(A, ColumnNorm())` or the direct `qr!(A; scratch)`), without
refactorizing A. With `A·P = Q·R`, `pinv(A) = P·pinv(R)·Qᴴ` and R has the singular
values of A, so singular values at or below `max(atol, rtol*maximum(svdvals(A)))`
are discarded exactly as by `pinv(A; atol, rtol)`; a nonzero `atol` defaults
`rtol` to zero. Square, tall, wide and rank-deficient factors are accepted.

For a retained QR the SVD of R runs in caller-owned `scratch`
(`scratch_requirements(pinv, F; team)`; a temporary workspace when omitted) on
the optional `team`; the retained singular columns are applied through the
saved Householder reflectors, so no dense Q is formed. A direct
`qr!(A[, ColumnNorm()]; scratch)` factor uses the native SVD pseudoinverse of
`F.R` and the thin `F.Q` on `team` (no caller scratch form); the column-pivoted
one then permutes the rows of `R⁺Qᴴ` by `F.p` in Julia.
"""
function LinearAlgebra.pinv(
    f::_DenseQrFactor;
    atol::Real=0,
    rtol::Real=_default_spectral_rtol(f, atol),
    team=nothing,
    scratch=nothing,
)
    _spectral_tolerances(atol, rtol)
    lock(_matrix_lock) do
        # An explicit rtol selects the rank-truncated action for square QR too.
        op = FaerLinearOperator(f; atol, rtol, scratch, team)
        m, n = size(f)
        T = eltype(f)
        # Apply to the smaller identity: a tall factor forms (A⁺)ᴴ = op'·Iₙ.
        tall = m > n
        k = tall ? n : m
        g = tall ? adjoint(op) : op
        identity = out = nothing
        try
            identity = FaerMatrix{T}(Matrix{T}(LinearAlgebra.I, k, k))
            out = zeros(FaerMatrix{T}, size(g, 1), k)
            LinearAlgebra.mul!(out, g, identity; team)
            tall || return out
            return copy(adjoint(out))
        catch
            tall || isnothing(out) || close(out)
            rethrow()
        finally
            tall && !isnothing(out) && close(out)
            isnothing(identity) || close(identity)
            g === op || close(g)
            close(op)
        end
    end
end
function LinearAlgebra.pinv(
    f::_DirectQr{T}; atol::Real=0, rtol::Real=_default_spectral_rtol(f, atol), team=nothing
) where {T}
    _spectral_tolerances(atol, rtol)
    lock(_matrix_lock) do
        r = inner = q = nothing
        try
            r = f.R
            inner = LinearAlgebra.pinv(r; atol, rtol, team)
            q = qr_q(f; thin=true, team)
            out = zeros(FaerMatrix{T}, size(f, 2), size(f, 1))
            try
                LinearAlgebra.mul!(out, inner, adjoint(q); team)
                f isa FaerInPlaceFactorization{T,:pivoted_qr} || return out
                # A·P = Q·R gives A⁺ = P·R⁺·Qᴴ: row i of R⁺·Qᴴ is row p[i] of A⁺.
                permuted = FaerMatrix(Matrix(out)[invperm(f.p), :])
                close(out)
                return permuted
            catch
                close(out)
                rethrow()
            end
        finally
            for x in (r, inner, q)
                isnothing(x) || close(x)
            end
        end
    end
end
