"""
    svdvals(F::FaerSparseFactorization{T,:qr})

Compute singular values from the saved sparse R factor using native dense SVD.
R is materialized as an min(m,n)×n native dense matrix; the input snapshot is unused.
"""
function LinearAlgebra.svdvals(f::FaerSparseFactorization{T,:qr}) where {T}
    lock(_matrix_lock) do
        r = f.R
        dense = nothing
        try
            dense = FaerMatrix(r)
            return LinearAlgebra.svdvals(dense)
        finally
            isnothing(dense) || close(dense)
            close(r)
        end
    end
end
"""
    rank(F::FaerSparseFactorization{T,:qr}; atol=0, rtol)

Count singular values of saved R above `max(atol,rtol*maximum(svdvals(F)))`.
The default relative tolerance is `min(size(F)...)*eps(real(T))` when atol is zero.
"""
function LinearAlgebra.rank(
    f::FaerSparseFactorization{T,:qr};
    atol::Real=0,
    rtol::Real=_default_spectral_rtol(f, atol),
) where {T}
    _spectral_tolerances(atol, rtol)
    return _spectral_rank(f, LinearAlgebra.svdvals(f), atol, rtol)
end
"""
    pinv(F::FaerSparseFactorization{T,:qr}; atol=0, rtol)

Return an independent native dense Moore–Penrose inverse using saved Q/R and
column permutations. Uses native dense SVD of R and materializes thin Q;
rank-deficient inputs are truncated with the same tolerance contract as rank.
The source snapshot may be changed or closed.
"""
function LinearAlgebra.pinv(
    f::FaerSparseFactorization{T,:qr};
    atol::Real=0,
    rtol::Real=_default_spectral_rtol(f, atol),
) where {T}
    _spectral_tolerances(atol, rtol)
    lock(_matrix_lock) do
        r = dense = inverse = q = product = p = back = nothing
        try
            r = f.R
            dense = FaerMatrix(r)
            inverse = LinearAlgebra.pinv(dense; atol, rtol)
            q = f.Qthin
            product = inverse * adjoint(q)
            p = f.pcol
            back = invperm(p)
            return permute_rows(product, back)
        finally
            for x in (r, dense, inverse, q, product, p, back)
                isnothing(x) || close(x)
            end
        end
    end
end

function _sparse_qr_minimum_norm_solve(f, b; atol=nothing, rtol=nothing)
    base = f isa FaerSparseFactorization ? f : parent(f)
    base isa FaerSparseFactorization{<:Any,:qr} ||
        throw(ArgumentError("rank-aware solve requires sparse QR factors"))
    at = isnothing(atol) ? 0 : atol
    rt = isnothing(rtol) ? _default_spectral_rtol(base, at) : rtol
    return _sparse_qr_resource_result(f, b; atol=at, rtol=rt)
end
