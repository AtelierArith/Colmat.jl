# Estimated 1-/Inf-norm condition numbers of native sparse matrices, following
# SparseArrays' `cond(::SparseMatrixCSC, p)`: ‖A‖ is exact and ‖A⁻¹‖ is the
# Higham–Tisseur block 1-norm estimate driven by solves with a saved native LU.

"""
    opnorm(A::FaerSparseMatrix, p::Real=2)

Operator 1- or Inf-norm (maximum absolute column or row sum) computed from one
snapshot of A's stored entries, in either CSC or CSR format. As for
`SparseMatrixCSC`, `p=2` and other values throw `ArgumentError`.
"""
function LinearAlgebra.opnorm(a::FaerSparseMatrix{T}, p::Real=2) where {T}
    p == 1 ||
        p == Inf ||
        throw(
            ArgumentError(
                if p == 2
                    "2-norm not yet implemented for sparse matrices. Try opnorm(Array(A)) or opnorm(A, p) where p=1 or Inf."
                else
                    "invalid operator p-norm p=$p. Valid: 1, Inf"
                end,
            ),
        )
    data = compressed_storage(a)
    R = _realtype(T)
    # Outer slices are columns of CSC and rows of CSR.
    outer = (p == 1) == (data.format === :csc)
    sums = zeros(R, size(a, (p == 1) ? 2 : 1))
    if outer
        for j in eachindex(sums), k in data.pointers[j]:(data.pointers[j + 1] - 1)
            sums[j] += abs(data.values[k])
        end
    else
        for k in eachindex(data.values)
            sums[data.indices[k]] += abs(data.values[k])
        end
    end
    return isempty(sums) ? zero(R) : maximum(sums)
end

"""
    cond(A::FaerSparseMatrix, p::Real=2)

Estimated 1- or Inf-norm condition number `opnorm(A,p) * ‖A⁻¹‖ₚ` of a square native
sparse matrix, as for `SparseMatrixCSC`: `‖A⁻¹‖ₚ` is the Higham–Tisseur block
estimate (a lower bound, usually exact) from a few solves with one native sparse LU
of A, so A⁻¹ is never formed. The estimate is deterministic. Empty input returns
zero and an LU that detects singularity returns `Inf`, as for dense `FaerMatrix`.
`p=2` throws `ArgumentError`; use `cond(FaerMatrix(A), 2)`.
"""
function LinearAlgebra.cond(a::FaerSparseMatrix{T}, p::Real=2) where {T}
    p == 1 ||
        p == Inf ||
        throw(
            ArgumentError(
                if p == 2
                    "only 1- and Inf-norm condition numbers are implemented for sparse matrices, for 2-norm try cond(FaerMatrix(A), 2) instead"
                else
                    "second argument must be either 1 or Inf, got $p"
                end,
            ),
        )
    lock(_matrix_lock) do
        _handle(a)
        n = LinearAlgebra.checksquare(a)
        R = _realtype(T)
        n == 0 && return zero(R)
        F = try
            LinearAlgebra.lu(a)
        catch e
            e isa LinearAlgebra.SingularException || rethrow()
            return R(Inf)
        end
        try
            LinearAlgebra.issuccess(F) || return R(Inf)
            # ‖A⁻¹‖_∞ = ‖A⁻ᴴ‖₁, so Inf swaps the roles of the two solves.
            inverse = _opnormestinv(F, n, T, p == Inf)
            isfinite(inverse) || return R(Inf)
            return R(LinearAlgebra.opnorm(a, p) * inverse)
        finally
            close(F)
        end
    end
end

# Solve with the saved factor (or its adjoint) and return one Julia matrix.
function _condest_solve(F, X, adjoint::Bool)
    Y = adjoint ? F' \ X : F \ X
    try
        return Matrix(Y)
    finally
        Y isa FaerMatrix && close(Y)
    end
end

# Higham–Tisseur block 1-norm estimate of B = A⁻¹ (or A⁻ᴴ when `adjoint`),
# ported from SparseArrays.opnormestinv with a fixed-seed generator so the
# estimate does not depend on, or advance, the global RNG.
function _opnormestinv(F, n::Int, ::Type{T}, adjoint::Bool, t::Int=min(2, n)) where {T}
    maxiter = 5
    rng = Random.Xoshiro(0x6f6e6f726d657374)
    Ti = typeof(float(zero(T)))
    ind = Vector{Int}(undef, n)
    ind_hist = Vector{Int}(undef, maxiter * t)
    S = zeros(T <: Real ? Int : Ti, n, t)
    any_abs_eq(v, k) = any(x -> abs(x) == k, v)
    X = Matrix{Ti}(undef, n, t)
    X[:, 1] .= 1
    for j in 2:t
        while true
            Random.rand!(rng, view(X, :, j), (-1, 1))
            any_abs_eq(X[:, j]' * X[:, 1:(j - 1)], n) || break
        end
    end
    LinearAlgebra.rmul!(X, inv(n))
    # Enough non-parallel ±1 vectors exist; avoids 2^(n-1) overflowing Int.
    resample_old = n > 62 || 2^(n - 1) >= 2t
    iter = 0
    est = est_old = zero(real(Ti))
    ind_best = 0
    while iter < maxiter
        iter += 1
        Y = _condest_solve(F, X, adjoint)
        est = zero(real(eltype(Y)))
        est_ind = 0
        for i in 1:t
            y = LinearAlgebra.norm(view(Y, :, i), 1)
            if y > est
                est = y
                est_ind = i
            end
        end
        iter == 1 && (est_old = est)
        (est > est_old || iter == 2) && (ind_best = est_ind)
        if iter >= 2 && est <= est_old
            est = est_old
            break
        end
        est_old = est
        S_old = copy(S)
        for j in 1:t, i in 1:n
            S[i, j] = Y[i, j] == 0 ? one(Y[i, j]) : sign(Y[i, j])
        end
        if T <: Real
            # Resample columns of S parallel to earlier columns of S or S_old.
            for j in 1:t
                while true
                    repeated = j > 1 && any_abs_eq(S[:, j]' * S[:, 1:(j - 1)], n)
                    repeated = repeated || (resample_old && any_abs_eq(S[:, j]' * S_old, n))
                    repeated || break
                    Random.rand!(rng, view(S, :, j), (-1, 1))
                end
            end
        end
        Z = _condest_solve(F, S, !adjoint)
        h = zeros(real(eltype(Z)), n)
        h_max = zero(real(eltype(Z)))
        h_ind = 0
        for i in 1:n
            h[i] = LinearAlgebra.norm(view(Z, i, :), Inf)
            if h[i] > h_max
                h_max = h[i]
                h_ind = i
            end
            ind[i] = i
        end
        iter >= 2 && ind_best == h_ind && break
        order = sortperm(h; rev=true)
        h = h[order]
        permute!(ind, order)
        if t > 1
            addcounter = t
            elemcounter = 0
            while addcounter > 0 && elemcounter < n
                elemcounter += 1
                current = ind[elemcounter]
                if current in view(ind_hist, 1:(t * (iter - 1)))
                    # The t largest rows were all used already: converged.
                    elemcounter == t && addcounter == t && break
                    continue
                end
                addcounter -= 1
                X[:, t - addcounter] .= 0
                X[current, t - addcounter] = 1
                ind_hist[iter * t - addcounter] = current
            end
        else
            ind_hist[1] = ind[1]
            X[:, 1] .= 0
            X[ind[1], 1] = 1
        end
    end
    return est
end
