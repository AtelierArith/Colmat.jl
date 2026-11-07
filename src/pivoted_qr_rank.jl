"""
    rank(F::FaerFactorization{T,:pivoted_qr}; atol=0, rtol=min(m,n)*eps(real(T))*iszero(atol))

Numerical rank of a column-pivoted QR factorization, with the rule of
`rank(::QRPivoted)`: the first index whose stored `|R[i,i]|` is at most
`max(atol, rtol*|R[1,1]|)`. The diagonal is read from the native factor without
reconstructing or copying `R`. Negative tolerances throw `ArgumentError`.
"""
function LinearAlgebra.rank(
    f::FaerFactorization{T,:pivoted_qr};
    atol::Real = 0,
    rtol::Real = min(size(f)...)*eps(real(float(T)))*iszero(atol),
) where {T}
    atol>=0 && rtol>=0 || throw(ArgumentError("rank tolerances must be nonnegative"))
    return Int(_native_call(RustCrate.pivoted_qr_rank, f, Float64(atol), Float64(rtol)))
end

function _pivoted_qr_rhs(b, m, ::Type{T}) where {T}
    x=b isa FaerMatrix ? _copy_view_matrix(view(b, 1:m, :)) :
      b isa FaerVector ? _copy_view_matrix(view(b, 1:m)) :
      FaerMatrix{T}(reshape(b[1:m, :], m, :))
    return eltype(x)===T ? x : FaerMatrix{T}(Matrix(x))
end
"""
    ldiv!(F::FaerFactorization{T,:pivoted_qr}, B, rcond; team=nothing) -> (B, rank)

Rank-truncated minimum-norm least-squares solve, matching
`ldiv!(::QRPivoted, B, rcond)`: `B` needs at least `max(m,n)` rows, its first
`m` rows are the right-hand side and its first `n` rows receive the solution.
The rank is decided by `rank(F; rtol=rcond)` on the stored R diagonal (LAPACK
`xgelsy` estimates it incrementally, which can differ near the threshold); the
leading `rank` rows of R are completed orthogonally so the solution has minimum
norm. `B` may be a native or Julia vector or matrix. `team` runs the native
kernels on a caller-owned `FaerTeam`.
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
        isnothing(team) || _handle(team)
        rhs=_pivoted_qr_rhs(b, m, T)
        h=T===Float64 ? RustCrate.OwnedMatrix() : RustCrate.TypedMatrix()
        r=_native_call(
            RustCrate.pivoted_qr_truncated_solve,
            h,
            f,
            rhs,
            Float64(rcond),
            team,
        )
        k=size(rhs, 2)
        x=T===Float64 ? _owned(h, (n, k)) : FaerMatrix{T,typeof(h)}(h, (n, k))
        if b isa Union{FaerMatrix,FaerVector}
            b isa FaerVector ? copyto!(view(b, 1:n), FaerVector(x)) :
            copyto!(view(b, 1:n, :), x)
        else
            b isa AbstractVector ? (b[1:n]=vec(Matrix(x))) : (b[1:n, :]=Matrix(x))
        end
        return b, Int(r)
    end
end
