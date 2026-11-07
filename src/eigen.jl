# t4a-faer returns complex eigenpairs for general real matrices and real pairs
# for self-adjoint matrices. The Julia result types follow those conventions.
_eigensort(x::Real) = x
_eigensort(x::Complex) = (real(x), imag(x))
function _eigen(
    a::FaerMatrix{Float64},
    mode;
    vectors = true,
    lower = true,
    b = nothing,
    other_lower = true,
    sortby = _eigensort,
)
    n=LinearAlgebra.checksquare(a)
    if !isnothing(b)
        size(b)==size(a) ||
            throw(DimensionMismatch("generalized eigen inputs must have the same shape"))
    end
    T=mode in (1, 3) ? Float64 : ComplexF64
    values=Vector{T}(undef, n)
    eigenvectors=Matrix{T}(undef, n, vectors ? n : 0)
    lock(_matrix_lock) do
        other = isnothing(b) ? a : b
        GC.@preserve values eigenvectors begin
            _native_call(
                RustCrate.eigen_buffers,
                a,
                Ptr{UInt8}(pointer(values)),
                Ptr{UInt8}(pointer(eigenvectors)),
                vectors,
                Int32(mode),
                lower,
                other,
                other_lower,
            )
        end
    end
    if !isnothing(sortby)
        order=sortperm(values; by = sortby)
        values=values[order]
        vectors && (eigenvectors=eigenvectors[:, order])
    end
    return vectors ? LinearAlgebra.Eigen(values, eigenvectors) : values
end

"""
    eigen(A::FaerMatrix; side=:right, sortby, scratch, team, permute=false, scale=false)
    eigen(A::Union{Hermitian,Symmetric}; sortby)

Compute eigenvalues and right eigenvectors using t4a-faer. Self-adjoint
wrappers read only the specified triangle. Results own independent Julia
arrays. General real matrices produce complex eigenpairs. General native
matrices accept `side=:left` or `side=:both`. The left-only result is a named
tuple `(values,vectors)` satisfying `vectors'*A ≈ Diagonal(values)*vectors'`.
Both-sided results are `(values,left,right)` sharing one eigenvalue sequence.
The default right-sided result remains `Eigen`. Construct/query scratch with
the requested `side`; output arrays and sorting allocate separately. Side
selection applies to general matrices, not self-adjoint wrappers or bang calls.

faer's EVD driver does not balance. `permute=true, scale=true` (which must be
given together, because faer's `balance_in_place` always does both) balances
natively first, runs the EVD on the balanced matrix with the same `side`,
`scratch` and `team`, maps eigenvectors back and renormalizes each column to
unit 2-norm. The default `false` keeps the unbalanced native driver; unlike
LinearAlgebra's LAPACK default, balancing is opt-in.
"""
LinearAlgebra.eigen(
    a::FaerMatrix{Float64};
    side::Symbol = :right,
    sortby = _eigensort,
    scratch = nothing,
    team = nothing,
    permute::Bool = false,
    scale::Bool = false,
) =
    _balanced_check(permute, scale) ?
    _balanced_eigen(a, true; side, sortby, scratch, team) :
    side===:right && _general_evd_default_path(scratch, team) ? _eigen(a, 0; sortby) :
    _general_evd_workspace_result(a; sortby, scratch, team, side)
LinearAlgebra.eigvals(
    a::FaerMatrix{Float64};
    sortby = _eigensort,
    scratch = nothing,
    team = nothing,
    permute::Bool = false,
    scale::Bool = false,
) =
    _balanced_check(permute, scale) ? _balanced_eigen(a, false; sortby, scratch, team) :
    _general_evd_default_path(scratch, team) ? _eigen(a, 0; vectors = false, sortby) :
    _general_evd_workspace_result(a; vectors = false, sortby, scratch, team)
function LinearAlgebra.eigen(
    a::_FaerSelfAdjoint;
    native::Bool = false,
    sortby = _eigensort,
    scratch = nothing,
    team = nothing,
    alg = nothing,
)
    alg===nothing || return _selfadjoint_alg(alg; native) do
        LinearAlgebra.eigen(a; native, sortby, scratch, team)
    end
    return native ?
    _owned_spectral(parent(a); eigen = true, lower = _uplo(a)===:L, sortby, scratch, team) :
    _evd_default_path(scratch, team) ? _eigen(parent(a), 1; lower = _uplo(a)===:L, sortby) :
    _evd_workspace_result(a; sortby, scratch, team)
end
function LinearAlgebra.eigvals(
    a::_FaerSelfAdjoint;
    sortby = _eigensort,
    scratch = nothing,
    team = nothing,
    alg = nothing,
)
    alg===nothing || return _selfadjoint_alg(alg) do
        LinearAlgebra.eigvals(a; sortby, scratch, team)
    end
    return _evd_default_path(scratch, team) ?
           _eigen(parent(a), 1; vectors = false, lower = _uplo(a)===:L, sortby) :
           _evd_workspace_result(a; vectors = false, sortby, scratch, team)
end

"""
    eigen(A::FaerMatrix, B::FaerMatrix; homogeneous=false, side=:right, sortby, scratch, team)

Compute right generalized eigenvectors. The default returns `GeneralizedEigen`
with `A*V ≈ B*V*Diagonal(values)`. With `homogeneous=true`, return the named tuple
`(α, β, vectors)`, retaining native numerator/denominator pairs and satisfying
`A*V*Diagonal(β) ≈ B*V*Diagonal(α)`. This mode defaults to `sortby=nothing` and
performs no eigenvalue division. Explicit `sortby` sorts by the quotient while
retaining the homogeneous pairs. `eigvals(A,B; homogeneous=true)` returns `(α,β)`.
With `side=:left`, quotient mode returns the named tuple `(values,vectors)`
containing normalized left eigenvectors,
satisfying `V'*A ≈ Diagonal(values)*V'*B`, or
`Diagonal(β)*V'*A ≈ Diagonal(α)*V'*B` in homogeneous mode.
With `side=:both`, compute both sides in one native driver call and return
`(values,left,right)` or `(α,β,left,right)` in homogeneous mode. Query or
construct scratch with `side=:both` to include both packed real outputs.
Both modes accept reusable scratch and explicit teams. β=0 denotes an infinite
eigenvalue when α≠0; α=β=0 is indeterminate. The homogeneous option applies to
general native matrix pairs; bang routines retain their usual return types.
"""
function LinearAlgebra.eigen(
    a::FaerMatrix{Float64},
    b::FaerMatrix{Float64};
    homogeneous::Bool = false,
    side::Symbol = :right,
    sortby = homogeneous ? nothing : _eigensort,
    scratch = nothing,
    team = nothing,
)
    return _general_gevd_workspace_result(a, b; sortby, scratch, team, homogeneous, side)
end
function LinearAlgebra.eigvals(
    a::FaerMatrix{Float64},
    b::FaerMatrix{Float64};
    homogeneous::Bool = false,
    sortby = homogeneous ? nothing : _eigensort,
    scratch = nothing,
    team = nothing,
)
    return _general_gevd_workspace_result(
        a,
        b;
        vectors = false,
        sortby,
        scratch,
        team,
        homogeneous,
    )
end
function LinearAlgebra.eigen(
    a::_FaerSelfAdjoint,
    b::_FaerSelfAdjoint;
    sortby = _eigensort,
    scratch = nothing,
    team = nothing,
)
    return _selfadj_gevd_workspace_result(a, b; sortby, scratch, team)
end
function LinearAlgebra.eigvals(
    a::_FaerSelfAdjoint,
    b::_FaerSelfAdjoint;
    sortby = _eigensort,
    scratch = nothing,
    team = nothing,
)
    return _selfadj_gevd_workspace_result(a, b; vectors = false, sortby, scratch, team)
end

"""
    pfaffian(A::FaerMatrix{Float64})

Compute the Pfaffian of an even-dimensional skew-symmetric matrix with
 t4a-faer. Non-square, odd-dimensional and invalid inputs raise an exception.
"""
pfaffian(a::FaerMatrix{Float64}; scratch = nothing, team = nothing) =
    isnothing(scratch) && isnothing(team) ? _native_call(RustCrate.pfaffian_value, a) :
    _pfaffian_options(a, scratch, team)
