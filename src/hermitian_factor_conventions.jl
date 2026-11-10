# Julia LinearAlgebra conventions for self-adjoint dense factors that are not
# covered by the plain constructor methods: complex `Symmetric` wrappers and
# Bunch-Kaufman inertia.

const _ComplexSymmetric = LinearAlgebra.Symmetric{T,<:FaerMatrix{T}} where {T<:Complex}

# A complex Symmetric wrapper is Hermitian exactly when its stored triangle,
# diagonal included, is real. The native predicate reads only that triangle.
function _real_triangle(a::_ComplexSymmetric)
    return _native_call(RustCrate.predicate, parent(a), Int32(a.uplo == 'L' ? 5 : 6))
end
function _hermitian_wrapper(a::_ComplexSymmetric)
    return LinearAlgebra.Hermitian(parent(a), a.uplo == 'L' ? :L : :U)
end

"""
    cholesky(Symmetric(A::FaerMatrix{<:Complex}); check=true, team=nothing)

Follow Julia: a complex `Symmetric` wrapper is factored only when it is also
Hermitian (its stored triangle is real). Otherwise `check=true` throws
`PosDefException(-1)` and `check=false` returns an unsuccessful factor with
`info == -1`. `cholesky!` leaves the input unchanged in that case.
"""
function LinearAlgebra.cholesky(
    a::_ComplexSymmetric; check::Bool=true, team=nothing, kwargs...
)
    _cholesky_kw(size(a, 1); kwargs...)
    _real_triangle(a) &&
        return LinearAlgebra.cholesky(_hermitian_wrapper(a); check, team, kwargs...)
    return _non_hermitian_cholesky(parent(a), check, RustCrate.TypedFactor)
end
function LinearAlgebra.cholesky!(
    a::_ComplexSymmetric; check::Bool=true, team=nothing, scratch=nothing, kwargs...
)
    options = _cholesky_kw(size(a, 1); kwargs...)
    _real_triangle(a) && return LinearAlgebra.cholesky!(
        _hermitian_wrapper(a); check, team, scratch, kwargs...
    )
    isnothing(scratch) ||
        return _direct_cholesky!(parent(a), true, false, check, scratch, team, options)
    return _non_hermitian_cholesky(parent(a), check, RustCrate.TypedFactor)
end

"""
    eigen(Symmetric(A::FaerMatrix{<:Complex}); sortby, team, permute=false, scale=false)
    eigvals(Symmetric(A::FaerMatrix{<:Complex}); ...)

Follow Julia, whose generic EVD of a complex `Symmetric` wrapper takes the
Hermitian branch exactly when the matrix is Hermitian. A real stored triangle
(diagonal included) runs faer's self-adjoint EVD on that triangle and returns
real eigenvalues with orthonormal eigenvectors (`permute`/`scale` are ignored,
as in Julia). Otherwise the matrix is complex symmetric but generally not
normal: the stored triangle is expanded natively into a full owner (one O(n²)
copy) and the general native EVD runs on it with these keywords, returning
complex eigenpairs. No `scratch` form is provided: the driver, and so the
workspace, depends on the stored data. `eigen!`/`eigvals!` are not defined for complex
`Symmetric` wrappers, as in Julia.
"""
function LinearAlgebra.eigen(
    a::_ComplexSymmetric;
    sortby=_eigensort,
    team=nothing,
    permute::Bool=false,
    scale::Bool=false,
)
    _real_triangle(a) && return LinearAlgebra.eigen(_hermitian_wrapper(a); sortby, team)
    full = _selfadjoint_full(a)
    try
        return LinearAlgebra.eigen(full; sortby, team, permute, scale)
    finally
        close(full)
    end
end
function LinearAlgebra.eigvals(
    a::_ComplexSymmetric;
    sortby=_eigensort,
    team=nothing,
    permute::Bool=false,
    scale::Bool=false,
)
    _real_triangle(a) && return LinearAlgebra.eigvals(_hermitian_wrapper(a); sortby, team)
    full = _selfadjoint_full(a)
    try
        return LinearAlgebra.eigvals(full; sortby, team, permute, scale)
    finally
        close(full)
    end
end

# t4a-faer's LDLᴴ and LBLᴴ are self-adjoint only (they conjugate on every
# symmetric swap and reconstruction), so complex symmetric non-Hermitian LDLᵀ
# such as LAPACK `zsytrf` is not available natively.
function _complex_symmetric_unsupported(a::_ComplexSymmetric)
    _real_triangle(a) && return nothing
    return throw(
        ArgumentError(
            "complex symmetric (non-Hermitian) LDLᵀ/Bunch-Kaufman is not provided by " *
            "t4a-faer; use Hermitian(A) for self-adjoint input",
        ),
    )
end
function LinearAlgebra.ldlt(a::_ComplexSymmetric; check::Bool=true, team=nothing, kwargs...)
    _complex_symmetric_unsupported(a)
    return LinearAlgebra.ldlt(_hermitian_wrapper(a); check, team, kwargs...)
end
function LinearAlgebra.ldlt!(
    a::_ComplexSymmetric; check::Bool=true, team=nothing, scratch=nothing, kwargs...
)
    _complex_symmetric_unsupported(a)
    return LinearAlgebra.ldlt!(_hermitian_wrapper(a); check, team, scratch, kwargs...)
end
function LinearAlgebra.bunchkaufman(
    a::_ComplexSymmetric, rook::Bool=false; check::Bool=true, team=nothing, kwargs...
)
    _complex_symmetric_unsupported(a)
    return LinearAlgebra.bunchkaufman(_hermitian_wrapper(a), rook; check, team, kwargs...)
end
function LinearAlgebra.bunchkaufman!(
    a::_ComplexSymmetric,
    rook::Bool=false;
    check::Bool=true,
    team=nothing,
    scratch=nothing,
    kwargs...,
)
    _complex_symmetric_unsupported(a)
    return LinearAlgebra.bunchkaufman!(
        _hermitian_wrapper(a), rook; check, team, scratch, kwargs...
    )
end

"""
    inertia(F::FaerFactorization{T,:bunchkaufman}; atol=0, rtol=0) -> (np, nn, nz)

Count the positive, negative and zero eigenvalues of the factored Hermitian
matrix from the block diagonal `F.D`, using the same tolerance rule and 2×2
block bounds as `LinearAlgebra.inertia(::BunchKaufman)`. Without tolerances,
`rtol = n*eps(real(T))`. Results near the threshold depend on the pivot
sequence; rook pivoting (`bunchkaufman(A, true)`) is recommended.
"""
function LinearAlgebra.inertia(
    F::FaerFactorization{T,:bunchkaufman}; atol::Real=0, rtol::Real=0
) where {T}
    return _bunchkaufman_inertia(F, T, atol, rtol)
end
# Shared by retained and direct in-place Bunch–Kaufman factors (reads `F.D`).
function _bunchkaufman_inertia(F, ::Type{T}, atol::Real, rtol::Real) where {T}
    R = real(T)
    atol = R(atol)
    rtol = R(rtol)
    n = size(F, 1)
    np, nn, nz = 0, 0, 0
    n == 0 && return np, nn, nz
    rtol <= 0 && atol <= 0 && (rtol = n * eps(R))
    D = Matrix(F.D)
    realdata = T <: Real
    abs1(x) = realdata ? abs(x) : abs(real(x)) + abs(imag(x))
    c = realdata ? one(R) : 1 / sqrt(R(2))
    # First pass: estimate the largest singular value and 2×2 block bounds.
    s1 = zero(R)
    bounds = Dict{Int,Tuple{R,R,R}}()
    i = 1
    while i <= n
        if i < n && D[i, i + 1] != 0
            dii = abs1(D[i, i])
            dxx = abs1(D[i + 1, i + 1])
            sblock = 2 * max(dii, abs1(D[i, i + 1]), dxx)
            s1 = max(s1, sblock)
            smin = if dii >= dxx
                c * abs1(
                    (D[i, i] / sblock) * D[i + 1, i + 1] -
                    (D[i, i + 1] / sblock) * D[i + 1, i],
                )
            else
                c * abs1(
                    D[i, i] * (D[i + 1, i + 1] / sblock) -
                    (D[i, i + 1] / sblock) * D[i + 1, i],
                )
            end
            bounds[i] = (sblock, smin, real(D[i, i] + D[i + 1, i + 1]))
            i += 2
        else
            s1 = max(s1, abs1(D[i, i]))
            i += 1
        end
    end
    # Second pass: count signs. A 2×2 Hermitian block has one eigenvalue of
    # each sign unless the smaller one is below the tolerance.
    tol = max(atol, rtol * s1)
    i = 1
    while i <= n
        if haskey(bounds, i)
            sblock, smin, trace = bounds[i]
            if (c / 2) * sblock <= tol
                nz += 2
            elseif smin > tol || trace == 0
                np += 1
                nn += 1
            else
                nz += 1
                trace >= 0 ? (np += 1) : (nn += 1)
            end
            i += 2
        else
            eig = real(D[i, i])
            if eig > tol
                (np += 1)
            elseif eig < -tol
                (nn += 1)
            else
                (nz += 1)
            end
            i += 1
        end
    end
    return np, nn, nz
end

function LinearAlgebra.isposdef(a::_ComplexSymmetric)
    return LinearAlgebra.issuccess(LinearAlgebra.cholesky(a; check=false))
end
function LinearAlgebra.isposdef(f::FaerFactorization{T,:cholesky}) where {T}
    return LinearAlgebra.issuccess(f)
end

# Julia's explicit `NoPivot()` spelling of the destructive LLᴴ stays native.
function LinearAlgebra.cholesky!(a::FaerMatrix, ::LinearAlgebra.NoPivot; kwargs...)
    return LinearAlgebra.cholesky!(a; kwargs...)
end
for W in (
    :(LinearAlgebra.Hermitian{<:Any,<:FaerMatrix}),
    :(LinearAlgebra.Symmetric{<:Real,<:FaerMatrix}),
    :_ComplexSymmetric,
)
    @eval LinearAlgebra.cholesky!(a::$W, ::LinearAlgebra.NoPivot; kwargs...) =
        LinearAlgebra.cholesky!(a; kwargs...)
end
