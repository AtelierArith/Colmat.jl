# Julia LinearAlgebra conventions for self-adjoint dense factors that are not
# covered by the plain constructor methods: complex `Symmetric` wrappers and
# Bunch-Kaufman inertia.

const _ComplexSymmetric = LinearAlgebra.Symmetric{T,<:FaerMatrix{T}} where {T<:Complex}

# A complex Symmetric wrapper is Hermitian exactly when its stored triangle,
# diagonal included, is real. The native predicate reads only that triangle.
_real_triangle(a::_ComplexSymmetric) =
    _native_call(RustCrate.predicate, parent(a), Int32(a.uplo=='L' ? 5 : 6))
_hermitian_wrapper(a::_ComplexSymmetric) =
    LinearAlgebra.Hermitian(parent(a), a.uplo=='L' ? :L : :U)

"""
    cholesky(Symmetric(A::FaerMatrix{<:Complex}); check=true, team=nothing)

Follow Julia: a complex `Symmetric` wrapper is factored only when it is also
Hermitian (its stored triangle is real). Otherwise `check=true` throws
`PosDefException(-1)` and `check=false` returns an unsuccessful factor with
`info == -1`. `cholesky!` leaves the input unchanged in that case.
"""
function LinearAlgebra.cholesky(a::_ComplexSymmetric; check::Bool = true, team = nothing)
    _real_triangle(a) &&
        return LinearAlgebra.cholesky(_hermitian_wrapper(a); check, team)
    return _non_hermitian_cholesky(parent(a), check, RustCrate.TypedFactor)
end
function LinearAlgebra.cholesky!(a::_ComplexSymmetric; check::Bool = true, team = nothing)
    _real_triangle(a) &&
        return LinearAlgebra.cholesky!(_hermitian_wrapper(a); check, team)
    return _non_hermitian_cholesky(parent(a), check, RustCrate.TypedFactor)
end

# t4a-faer's LDLᴴ and LBLᴴ are self-adjoint only (they conjugate on every
# symmetric swap and reconstruction), so complex symmetric non-Hermitian LDLᵀ
# such as LAPACK `zsytrf` is not available natively.
function _complex_symmetric_unsupported(a::_ComplexSymmetric)
    _real_triangle(a) && return nothing
    throw(
        ArgumentError(
            "complex symmetric (non-Hermitian) LDLᵀ/Bunch-Kaufman is not provided by " *
            "t4a-faer; use Hermitian(A) for self-adjoint input",
        ),
    )
end
function LinearAlgebra.ldlt(a::_ComplexSymmetric; check::Bool = true, team = nothing)
    _complex_symmetric_unsupported(a)
    return LinearAlgebra.ldlt(_hermitian_wrapper(a); check, team)
end
function LinearAlgebra.ldlt!(a::_ComplexSymmetric; check::Bool = true, team = nothing)
    _complex_symmetric_unsupported(a)
    return LinearAlgebra.ldlt!(_hermitian_wrapper(a); check, team)
end
function LinearAlgebra.bunchkaufman(
    a::_ComplexSymmetric,
    rook::Bool = false;
    check::Bool = true,
    team = nothing,
)
    _complex_symmetric_unsupported(a)
    return LinearAlgebra.bunchkaufman(_hermitian_wrapper(a), rook; check, team)
end
function LinearAlgebra.bunchkaufman!(
    a::_ComplexSymmetric,
    rook::Bool = false;
    check::Bool = true,
    team = nothing,
)
    _complex_symmetric_unsupported(a)
    return LinearAlgebra.bunchkaufman!(_hermitian_wrapper(a), rook; check, team)
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
    F::FaerFactorization{T,:bunchkaufman};
    atol::Real = 0,
    rtol::Real = 0,
) where {T}
    R=real(T)
    atol=R(atol)
    rtol=R(rtol)
    n=size(F, 1)
    np, nn, nz = 0, 0, 0
    n==0 && return np, nn, nz
    rtol<=0 && atol<=0 && (rtol=n*eps(R))
    D=Matrix(F.D)
    realdata=T<:Real
    abs1(x) = realdata ? abs(x) : abs(real(x))+abs(imag(x))
    c=realdata ? one(R) : 1/sqrt(R(2))
    # First pass: estimate the largest singular value and 2×2 block bounds.
    s1=zero(R)
    bounds=Dict{Int,Tuple{R,R,R}}()
    i=1
    while i<=n
        if i<n && D[i, i+1]!=0
            dii=abs1(D[i, i])
            dxx=abs1(D[i+1, i+1])
            sblock=2*max(dii, abs1(D[i, i+1]), dxx)
            s1=max(s1, sblock)
            smin=if dii>=dxx
                c*abs1((D[i, i]/sblock)*D[i+1, i+1]-(D[i, i+1]/sblock)*D[i+1, i])
            else
                c*abs1(D[i, i]*(D[i+1, i+1]/sblock)-(D[i, i+1]/sblock)*D[i+1, i])
            end
            bounds[i]=(sblock, smin, real(D[i, i]+D[i+1, i+1]))
            i+=2
        else
            s1=max(s1, abs1(D[i, i]))
            i+=1
        end
    end
    # Second pass: count signs. A 2×2 Hermitian block has one eigenvalue of
    # each sign unless the smaller one is below the tolerance.
    tol=max(atol, rtol*s1)
    i=1
    while i<=n
        if haskey(bounds, i)
            sblock, smin, trace = bounds[i]
            if (c/2)*sblock<=tol
                nz+=2
            elseif smin>tol || trace==0
                np+=1
                nn+=1
            else
                nz+=1
                trace>=0 ? (np+=1) : (nn+=1)
            end
            i+=2
        else
            eig=real(D[i, i])
            eig>tol ? (np+=1) : eig < -tol ? (nn+=1) : (nz+=1)
            i+=1
        end
    end
    return np, nn, nz
end

LinearAlgebra.isposdef(a::_ComplexSymmetric) =
    LinearAlgebra.issuccess(LinearAlgebra.cholesky(a; check = false))
LinearAlgebra.isposdef(f::FaerFactorization{T,:cholesky}) where {T} =
    LinearAlgebra.issuccess(f)

# Julia's explicit `NoPivot()` spelling of the destructive LLᴴ stays native.
LinearAlgebra.cholesky!(a::FaerMatrix, ::LinearAlgebra.NoPivot; kwargs...) =
    LinearAlgebra.cholesky!(a; kwargs...)
for W in (
    :(LinearAlgebra.Hermitian{<:Any,<:FaerMatrix}),
    :(LinearAlgebra.Symmetric{<:Real,<:FaerMatrix}),
    :_ComplexSymmetric,
)
    @eval LinearAlgebra.cholesky!(a::$W, ::LinearAlgebra.NoPivot; kwargs...) =
        LinearAlgebra.cholesky!(a; kwargs...)
end
