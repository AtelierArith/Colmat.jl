# Condition numbers of structured dense wrappers: triangular 1-/Inf-norm
# estimates (as LinearAlgebra's LAPACK `trcon` path) and Hermitian 2-norm
# condition numbers from native self-adjoint eigenvalues.

_cond_p_check(p) =
    p==1 ||
    p==2 ||
    p==Inf ||
    throw(ArgumentError("p-norm must be 1, 2 or Inf, got $p"))

# The same triangular wrapper over a Julia copy of the native parent.
function _julia_triangular(w)
    parent_copy=Matrix(parent(w))
    return Base.typename(typeof(w)).wrapper(parent_copy)
end

"""
    cond(A::UpperTriangular{T,<:FaerMatrix}, p::Real=2)

Condition number of a `LowerTriangular`/`UpperTriangular`/`UnitLowerTriangular`/
`UnitUpperTriangular` wrapper of a `FaerMatrix`, a native view, or their
transpose/adjoint. As LinearAlgebra does for strided triangular matrices
(LAPACK `trcon`), `p=1` and `p=Inf` are estimates: `opnorm(A,p)` is exact and
`‖A⁻¹‖ₚ` is the Higham–Tisseur block estimate (a lower bound, usually exact)
from a few native triangular solves with A and Aᴴ, so A⁻¹ is never formed. The
estimate is deterministic. A zero diagonal gives `Inf`, empty input zero.
`p=2` uses the singular values of the triangle.
"""
function LinearAlgebra.cond(w::_DenseTriangular, p::Real = 2)
    _cond_p_check(p)
    p==2 && return invoke(LinearAlgebra.cond, Tuple{AbstractMatrix,Real}, w, p)
    T=eltype(w)
    R=_realtype(T)
    n=LinearAlgebra.checksquare(w)
    n==0 && return zero(R)
    inverse=try
        # ‖A⁻¹‖_∞ = ‖A⁻ᴴ‖₁, so Inf swaps the roles of the two solves.
        _opnormestinv(w, n, T, p==Inf)
    catch e
        e isa LinearAlgebra.SingularException || rethrow()
        return R(Inf)
    end
    isfinite(inverse) || return R(Inf)
    return R(LinearAlgebra.opnorm(_julia_triangular(w), p)*inverse)
end

"""
    cond(A::Hermitian{T,<:FaerMatrix}, p::Real=2)

Condition number of a native Hermitian (or real `Symmetric`) matrix. The 2-norm
value is `maximum(abs.(λ))/minimum(abs.(λ))` from faer's self-adjoint
eigenvalues of the selected triangle, without an SVD; a zero eigenvalue gives
`Inf` and empty input zero. `p=1` and `p=Inf` use the generic exact path.
"""
function LinearAlgebra.cond(a::_PinvSelfAdjoint, p::Real = 2)
    _cond_p_check(p)
    p==2 || return invoke(LinearAlgebra.cond, Tuple{AbstractMatrix,Real}, a, p)
    R=_realtype(eltype(a))
    if isempty(a)
        LinearAlgebra.checksquare(a)
        return zero(R)
    end
    values=abs.(LinearAlgebra.eigvals(a))
    largest=maximum(values)
    return iszero(largest) ? R(Inf) : R(largest/minimum(values))
end
