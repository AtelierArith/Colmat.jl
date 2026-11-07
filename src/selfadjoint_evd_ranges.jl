# LinearAlgebra's positional subset forms of self-adjoint `eigen`/`eigvals`:
# `irange` selects indices of the ascending eigenvalues, `vl, vu` the
# half-open interval (vl, vu] as LAPACK `syevr`. faer has no dense subset
# driver (bisection/MRRR), so the full native spectrum is computed and the
# selection is copied; the cost is that of the full `eigen`/`eigvals`.

function _selfadjoint_irange(n, irange::UnitRange)
    1<=first(irange)<=last(irange)<=n || throw(
        ArgumentError(
            "illegal choice of eigenvalue indices (il = $(first(irange)), " *
            "iu = $(last(irange))), which must be between 1 and n = $n",
        ),
    )
    return irange
end
function _selfadjoint_interval(vl::Real, vu::Real)
    vl<vu ||
        throw(ArgumentError("lower boundary, $vl, must be less than upper boundary, $vu"))
    return values->findall(x->vl<x<=vu, values)
end
_selfadjoint_select(result::LinearAlgebra.Eigen, select) =
    let k=select(result.values)
        LinearAlgebra.Eigen(result.values[k], result.vectors[:, k])
    end
_selfadjoint_select(values::AbstractVector, select) = values[select(values)]
_irange_selector(a, irange) =
    (_selfadjoint_irange(LinearAlgebra.checksquare(a), irange); _->irange)

"""
    eigen(A::Union{Hermitian,Symmetric}, irange::UnitRange)
    eigen(A::Union{Hermitian,Symmetric}, vl::Real, vu::Real)
    eigvals(A::Union{Hermitian,Symmetric}, irange::UnitRange)
    eigvals(A::Union{Hermitian,Symmetric}, vl::Real, vu::Real)

Follow LinearAlgebra's subset forms on native self-adjoint wrappers: `irange`
selects indices of the ascending eigenvalues and `vl, vu` the eigenvalues in
the half-open interval `(vl, vu]`. t4a-faer has no dense subset driver, so
these compute the full native spectrum (eigenvectors included for `eigen`)
and return copies of the selected values and vector columns; the cost and
workspace are those of the full decomposition. `eigmin`/`eigmax` use these.

`eigen!`/`eigvals!` with a subset first overwrite the parent exactly as the
full `eigen!`/`eigvals!` do (all eigenvectors or the diagonal eigenvalue
matrix; LinearAlgebra leaves it unspecified) and return the selection.
"""
function LinearAlgebra.eigen(a::_AnyFaerSelfAdjoint, irange::UnitRange)
    select=_irange_selector(a, irange)
    return _selfadjoint_select(LinearAlgebra.eigen(a), select)
end
function LinearAlgebra.eigvals(a::_AnyFaerSelfAdjoint, irange::UnitRange)
    select=_irange_selector(a, irange)
    return _selfadjoint_select(LinearAlgebra.eigvals(a), select)
end
LinearAlgebra.eigen(a::_AnyFaerSelfAdjoint, vl::Real, vu::Real) =
    _selfadjoint_select(LinearAlgebra.eigen(a), _selfadjoint_interval(vl, vu))
LinearAlgebra.eigvals(a::_AnyFaerSelfAdjoint, vl::Real, vu::Real) =
    _selfadjoint_select(LinearAlgebra.eigvals(a), _selfadjoint_interval(vl, vu))
function LinearAlgebra.eigen!(a::_AnyFaerSelfAdjoint, irange::UnitRange)
    select=_irange_selector(a, irange)
    return _selfadjoint_select(LinearAlgebra.eigen!(a), select)
end
function LinearAlgebra.eigvals!(a::_AnyFaerSelfAdjoint, irange::UnitRange)
    select=_irange_selector(a, irange)
    return _selfadjoint_select(LinearAlgebra.eigvals!(a), select)
end
function LinearAlgebra.eigen!(a::_AnyFaerSelfAdjoint, vl::Real, vu::Real)
    select=_selfadjoint_interval(vl, vu)
    return _selfadjoint_select(LinearAlgebra.eigen!(a), select)
end
function LinearAlgebra.eigvals!(a::_AnyFaerSelfAdjoint, vl::Real, vu::Real)
    select=_selfadjoint_interval(vl, vu)
    return _selfadjoint_select(LinearAlgebra.eigvals!(a), select)
end
