# Determinant, trace and log-|det| of native operands by structure. Dense
# operands use faer's `determinant(par)` on the borrowed view (`team=` selects a
# caller-owned team); triangular and diagonal operands read only the stored
# diagonal. Views and transposes are borrowed, never copied.
const _MeasureDense=Union{
    FaerMatrix,
    FaerMatrixView,
    LinearAlgebra.Transpose{<:Any,<:Union{FaerMatrix,FaerMatrixView}},
    LinearAlgebra.Adjoint{<:Any,<:Union{FaerMatrix,FaerMatrixView}},
}
const _MeasureTriangular=Union{
    LinearAlgebra.LowerTriangular{<:Any,<:_MeasureDense},
    LinearAlgebra.UpperTriangular{<:Any,<:_MeasureDense},
    LinearAlgebra.UnitLowerTriangular{<:Any,<:_MeasureDense},
    LinearAlgebra.UnitUpperTriangular{<:Any,<:_MeasureDense},
}

function _measure_source(x)
    x isa FaerDiagonal && return (_handle(x); (x, _descriptor(x, 1:x.len, 1:1), Int32(5)))
    if x isa _MeasureTriangular
        p, d=_view_source(parent(x))
        return p, d, _product_structure(x)
    end
    p, d=_view_source(x)
    return p, d, Int32(0)
end
# Codes: 0 det, 1 trace, 2 (log|det|, phase) of a triangular or diagonal operand.
function _measure(x, code; team::Union{Nothing,FaerTeam} = nothing)
    lock(_matrix_lock) do
        LinearAlgebra.checksquare(x)
        isnothing(team) || _handle(team)
        p, d, structure=_measure_source(x)
        T=eltype(x)
        out=Vector{T}(undef, 2)
        GC.@preserve d out _native_call(
            RustCrate.view_measure,
            p,
            pointer(d),
            structure,
            Int32(code),
            Ptr{UInt8}(pointer(out)),
            team,
        )
        return code==2 ? (real(out[1]), out[2]) : out[1]
    end
end

"""
    det(A::FaerMatrix; team=nothing)

Native determinant of a square native matrix, range view, `transpose` or
`adjoint`, computed by faer's `determinant` on the borrowed view. `team` runs it
on a caller-owned `FaerTeam`. Triangular wrappers of these and `FaerDiagonal`
multiply only the stored diagonal (unit wrappers give one). A non-square operand
throws `DimensionMismatch`; an empty operand gives one.
"""
LinearAlgebra.det(x::_MeasureDense; team::Union{Nothing,FaerTeam} = nothing) =
    _measure(x, 0; team)
LinearAlgebra.det(x::Union{_MeasureTriangular,FaerDiagonal}) = _measure(x, 0)
LinearAlgebra.tr(x::Union{FaerMatrixView,FaerDiagonal}) = _measure(x, 1)

"""
    logabsdet(A)

For triangular wrappers of native matrices and `FaerDiagonal`, the sum of
`log(abs(d))` and the product of `sign(d)` over the stored diagonal, natively.
Views use a native copy and an LU factorization; transposes reuse their parent.
A zero entry gives `(-Inf, 0)`. An empty operand gives det one, trace zero and
`(0, 1)`, as for an empty dense matrix (LinearAlgebra rejects an empty `Diagonal`).
"""
LinearAlgebra.logabsdet(x::Union{_MeasureTriangular,FaerDiagonal}) = _measure(x, 2)
LinearAlgebra.logabsdet(x::FaerMatrixView) = LinearAlgebra.logabsdet(copy(x))
LinearAlgebra.logabsdet(
    x::LinearAlgebra.Transpose{<:Any,<:Union{FaerMatrix,FaerMatrixView}},
) = LinearAlgebra.logabsdet(parent(x))
function LinearAlgebra.logabsdet(
    x::LinearAlgebra.Adjoint{<:Any,<:Union{FaerMatrix,FaerMatrixView}},
)
    value, phase=LinearAlgebra.logabsdet(parent(x))
    return value, conj(phase)
end
function _logdet_from(value, phase)
    if phase isa Real
        phase<0 &&
            throw(DomainError(phase, "real logdet requires a nonnegative determinant"))
        return iszero(phase) ? oftype(value, -Inf) : value
    end
    return iszero(phase) ? complex(oftype(value, -Inf)) : value+log(phase)
end
const _LogdetOperand=Union{
    _MeasureTriangular,
    FaerDiagonal,
    FaerMatrixView,
    LinearAlgebra.Transpose{<:Any,<:Union{FaerMatrix,FaerMatrixView}},
    LinearAlgebra.Adjoint{<:Any,<:Union{FaerMatrix,FaerMatrixView}},
}
LinearAlgebra.logdet(x::_LogdetOperand) = _logdet_from(LinearAlgebra.logabsdet(x)...)
