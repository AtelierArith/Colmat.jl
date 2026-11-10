"""
    FaerTridiagonalQ

Owned implicit Q returned by `tridiagonalize(A; compact=true)`. Products and
in-place Householder operations apply saved native reflectors and complex phases
without constructing dense Q. `Matrix(Q)` explicitly materializes it.

`basis`, `factor` and `phases` return independent Julia arrays of the base representation:
Q = diag(1, HouseholderSequence(basis,factor)) * Diagonal(phases).
Orientation is recorded by `transposed`/`conjugated`. Transpose, adjoint and conj
share immutable native storage; closing one Julia wrapper leaves the others open.
Copy/deepcopy duplicate the native storage. Kernel scratch and teams are supplied
independently for later applications; the construction policy is not retained.
"""
mutable struct FaerTridiagonalQ{T,H} <: AbstractFaerCompactQ{T}
    handle::Union{Nothing,H}
    n::Int
    k::Int
    blocksize::Int
    transposed::Bool
    conjugated::Bool
end

_compact_q_wrapper(::FaerTridiagonalQ) = FaerTridiagonalQ

function _compact_tridiagonal_eigen(
    f; recursion_threshold::Integer=128, sortby=nothing, scratch=nothing, team=nothing
)
    _iterative_resource_check(scratch, team)
    _compact_q_validate(f.Q)
    req = scratch_requirements(LinearAlgebra.eigen, f; recursion_threshold, team)
    work = isnothing(scratch) ? FaerScratch(req.bytes; alignment=req.alignment) : scratch
    original = nothing
    vectors = nothing
    success = false
    try
        cap = scratch_capacity(work)
        cap.bytes >= req.bytes && cap.alignment >= req.alignment || throw(
            ArgumentError(
                "saved compact tridiagonal eigen scratch capacity/alignment is insufficient; call reserve!",
            ),
        )
        E = tridiagonal_eigen(
            f.H.dv, f.H.ev; recursion_threshold, sortby, scratch=work, team
        )
        original = E.vectors
        vectors = _coerce(original, eltype(f.Q))
        LinearAlgebra.lmul!(f.Q, vectors; scratch=work, team)
        success = true
        return LinearAlgebra.Eigen(E.values, vectors)
    finally
        isnothing(original) || (success && original === vectors) || close(original)
        success || isnothing(vectors) || vectors === original || close(vectors)
        isnothing(scratch) && close(work)
    end
end
