# Operator/scalar promotion and layout are resolved before querying native scratch.
function _operator_workspace_query(op::FaerLinearOperator, columns; team=nothing)
    _handle(op)
    isnothing(team) || team isa FaerTeam || throw(ArgumentError("team must be a FaerTeam"))
    isnothing(team) || _handle(team)
    info = zeros(UInt, 2)
    args = (op, UInt(columns), UInt(size(op, 1)), UInt(size(op, 2)), _tag(eltype(op)), info)
    GC.@preserve info begin
        _native_call(RustCrate.operator_requirements, args..., team)
    end
    return (bytes=Int(info[1]), alignment=Int(info[2]))
end
"""
    scratch_requirements(*, op::FaerLinearOperator, B; team=nothing)
    scratch_requirements(mul!, X, op::FaerLinearOperator, B[, alpha,beta]; team=nothing)

Query native operator application scratch at the promoted scalar type and team.
Right-operator products have the corresponding `(A, op)` order. Includes intermediates declared by native LinOp scratch queries. Independent
result matrices, Julia scaling/conversion buffers and promoted operator snapshots
allocate separately. Prepared owned/wide sparse-QR inverse actions use caller
scratch and propagate the team through projection, triangular solves and products.
Wide inverse construction caches a dense pseudoinverse of R; its construction
temporaries use `scratch_requirements(FaerLinearOperator, F)` separately.
"""
function scratch_requirements(
    ::typeof(*),
    op::FaerLinearOperator,
    a::Union{_OperatorMatrix,_OperatorVector};
    team=nothing,
)
    lock(_matrix_lock) do
        _handle(op)
        a isa _RotationArray && _rotation_open(a)
        size(op, 2) == size(a, 1) ||
            throw(DimensionMismatch("operator and RHS dimensions must match"))
        P = promote_type(eltype(op), _reflector_type(eltype(a)))
        return _operator_workspace_query(
            _operator_coerce(op, P), ndims(a) == 1 ? 1 : size(a, 2); team
        )
    end
end
function scratch_requirements(
    ::typeof(*), a::_OperatorMatrix, op::FaerLinearOperator; team=nothing
)
    lock(_matrix_lock) do
        _handle(op)
        a isa _RotationArray && _rotation_open(a)
        size(a, 2) == size(op, 1) ||
            throw(DimensionMismatch("operator/source product dimensions must match"))
        P = promote_type(eltype(op), _reflector_type(eltype(a)))
        return _operator_workspace_query(adjoint(_operator_coerce(op, P)), size(a, 1); team)
    end
end
function _operator_mul_query(dest, op, a, alpha, beta, right; team=nothing)
    lock(_matrix_lock) do
        P = _operator_mul_prepare(dest, op, a, alpha, beta, right)
        effective = _operator_coerce(op, P)
        right && (effective = adjoint(effective))
        columns = if right
            size(a, 1)
        elseif ndims(a) == 1
            1
        else
            size(a, 2)
        end
        return _operator_workspace_query(effective, columns; team)
    end
end
function scratch_requirements(
    ::typeof(LinearAlgebra.mul!),
    dest::Union{_OperatorMatrix,_OperatorVector},
    op::FaerLinearOperator,
    a::Union{_OperatorMatrix,_OperatorVector},
    alpha::Number=1,
    beta::Number=0;
    team=nothing,
)
    return _operator_mul_query(dest, op, a, alpha, beta, false; team)
end
function scratch_requirements(
    ::typeof(LinearAlgebra.mul!),
    dest::_OperatorMatrix,
    a::_OperatorMatrix,
    op::FaerLinearOperator,
    alpha::Number=1,
    beta::Number=0;
    team=nothing,
)
    return _operator_mul_query(dest, op, a, alpha, beta, true; team)
end
function _new_operator_workspace(fun, args...; team=nothing)
    req = scratch_requirements(fun, args...; team)
    return FaerScratch(req.bytes; alignment=req.alignment)
end
function FaerScratch(
    fun::typeof(*),
    op::FaerLinearOperator,
    a::Union{_OperatorMatrix,_OperatorVector};
    team=nothing,
)
    return _new_operator_workspace(fun, op, a; team)
end
function FaerScratch(
    fun::typeof(*), a::_OperatorMatrix, op::FaerLinearOperator; team=nothing
)
    return _new_operator_workspace(fun, a, op; team)
end
function FaerScratch(
    fun::typeof(LinearAlgebra.mul!),
    dest::Union{_OperatorMatrix,_OperatorVector},
    op::FaerLinearOperator,
    a::Union{_OperatorMatrix,_OperatorVector},
    args::Number...;
    team=nothing,
)
    return _new_operator_workspace(fun, dest, op, a, args...; team)
end
function FaerScratch(
    fun::typeof(LinearAlgebra.mul!),
    dest::_OperatorMatrix,
    a::_OperatorMatrix,
    op::FaerLinearOperator,
    args::Number...;
    team=nothing,
)
    return _new_operator_workspace(fun, dest, a, op, args...; team)
end
function _reserve_operator_workspace(s, fun, args...; team=nothing)
    req = scratch_requirements(fun, args...; team)
    return reserve!(s, req.bytes; alignment=req.alignment)
end
function reserve!(
    s::FaerScratch,
    fun::typeof(*),
    op::FaerLinearOperator,
    a::Union{_OperatorMatrix,_OperatorVector};
    team=nothing,
)
    return _reserve_operator_workspace(s, fun, op, a; team)
end
function reserve!(
    s::FaerScratch, fun::typeof(*), a::_OperatorMatrix, op::FaerLinearOperator; team=nothing
)
    return _reserve_operator_workspace(s, fun, a, op; team)
end
function reserve!(
    s::FaerScratch,
    fun::typeof(LinearAlgebra.mul!),
    dest::Union{_OperatorMatrix,_OperatorVector},
    op::FaerLinearOperator,
    a::Union{_OperatorMatrix,_OperatorVector},
    args::Number...;
    team=nothing,
)
    return _reserve_operator_workspace(s, fun, dest, op, a, args...; team)
end
function reserve!(
    s::FaerScratch,
    fun::typeof(LinearAlgebra.mul!),
    dest::_OperatorMatrix,
    a::_OperatorMatrix,
    op::FaerLinearOperator,
    args::Number...;
    team=nothing,
)
    return _reserve_operator_workspace(s, fun, dest, a, op, args...; team)
end
function _with_operator_workspace(f, op, columns; scratch=nothing, team=nothing)
    req = _operator_workspace_query(op, columns; team)
    temporary = isnothing(scratch)
    work = temporary ? FaerScratch(req.bytes; alignment=req.alignment) : scratch
    work isa FaerScratch || throw(ArgumentError("scratch must be a FaerScratch"))
    try
        cap = scratch_capacity(work)
        cap.bytes >= req.bytes && cap.alignment >= req.alignment || throw(
            ArgumentError(
                "operator scratch capacity/alignment is insufficient; reserve! for the working scalar/team",
            ),
        )
        return f(work)
    finally
        temporary && close(work)
    end
end
function _operator_workspace_result(op, rhs; scratch=nothing, team=nothing)
    return _with_operator_workspace(op, size(rhs, 2); scratch, team) do work
        dims = (size(op, 1), size(rhs, 2))
        T = eltype(op)
        args = (
            op, rhs, work, UInt(size(op, 1)), UInt(size(op, 2)), UInt(size(rhs, 2)), _tag(T)
        )
        f = RustCrate.operator_workspace
        args = (args..., team)
        return if T === Float64
            _matrix_result(f, dims, args...)
        else
            _typed_result(T, f, dims, args...)
        end
    end
end
