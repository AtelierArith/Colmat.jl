# Native iteration and residual kernels share one caller buffer sequentially.
const _IterativeFunction = Union{typeof(cg),typeof(bicgstab),typeof(lsmr)}
const _IterativeBangFunction = Union{typeof(cg!),typeof(bicgstab!),typeof(lsmr!)}
_iterative_kind(::Union{typeof(cg),typeof(cg!)}) = 0
_iterative_kind(::Union{typeof(bicgstab),typeof(bicgstab!)}) = 1
_iterative_kind(::Union{typeof(lsmr),typeof(lsmr!)}) = 2
function _iterative_resource_check(scratch, team)
    isnothing(scratch) ||
        scratch isa FaerScratch ||
        throw(ArgumentError("scratch must be a FaerScratch"))
    isnothing(scratch) || _handle(scratch)
    isnothing(team) || team isa FaerTeam || throw(ArgumentError("team must be a FaerTeam"))
    isnothing(team) || _handle(team)
    return nothing
end
function _iterative_preconditioners(op, m, n)
    T = eltype(op)
    left = m isa FaerLinearOperator ? m : identity_preconditioner(size(op, 2); T)
    right = n isa FaerLinearOperator ? n : identity_preconditioner(size(op, 2); T)
    return left, right
end
function _iterative_prepared_query(kind, op, m, n, columns, team)
    _handle(op)
    _handle(m)
    _handle(n)
    info = zeros(UInt, 2)
    args = (
        op,
        m,
        n,
        Int32(kind),
        UInt(columns),
        UInt(size(op, 1)),
        UInt(size(op, 2)),
        _tag(eltype(op)),
        info,
    )
    GC.@preserve info begin
        _native_call(RustCrate.iterative_requirements, args..., team)
    end
    return (bytes=Int(info[1]), alignment=Int(info[2]))
end
function _iterative_resource_prepare(kind, op, m, n, columns, scratch, team)
    isnothing(scratch) && isnothing(team) && return nothing
    _iterative_resource_check(scratch, team)
    m, n = _iterative_preconditioners(op, m, n)
    req = _iterative_prepared_query(kind, op, m, n, columns, team)
    if !isnothing(scratch)
        cap = scratch_capacity(scratch)
        cap.bytes >= req.bytes && cap.alignment >= req.alignment ||
            throw(ArgumentError("insufficient iterative scratch; call reserve!"))
    end
    return (; m, n, req)
end
function _iterative_solution(args...)
    task = current_task()
    sticky = task.sticky
    task.sticky = true
    try
        return _iterative_solution_impl(args...)
    finally
        task.sticky = sticky
    end
end
function _iterative_solution_impl(context, kind, P, legacy, dims, args, scratch, team)
    if isnothing(context)
        return if P === Float64
            _matrix_result(legacy, dims, args...)
        else
            _typed_result(P, legacy, dims, args...)
        end
    end
    # Legacy argument layouts contain optional raw owners; the new entry point
    # receives checked operator/preconditioner references for all three slots.
    source = args[1]
    tail = kind == 1 ? args[4:end] : args[3:end]
    work = if isnothing(scratch)
        FaerScratch(context.req.bytes; alignment=context.req.alignment)
    else
        scratch
    end
    try
        native = RustCrate.iterative_workspace
        callargs = (source, context.m, context.n, Int32(kind), tail..., work, team)
        return if P === Float64
            _matrix_result(native, dims, callargs...)
        else
            _typed_result(P, native, dims, callargs...)
        end
    finally
        isnothing(scratch) && close(work)
    end
end
function _iterative_query(
    fun, A, b; M=nothing, N=nothing, x0=nothing, team=nothing, destination=nothing
)
    lock(_matrix_lock) do
        _iterative_resource_check(nothing, team)
        kind = _iterative_kind(fun)
        kind == 1 ||
            isnothing(N) ||
            throw(ArgumentError("only BiCGSTAB accepts a right preconditioner N"))
        _dense_rhs_validate(b)
        isnothing(x0) || _dense_rhs_validate(x0)
        op = A isa FaerLinearOperator ? A : FaerLinearOperator(A)
        _handle(op)
        kind == 2 ||
            size(op, 1) == size(op, 2) ||
            throw(DimensionMismatch("iterative operator must be square"))
        size(b, 1) == size(op, 1) ||
            throw(DimensionMismatch("iterative RHS rows must match operator"))
        expected = b isa AbstractVector ? (size(op, 2),) : (size(op, 2), size(b, 2))
        isnothing(x0) ||
            size(x0) == expected ||
            throw(DimensionMismatch("iterative initial shape must match solution"))
        m = if isnothing(M)
            nothing
        elseif M isa FaerLinearOperator
            M
        else
            FaerLinearOperator(M)
        end
        n = if isnothing(N)
            nothing
        elseif N isa FaerLinearOperator
            N
        else
            FaerLinearOperator(N)
        end
        for p in (m, n)
            isnothing(p) && continue
            _handle(p)
            size(p) == (size(op, 2), size(op, 2)) ||
                throw(DimensionMismatch("iterative preconditioner shape mismatch"))
        end
        P = promote_type(
            eltype(op),
            _reflector_type(eltype(b)),
            isnothing(x0) ? eltype(op) : _reflector_type(eltype(x0)),
            isnothing(m) ? eltype(op) : eltype(m),
            isnothing(n) ? eltype(op) : eltype(n),
        )
        isnothing(destination) ||
            eltype(destination) === P ||
            throw(ArgumentError("iterative destination must match promoted scalar type"))
        op = _operator_coerce(op, P)
        m = isnothing(m) ? nothing : _operator_coerce(m, P)
        n = isnothing(n) ? nothing : _operator_coerce(n, P)
        m, n = _iterative_preconditioners(op, m, n)
        return _iterative_prepared_query(
            kind, op, m, n, b isa AbstractVector ? 1 : size(b, 2), team
        )
    end
end
"""
    scratch_requirements(cg|bicgstab|lsmr, A, b; M=nothing, N=nothing, x0=nothing, team=nothing)
    scratch_requirements(cg!|bicgstab!|lsmr!, x, A, b; M=nothing, N=nothing, team=nothing)

Query native iteration scratch and independent residual/normal-residual scratch
at the promoted scalar type and actual team shape. N is BiCGSTAB-only. RHS/result
ownership, operator promotion, Julia callback arrays and history storage allocate
separately. Queries do not apply operators or invoke iteration callbacks.
"""
function scratch_requirements(
    fun::_IterativeFunction,
    A,
    b::Union{AbstractVector{<:Number},AbstractMatrix{<:Number}};
    kwargs...,
)
    return _iterative_query(fun, A, b; kwargs...)
end
function scratch_requirements(
    fun::_IterativeBangFunction,
    x::_HouseholderArray,
    A,
    b::Union{AbstractVector{<:Number},AbstractMatrix{<:Number}};
    kwargs...,
)
    return _iterative_query(fun, A, b; x0=x, destination=x, kwargs...)
end
function FaerScratch(
    fun::Union{_IterativeFunction,_IterativeBangFunction}, args...; kwargs...
)
    req = scratch_requirements(fun, args...; kwargs...)
    return FaerScratch(req.bytes; alignment=req.alignment)
end
function reserve!(
    s::FaerScratch,
    fun::Union{_IterativeFunction,_IterativeBangFunction},
    args...;
    kwargs...,
)
    req = scratch_requirements(fun, args...; kwargs...)
    return reserve!(s, req.bytes; alignment=req.alignment)
end
