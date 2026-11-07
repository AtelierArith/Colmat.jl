_schur_validate(a::_OwnedCollection) = _dense_rhs_validate(a)
function _schur_validate(a::FaerMatrix)
    _native_call(
        RustCrate.schur_input_validate,
        a,
        UInt(size(a, 1)),
        UInt(size(a, 2)),
        _tag(eltype(a)),
    )
end
function _schur_validate(a::FaerMatrixView)
    owner=_view_owner(a)
    eltype(a)===eltype(owner) || throw(ArgumentError("Schur view scalar metadata mismatch"))
    _schur_validate(owner)
end
function _schur_plan(a, initial, team)
    isnothing(team)||team isa FaerTeam || throw(ArgumentError("team must be a FaerTeam"))
    isnothing(team)||_handle(team)
    _schur_validate(a)
    n=LinearAlgebra.checksquare(a)
    if !isnothing(initial)
        _schur_validate(initial.Q)
        size(initial.Q)==(n, n) || throw(DimensionMismatch("H and Q dimensions must match"))
    end
    n
end
function _schur_requirement(n, T, stage, team)
    info=zeros(UInt, 2)
    GC.@preserve info begin
        args=(UInt(n), _tag(T), Int32(stage), pointer(info))
        _native_call(RustCrate.schur_construction_requirements, args..., team)
    end
    (bytes = Int(info[1]), alignment = Int(info[2]))
end
"""
    scratch_requirements(schur, A::Union{FaerMatrix,FaerMatrixView}; team=nothing)
    scratch_requirements(hessenberg, A::Union{FaerMatrix,FaerMatrixView}; team=nothing)
    scratch_requirements(schur, F::FaerHessenberg; team=nothing)

Query declared native temporary storage for single-matrix Hessenberg/Schur
construction. Corresponding bang methods have the same requirement. The query
and execution validate current native metadata and use the same actual team.
Householder coefficients and temporary eigenvalue columns use caller scratch;
owned outputs, source/view copies, Julia eigenvalues and staged writeback allocate
separately. The selected faer may allocate its internal deep Householder partial
buffer, whose external control awaits unmerged faer #267.
For `hessenberg[!](; compact=true)`, pass the same keyword here and to
`FaerScratch`/`reserve!`; only reduction-kernel scratch is declared, while the
returned packed reflectors and block factors own separate storage.
For general `hessenberg[!]`, pass the same `par_threshold` and
`blocking_threshold` to query and compute. `FaerParThreshold(:never)` uses the
sequential requirement even with an explicit team. Nondefault Hessenberg tuning
is rejected for direct `schur` queries.
"""
function scratch_requirements(
    fun::_QzConstructionFunction,
    a::_BorrowMatrix;
    team = nothing,
    compact::Bool = false,
    par_threshold = FaerParThreshold(),
    blocking_threshold::Integer = 256*256,
)
    lock(_matrix_lock) do
        n=_schur_plan(a, nothing, team)
        stage=fun in (LinearAlgebra.hessenberg, LinearAlgebra.hessenberg!) ? 1 : 0
        policy, blocking = _hessenberg_options(par_threshold, blocking_threshold)
        if stage==1
            return _hessenberg_requirement(n, eltype(a), team, compact, policy, blocking)
        end
        compact&&throw(ArgumentError("compact is supported for Hessenberg reduction"))
        (policy.value==0&&blocking==256*256)||throw(
            ArgumentError("Hessenberg tuning is supported for hessenberg[!] queries"),
        )
        _schur_requirement(n, eltype(a), stage, team)
    end
end
function scratch_requirements(
    fun::Union{typeof(LinearAlgebra.schur),typeof(LinearAlgebra.schur!)},
    f::FaerHessenberg;
    team = nothing,
)
    lock(_matrix_lock) do
        n=_schur_plan(f.H, f, team)
        _schur_requirement(n, eltype(f.H), 2, team)
    end
end
function FaerScratch(fun::_QzConstructionFunction, a::_BorrowMatrix; kwargs...)
    req=scratch_requirements(fun, a; kwargs...)
    return FaerScratch(req.bytes; alignment = req.alignment)
end
function reserve!(
    work::FaerScratch,
    fun::_QzConstructionFunction,
    a::_BorrowMatrix;
    kwargs...,
)
    req=scratch_requirements(fun, a; kwargs...)
    return reserve!(work, req.bytes; alignment = req.alignment)
end
function _schur_with_workspace(f, a, initial, stage, scratch, team)
    isnothing(scratch)||scratch isa FaerScratch ||
        throw(ArgumentError("scratch must be a FaerScratch"))
    n=_schur_plan(a, initial, team)
    req=_schur_requirement(n, eltype(a), stage, team)
    work=isnothing(scratch) ? FaerScratch(req.bytes; alignment = req.alignment) : scratch
    try
        cap=scratch_capacity(work)
        cap.bytes>=req.bytes&&cap.alignment>=req.alignment || throw(
            ArgumentError(
                "Schur construction scratch capacity/alignment is insufficient; call reserve! before computing",
            ),
        )
        f(n, work)
    finally
        isnothing(scratch) && close(work)
    end
end
function _schur_native_result(a::FaerMatrix{T}, initial, stage, n, work, team) where {T}
    q=_zeros(T, (n, n))
    values=Vector{Complex{_realtype(T)}}(undef, stage==1 ? 0 : n)
    iq=isnothing(initial) ? a : initial.Q
    fn=RustCrate.schur_construction_workspace
    try
        h=GC.@preserve values begin
            args=(
                a,
                iq,
                q,
                work,
                Ptr{UInt8}(pointer(values)),
                Int32(stage),
                UInt(n),
                _tag(T),
            )
            args=(args..., team)
            T===Float64 ? _matrix_result(fn, (n, n), args...) :
            _typed_result(T, fn, (n, n), args...)
        end
        stage==1 ? FaerHessenberg(h, q) : LinearAlgebra.Schur(h, q, values)
    catch
        close(q)
        rethrow()
    end
end
function _native_schur(
    a::_BorrowMatrix,
    only_hessenberg = false;
    scratch = nothing,
    team = nothing,
)
    lock(_matrix_lock) do
        stage=only_hessenberg ? 1 : 0
        _schur_with_workspace(a, nothing, stage, scratch, team) do n, work
            source=a isa FaerMatrixView ? copy(a) : a
            try
                _schur_native_result(source, nothing, stage, n, work, team)
            finally
                source===a || close(source)
            end
        end
    end
end
function _saved_schur(f::FaerHessenberg; scratch = nothing, team = nothing)
    lock(_matrix_lock) do
        _schur_with_workspace(f.H, f, 2, scratch, team) do n, work
            _schur_native_result(f.H, f, 2, n, work, team)
        end
    end
end
