function _hessenberg_options(par_threshold, blocking_threshold)
    policy=_par_threshold(par_threshold)
    0<=blocking_threshold<=typemax(UInt)||throw(
        ArgumentError("blocking_threshold must be between zero and typemax(UInt)"),
    )
    return policy, UInt(blocking_threshold)
end
function _hessenberg_requirement(n, T, team, compact, policy, blocking)
    info=zeros(UInt, 2)
    fn=compact ? RustCrate.compact_hessenberg_tuned_requirements :
       RustCrate.hessenberg_tuned_requirements
    _native_call(fn, UInt(n), _tag(T), policy.value, blocking, team, info)
    return (bytes = Int(info[1]), alignment = Int(info[2]))
end
function _tuned_hessenberg(
    a;
    scratch = nothing,
    team = nothing,
    compact = false,
    par_threshold = FaerParThreshold(),
    blocking_threshold = 256*256,
)
    lock(_matrix_lock) do
        _iterative_resource_check(scratch, team)
        n=_schur_plan(a, nothing, team)
        policy, blocking=_hessenberg_options(par_threshold, blocking_threshold)
        T=eltype(a)
        req=_hessenberg_requirement(n, T, team, compact, policy, blocking)
        work=isnothing(scratch) ? FaerScratch(req.bytes; alignment = req.alignment) :
             scratch
        source=nothing
        h=nothing
        q=nothing
        success=false
        try
            cap=scratch_capacity(work)
            cap.bytes>=req.bytes&&cap.alignment>=req.alignment||throw(
                ArgumentError(
                    "Hessenberg scratch capacity/alignment is insufficient; call reserve! with the same options",
                ),
            )
            source=a isa FaerMatrixView ? copy(a) : a
            if compact
                h=_zeros(T, (n, n))
                owner=_native_call(
                    RustCrate.compact_hessenberg_tuned,
                    source,
                    h,
                    work,
                    UInt(n),
                    _tag(T),
                    policy.value,
                    blocking,
                    team,
                )
                q=_compact_q(T, owner; wrapper = FaerHessenbergQ)
                result=FaerCompactHessenberg(h, q)
            else
                q=_zeros(T, (n, n))
                args=(source, q, work, UInt(n), _tag(T), policy.value, blocking, team)
                h=T===Float64 ?
                  _matrix_result(RustCrate.hessenberg_tuned_workspace, (n, n), args...) :
                  _typed_result(T, RustCrate.hessenberg_tuned_workspace, (n, n), args...)
                result=FaerHessenberg(h, q)
            end
            success=true
            return result
        finally
            isnothing(source)||source===a||close(source)
            if !success
                isnothing(h)||close(h)
                isnothing(q)||close(q)
            end
            isnothing(scratch)&&close(work)
        end
    end
end
