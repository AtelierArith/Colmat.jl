"""
    scratch_requirements(pinv, F::FaerSpectralFactor; team=nothing)

Query workspace for pseudoinverse reconstruction from retained native SVD/EVD
factors. No factorization is repeated. Execution and query use the actual team
conditions. The independent result matrix allocates outside this buffer.
"""
function scratch_requirements(
    ::typeof(LinearAlgebra.pinv),
    f::FaerSpectralFactor{T,K};
    team = nothing,
) where {T,K}
    lock(_matrix_lock) do
        _handle(f)
        isnothing(team) ||
            team isa FaerTeam ||
            throw(ArgumentError("team must be a FaerTeam"))
        !isnothing(team) && _handle(team)
        info=zeros(UInt, 2)
        args=(f, UInt(size(f, 1)), UInt(size(f, 2)), _tag(T), K===:eigen, info)
        GC.@preserve info begin
            _native_call(RustCrate.spectral_pinv_requirements, args..., team)
        end
        return (bytes = Int(info[1]), alignment = Int(info[2]))
    end
end
function FaerScratch(fun::typeof(LinearAlgebra.pinv), f::FaerSpectralFactor; kwargs...)
    req=scratch_requirements(fun, f; kwargs...)
    return FaerScratch(req.bytes; alignment = req.alignment)
end
function reserve!(
    s::FaerScratch,
    fun::typeof(LinearAlgebra.pinv),
    f::FaerSpectralFactor;
    kwargs...,
)
    req=scratch_requirements(fun, f; kwargs...)
    return reserve!(s, req.bytes; alignment = req.alignment)
end
function _spectral_pinv_with_scratch(
    f::FaerSpectralFactor{T,K},
    scratch;
    atol,
    rtol,
    team,
) where {T,K}
    lock(_matrix_lock) do
        _handle(f)
        scratch isa FaerScratch || throw(ArgumentError("scratch must be a FaerScratch"))
        at, rt=_spectral_tolerances(atol, rtol)
        req=scratch_requirements(LinearAlgebra.pinv, f; team)
        cap=scratch_capacity(scratch)
        cap.bytes>=req.bytes && cap.alignment>=req.alignment || throw(
            ArgumentError(
                "spectral pseudoinverse scratch capacity/alignment is insufficient; call reserve!",
            ),
        )
        dims=reverse(size(f))
        args=(
            f,
            scratch,
            UInt(dims[1]),
            UInt(dims[2]),
            _tag(T),
            K===:eigen,
            at,
            rt,
            promote_type(typeof(rtol), _realtype(T))===Float32,
        )
        fun=RustCrate.spectral_pinv_workspace
        tail=(team,)
        return T===Float64 ? _matrix_result(fun, dims, args..., tail...) :
               _typed_result(T, fun, dims, args..., tail...)
    end
end
