"""
    FaerScratch(bytes=0; alignment=64)
    FaerScratch(pfaffian, A::FaerMatrix)

Own an aligned native workspace for reuse across operations. `reserve!` grows
capacity/alignment explicitly; computations never grow the buffer automatically.
Pfaffian, SVD and single-matrix EVD consume `scratch=workspace`; an optional
FaerTeam supplies parallel execution. The workspace is mutably
borrowed during the native call and operations are serialized by the owner lock.
Copy/deepcopy allocate independent empty workspaces of the same capacity; their
uninitialized contents are not part of the public state. Close releases storage.
"""
mutable struct FaerScratch{H}
    handle::Union{Nothing,H}
    FaerScratch{H}(handle::H) where {H} = new{H}(handle)
end
function _handle(s::FaerScratch)
    h=s.handle
    isnothing(h) && throw(ArgumentError("native scratch has been closed"))
    return h
end
_native_argument(s::FaerScratch) = _handle(s)
_is_native_owner(::FaerScratch) = true
function FaerScratch(bytes::Integer = 0; alignment::Integer = 64)
    n, align=Int(bytes), Int(alignment)
    n>=0 || throw(ArgumentError("scratch byte capacity must be nonnegative"))
    ispow2(align) ||
        throw(ArgumentError("scratch alignment must be a positive power of two"))
    lock(_matrix_lock) do
        h=RustCrate.OwnedScratch()
        _native_call(RustCrate.scratch_reserve, h, UInt(n), UInt(align))
        return FaerScratch{typeof(h)}(h)
    end
end
"""Return `(bytes, alignment)` of an open native workspace."""
function scratch_capacity(s::FaerScratch)
    lock(_matrix_lock) do
        info=zeros(UInt, 2)
        GC.@preserve info _native_call(RustCrate.scratch_info, s, pointer(info))
        return (bytes = Int(info[1]), alignment = Int(info[2]))
    end
end
"""
    scratch_requirements(pfaffian, A::FaerMatrix)

Query faer's exact workspace byte count and alignment for A's scalar type and
order. A must be square and open; query does not read its coefficients.
"""
function scratch_requirements(::typeof(pfaffian), a::FaerMatrix{T}) where {T}
    lock(_matrix_lock) do
        _handle(a)
        size(a, 1)==size(a, 2) || throw(DimensionMismatch("Pfaffian requires square input"))
        info=zeros(UInt, 2)
        GC.@preserve info _native_call(
            RustCrate.pfaffian_requirements,
            UInt(size(a, 1)),
            _tag(T),
            pointer(info),
        )
        return (bytes = Int(info[1]), alignment = Int(info[2]))
    end
end
function FaerScratch(::typeof(pfaffian), a::FaerMatrix)
    req=scratch_requirements(pfaffian, a)
    return FaerScratch(req.bytes; alignment = req.alignment)
end
function reserve!(s::FaerScratch, bytes::Integer; alignment::Integer = 64)
    n, align=Int(bytes), Int(alignment)
    n>=0 || throw(ArgumentError("scratch byte capacity must be nonnegative"))
    ispow2(align) ||
        throw(ArgumentError("scratch alignment must be a positive power of two"))
    _native_call(RustCrate.scratch_reserve, s, UInt(n), UInt(align))
    return s
end
function reserve!(s::FaerScratch, ::typeof(pfaffian), a::FaerMatrix)
    req=scratch_requirements(pfaffian, a)
    return reserve!(s, req.bytes; alignment = req.alignment)
end
Base.isopen(s::FaerScratch) = lock(()->!isnothing(s.handle), _matrix_lock)
function Base.close(s::FaerScratch)
    lock(_matrix_lock) do
        if !isnothing(s.handle)
            _native_call(RustCrate.scratch_close, s)
            s.handle=nothing
        end
    end
    return nothing
end
function Base.copy(s::FaerScratch)
    cap=scratch_capacity(s)
    return FaerScratch(cap.bytes; alignment = cap.alignment)
end
Base.deepcopy_internal(s::FaerScratch, d::IdDict) = get!(() -> copy(s), d, s)
function _pfaffian_with_scratch(a::FaerMatrix{T}, s::FaerScratch, team) where {T}
    lock(_matrix_lock) do
        req=scratch_requirements(pfaffian, a)
        cap=scratch_capacity(s)
        cap.bytes>=req.bytes && cap.alignment>=req.alignment || throw(
            ArgumentError(
                "scratch capacity/alignment is insufficient; call reserve!(scratch, pfaffian, A)",
            ),
        )
        value=Ref{T}()
        GC.@preserve value begin
            out=Ptr{UInt8}(Base.unsafe_convert(Ptr{T}, value))
            isnothing(team) ? _native_call(RustCrate.pfaffian_workspace, a, s, out) :
            _native_call(RustCrate.pfaffian_workspace_team, a, s, team, out)
        end
        return value[]
    end
end
_pfaffian_with_scratch(a, s, team) = throw(ArgumentError("scratch must be a FaerScratch"))

function _pfaffian_options(a, scratch, team)
    isnothing(team) || team isa FaerTeam || throw(ArgumentError("team must be a FaerTeam"))
    if isnothing(scratch)
        temp=FaerScratch(pfaffian, a)
        try
            return _pfaffian_with_scratch(a, temp, team)
        finally
            close(temp)
        end
    end
    return _pfaffian_with_scratch(a, scratch, team)
end
