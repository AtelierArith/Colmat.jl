"""
    FaerTeam(width=Threads.nthreads())
    FaerTeam(f, width)

Own a reusable faer threadteam reservation. Width counts the calling thread and
width-1 reserved workers. Width one uses sequential execution. `team_mul` and
`mul!(dest, team, A, B[, alpha, beta])` borrow its execution authority only for
the native call; no borrowed team token is stored in Julia. Close releases the
reservation immediately; the do-block form always closes it. Copy/deepcopy
create independent pools, so each copy reserves its own workers.

Explicit multiplication (`team_mul`, dense `mul!(C, A, B[, α, β]; team)` and the
positional form, both written in place), Pfaffian, SVD, single-matrix EVD and
dense LU/QR/Cholesky/LDLT/Bunch–Kaufman construction and reconstruction
(`reconstruct/FaerMatrix/Matrix(F; team)`) use `team=` (including full/column-pivot
LU/QR and staged bang variants; pivoted Cholesky is separate). Other
Colmat methods retain sequential execution; see the resources manual. Team operations and owner lifecycle
are serialized by Colmat's matrix lock; this is not a Julia task scheduler.
"""
mutable struct FaerTeam{H}
    handle::Union{Nothing,H}
    const width::Int
    const options::NamedTuple
    FaerTeam{H}(handle::H, width::Int, options::NamedTuple = (;)) where {H} =
        new{H}(handle, width, options)
end
function _handle(t::FaerTeam)
    h=t.handle
    isnothing(h) && throw(ArgumentError("faer team has been closed"))
    return h
end
_native_argument(t::FaerTeam) = _handle(t)
_is_native_owner(::FaerTeam) = true
const _TEAM_QOS=(default = 0, utility = 1, user_initiated = 2, user_interactive = 3)
function _team_options(; acquire_timeout = 30, spin_polls = nothing, qos = nothing, pin = true)
    t=Float64(acquire_timeout)
    isfinite(t) && t>0 ||
        throw(ArgumentError("acquire_timeout must be a positive finite number of seconds"))
    isnothing(spin_polls) ||
        0<=spin_polls<=typemax(UInt32) ||
        throw(ArgumentError("spin_polls must be in 0:$(typemax(UInt32))"))
    isnothing(qos) ||
        haskey(_TEAM_QOS, qos) ||
        throw(ArgumentError("qos must be one of $(keys(_TEAM_QOS)) or nothing"))
    return (; acquire_timeout = t, spin_polls, qos, pin = Bool(pin))
end
"""
    FaerTeam(width=Threads.nthreads(); acquire_timeout=30, spin_polls=nothing, qos=nothing, pin=true)

Pool options of the threadteam reservation (the defaults are threadteam's,
except the 30 s acquisition timeout used before):

- `acquire_timeout`: seconds to wait for idle workers when the team is
  reserved, before construction fails.
- `spin_polls`: spin budget of a team wait (reserved members waiting for work,
  joins and barriers) in pause instructions before yielding and sleeping;
  `0` sleeps right after the yields. `nothing` keeps threadteam's default.
- `qos`: macOS quality-of-service class of the workers, one of `:default`,
  `:utility`, `:user_initiated` or `:user_interactive` (`nothing` keeps
  threadteam's default, user-initiated on macOS). No effect on other systems;
  the calling Julia thread's class is never changed.
- `pin`: place the participants on distinct CPUs of the caller's CPU set during
  team sections (Linux only); pass `false` for a pool that shares its CPUs with
  other work.

A width-one team runs sequentially and builds no pool. `copy` reuses the
options. See [`team_diagnostics`](@ref).
"""
function FaerTeam(width::Integer = Threads.nthreads(); kwargs...)
    n=Int(width)
    n>0 || throw(ArgumentError("team width must be positive"))
    o=_team_options(; kwargs...)
    lock(_matrix_lock) do
        h=RustCrate.OwnedTeam()
        _native_call(
            RustCrate.team_initialize_with,
            h,
            UInt(n),
            o.acquire_timeout,
            UInt32(something(o.spin_polls, 0)),
            Int32(isnothing(o.qos) ? 0 : _TEAM_QOS[o.qos]),
            UInt8(o.pin),
            UInt8[!isnothing(o.spin_polls), !isnothing(o.qos)],
        )
        return FaerTeam{typeof(h)}(h, n, o)
    end
end
function FaerTeam(f::Function, width::Integer = Threads.nthreads(); kwargs...)
    team=FaerTeam(width; kwargs...)
    try
        return f(team)
    finally
        close(team)
    end
end
Base.isopen(t::FaerTeam) = lock(()->!isnothing(t.handle), _matrix_lock)
function Base.close(t::FaerTeam)
    lock(_matrix_lock) do
        if !isnothing(t.handle)
            _native_call(RustCrate.team_close, t)
            t.handle=nothing
        end
    end
    return nothing
end
Base.copy(t::FaerTeam) = lock(_matrix_lock) do
    _handle(t)
    FaerTeam(t.width; t.options...)
end
"""
    team_diagnostics(team::FaerTeam)

Reservation state and wait counters of the team's threadteam pool, counted since
the team was built: `width` (participants actually reserved, at most the
requested width), `spin_polls`, `pin` (whether participants are pinned),
`resident_members` (workers currently kept awake), `pin_skips` (team sections
that placed nobody because the CPU set was narrower than the team),
`join_sleeps`, `barrier_sleeps` and `member_parks` (waits that slept rather than
spun). A width-one team has no pool and reports zeros except `width`.
"""
function team_diagnostics(t::FaerTeam)
    lock(_matrix_lock) do
        v=zeros(UInt64, 8)
        _native_call(RustCrate.team_state, t, v)
        return (
            width = Int(v[1]),
            spin_polls = Int(v[2]),
            pin = !iszero(v[3]),
            resident_members = Int(v[4]),
            pin_skips = Int(v[5]),
            join_sleeps = Int(v[6]),
            barrier_sleeps = Int(v[7]),
            member_parks = Int(v[8]),
        )
    end
end
Base.deepcopy_internal(t::FaerTeam, d::IdDict) = get!(() -> copy(t), d, t)
"""
    team_mul(team::FaerTeam, A::AbstractMatrix, B::AbstractMatrix)

Multiply using the caller-owned reusable team and return an independent
FaerMatrix. Four native scalar types and promotion are supported. Non-owner
inputs, including views and transposes, are snapshotted before execution.
"""
function team_mul(team::FaerTeam, a::AbstractMatrix, b::AbstractMatrix)
    lock(_matrix_lock) do
        _handle(team)
        a isa _RotationArray && _rotation_open(a)
        b isa _RotationArray && _rotation_open(b)
        size(a, 2)==size(b, 1) ||
            throw(DimensionMismatch("team product inner dimensions must match"))
        T=promote_type(_reflector_type(eltype(a)), _reflector_type(eltype(b)))
        x=a isa FaerMatrix ? _coerce(a, T) : FaerMatrix{T}(a)
        y=b isa FaerMatrix ? _coerce(b, T) : FaerMatrix{T}(b)
        dims=(size(a, 1), size(b, 2))
        return T===Float64 ? _matrix_result(RustCrate.team_product, dims, team, x, y) :
               _typed_result(T, RustCrate.team_product, dims, team, x, y)
    end
end
function LinearAlgebra.mul!(
    dest::_ProductDense,
    team::FaerTeam,
    a::AbstractMatrix,
    b::AbstractMatrix,
    alpha::Number,
    beta::Number,
)
    lock(_matrix_lock) do
        _handle(team)
        for x in (dest, a, b)
            x isa _RotationArray && _rotation_open(x)
        end
        size(a, 2)==size(b, 1) ||
            throw(DimensionMismatch("team product inner dimensions must match"))
        size(dest)==(size(a, 1), size(b, 2)) ||
            throw(DimensionMismatch("team product destination size must match"))
        # Written in place by the native gemm on the team (see `_dense_mul!`).
        return _dense_mul!(dest, a, b, alpha, beta; team)
    end
end
LinearAlgebra.mul!(
    dest::_ProductDense,
    team::FaerTeam,
    a::AbstractMatrix,
    b::AbstractMatrix,
) = LinearAlgebra.mul!(dest, team, a, b, 1, 0)
