"""
    FaerTeam(width=Threads.nthreads())
    FaerTeam(f, width)

Own a reusable faer threadteam reservation. Width counts the calling thread and
width-1 reserved workers. Width one uses sequential execution. `team_mul` and
`mul!(dest, team, A, B[, alpha, beta])` borrow its execution authority only for
the native call; no borrowed team token is stored in Julia. Close releases the
reservation immediately; the do-block form always closes it. Copy/deepcopy
create independent pools, so each copy reserves its own workers.

Explicit multiplication and `pfaffian(A; team=...)` use the team. Other
Colmat methods retain sequential execution. Team operations and owner lifecycle
are serialized by Colmat's matrix lock; this is not a Julia task scheduler.
"""
mutable struct FaerTeam{H}
    handle::Union{Nothing,H}
    const width::Int
    FaerTeam{H}(handle::H, width::Int) where {H} = new{H}(handle, width)
end
function _handle(t::FaerTeam)
    h=t.handle
    isnothing(h) && throw(ArgumentError("faer team has been closed"))
    return h
end
_native_argument(t::FaerTeam) = _handle(t)
_is_native_owner(::FaerTeam) = true
function FaerTeam(width::Integer = Threads.nthreads())
    n=Int(width)
    n>0 || throw(ArgumentError("team width must be positive"))
    lock(_matrix_lock) do
        h=RustCrate.OwnedTeam()
        _native_call(RustCrate.team_initialize, h, UInt(n))
        return FaerTeam{typeof(h)}(h, n)
    end
end
function FaerTeam(f::Function, width::Integer = Threads.nthreads())
    team=FaerTeam(width)
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
    FaerTeam(t.width)
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
        P=promote_type(eltype(dest), eltype(a), eltype(b), typeof(alpha), typeof(beta))
        result=iszero(alpha) ? zeros(P, size(dest)) :
               alpha .* Matrix{P}(team_mul(team, a, b))
        !iszero(beta) && (result .+= beta .* Matrix(dest))
        return copyto!(dest, result)
    end
end
LinearAlgebra.mul!(
    dest::_ProductDense,
    team::FaerTeam,
    a::AbstractMatrix,
    b::AbstractMatrix,
) = LinearAlgebra.mul!(dest, team, a, b, 1, 0)
