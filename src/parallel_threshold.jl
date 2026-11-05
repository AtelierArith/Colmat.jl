"""
    FaerParThreshold()
    FaerParThreshold(:derived)
    FaerParThreshold(n::Integer)
    FaerParThreshold(:never)

Select faer's native parallel-region policy. The default `:derived` uses its
cost model; an integer sets a fixed floor in the kernel family's work units.
Zero and one select the same fixed policy for every positive-work region.
The fixed floor must satisfy `0 <= n < typemax(UInt)`. `:never` selects
sequential numerical execution. A policy does not create a team.

Currently `bidiagonalize[!]` and their workspace queries accept `par_threshold`.
The fixed floor counts elements in the reduction's updated trailing block;
explicit basis applications retain their own parallel rule. `:never` also
keeps U/V formation sequential. A supplied team must still be open.
"""
struct FaerParThreshold
    value::UInt
    function FaerParThreshold(mode::Symbol = :derived)
        mode===:derived && return new(UInt(0))
        mode===:never && return new(typemax(UInt))
        throw(
            ArgumentError(
                "parallel policy must be :derived, :never, or a nonnegative integer floor",
            ),
        )
    end
    function FaerParThreshold(n::Integer)
        0<=n<typemax(UInt) ||
            throw(ArgumentError("parallel floor must be between zero and typemax(UInt)-1"))
        return new(max(UInt(1), UInt(n)))
    end
end
function Base.show(io::IO, p::FaerParThreshold)
    print(io, "FaerParThreshold(")
    p.value==0 ? print(io, ":derived") :
    p.value==typemax(UInt) ? print(io, ":never") : print(io, p.value)
    print(io, ")")
end
_par_threshold(p::FaerParThreshold) = p
_par_threshold(p) = throw(ArgumentError("par_threshold must be a FaerParThreshold"))
