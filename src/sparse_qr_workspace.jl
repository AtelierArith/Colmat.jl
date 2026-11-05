"""
    scratch_requirements(factorize, plan::FaerSparseSymbolic, A::FaerSparseMatrix; team=nothing)

Query numeric factor-construction workspace for tall/square QR, including explicit simplicial QR. Pass a
reserved buffer to `factorize(plan,A; scratch=s, team=team)`. Query and numeric
execution use the same scoped `FaerTeam`; omission selects sequential execution.
The explicit simplicial kernel is sequential even with a team.
With `team` alone, construction allocates temporary scratch. The numeric QR kernel and
source-precision rank-certificate preparation share this buffer sequentially.
Symbolic analysis, input normalization/snapshot/finite checks and owned numeric
factor storage allocate separately. The stored pattern must match.
"""
function scratch_requirements(
    ::typeof(LinearAlgebra.factorize),
    p::FaerSparseSymbolic{:qr},
    a::FaerSparseMatrix{T};
    team = nothing,
) where {T}
    lock(_matrix_lock) do
        isnothing(team) ||
            team isa FaerTeam ||
            throw(ArgumentError("team must be a FaerTeam"))
        isnothing(team) || _handle(team)
        _sparse_factor_inputs(p, a)
        size(p)==size(a) || throw(DimensionMismatch("plan and matrix shapes must match"))
        info=zeros(UInt, 2)
        GC.@preserve info begin
            isnothing(team) ?
            _native_call(
                RustCrate.sparse_qr_factor_requirements,
                p,
                _tag(T),
                pointer(info),
            ) :
            _native_call(
                RustCrate.sparse_qr_factor_requirements_team,
                p,
                _tag(T),
                pointer(info),
                team,
            )
        end
        (bytes = Int(info[1]), alignment = Int(info[2]))
    end
end
function FaerScratch(
    fun::typeof(LinearAlgebra.factorize),
    p::FaerSparseSymbolic,
    a::FaerSparseMatrix;
    team = nothing,
)
    req=scratch_requirements(fun, p, a; team)
    FaerScratch(req.bytes; alignment = req.alignment)
end
function reserve!(
    s::FaerScratch,
    fun::typeof(LinearAlgebra.factorize),
    p::FaerSparseSymbolic,
    a::FaerSparseMatrix;
    team = nothing,
)
    req=scratch_requirements(fun, p, a; team)
    reserve!(s, req.bytes; alignment = req.alignment)
end
function _sparse_qr_factor_validate(p, a, s, team)
    p isa FaerSparseSymbolic{:qr} ||
        throw(ArgumentError("construction resources currently require tall/square QR"))
    isnothing(s) ||
        s isa FaerScratch ||
        throw(ArgumentError("scratch must be a FaerScratch"))
    req=scratch_requirements(LinearAlgebra.factorize, p, a; team)
    if !isnothing(s)
        cap=scratch_capacity(s)
        cap.bytes>=req.bytes && cap.alignment>=req.alignment ||
            throw(ArgumentError("insufficient factorization scratch; call reserve!"))
    end
    return req
end
function _sparse_qr_factor_workspace(h, p, a, scratch, team = nothing)
    req=_sparse_qr_factor_validate(p, a, scratch, team)
    s=isnothing(scratch) ? FaerScratch(req.bytes; alignment = req.alignment) : scratch
    try
        return Int(
            isnothing(team) ?
            _native_call(RustCrate.sparse_qr_factor_workspace, h, p, a, s) :
            _native_call(RustCrate.sparse_qr_factor_workspace_team, h, p, a, s, team),
        )
    finally
        isnothing(scratch) && close(s)
    end
end
