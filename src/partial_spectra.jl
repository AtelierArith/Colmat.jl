"""
    partial_eigen(op::FaerLinearOperator, nev; v0, tol, min_dim=0,
                  max_dim=0, max_restarts=1000, scratch=nothing, team=nothing)
    partial_self_adjoint_eigen(op::FaerLinearOperator, nev; kwargs...)

Estimate largest-magnitude eigenpairs through faer's restarted partial spectrum
algorithms. The self-adjoint variant assumes the operator is Hermitian.
Return values, vectors, nconverged, requested, residuals and algorithm. Only
converged columns are returned; max_restarts=0 may return none. Residuals are
independently recomputed as LinearAlgebra.norm(A*v-lambda*v). Eigenpairs are sorted by
magnitude. v0 defaults to a deterministic nonzero vector; use another start if
it is orthogonal to a desired invariant subspace.

Small/full-subspace self-adjoint requests use native dense eigen as an explicit
workaround for faer #204; algorithm reports :dense_fallback in that case.
Projection dimension controls are lower bounds in faer (at least 64 and 2nev).
`scratch` reuses caller-owned native kernel/materialization and final residual
workspace; `team` supplies the scoped native team. Query with
`scratch_requirements(partial_eigen,op,nev;min_dim,max_dim,max_restarts,team)`.
Result staging, promotion and Julia callback buffers allocate separately.
"""
function partial_eigen end
function partial_self_adjoint_eigen end
"""
    partial_svd(op::FaerLinearOperator, nsv; kwargs...)

Estimate largest singular triplets using faer's partial SVD. Keywords match
[`partial_eigen`](@ref). Return U, S, Vt, nconverged, requested, residuals and
algorithm. Residuals check both A*v=s*u and A'*u=s*v. Only reported converged
triplets are returned. Full-projection requests use a native dense SVD fallback; algorithm reports
:dense_fallback. Other requests do not materialize the operator.
"""
function partial_svd end
function _partial_spectrum(
    op::FaerLinearOperator{T},
    k::Integer,
    kind::Int32;
    v0 = nothing,
    tol = sqrt(eps(_realtype(T))),
    min_dim::Integer = 0,
    max_dim::Integer = 0,
    max_restarts::Integer = 1000,
    scratch = nothing,
    team = nothing,
) where {T}
    lock(_matrix_lock) do
        _iterative_resource_check(scratch, team)
        m, n, k, tolerance, start=_partial_validate(
            op,
            k,
            kind;
            v0,
            tol,
            min_dim,
            max_dim,
            max_restarts,
        )
        req=_partial_workspace_query(op, kind, k; min_dim, max_dim, max_restarts, team)
        if !isnothing(scratch)
            capacity=scratch_capacity(scratch)
            capacity.bytes>=req.bytes && capacity.alignment>=req.alignment ||
                throw(ArgumentError("insufficient partial spectrum scratch; call reserve!"))
        end
        work=isnothing(scratch) ? FaerScratch(req.bytes; alignment = req.alignment) :
             scratch
        try
            E=kind==0 ? Complex{_realtype(T)} : T
            vectors=zeros(E, m, k)
            values=zeros(E, k)
            right=zeros(T, n, kind==2 ? k : 0)
            algorithm=:partial
            count=0
            if k>0 && max_restarts>0
                algorithm=kind in (1, 2) && max(Int(max_dim), 64, 2k)>=min(m, n) ?
                          :dense_fallback : :partial
                GC.@preserve start vectors values right begin
                    args=(
                        kind,
                        UInt(k),
                        UInt(m),
                        UInt(n),
                        _tag(T),
                        Ptr{UInt8}(pointer(start)),
                        Float64(tolerance),
                        UInt(min_dim),
                        UInt(max_dim),
                        UInt(max_restarts),
                        Ptr{UInt8}(pointer(vectors)),
                        Ptr{UInt8}(pointer(values)),
                        Ptr{UInt8}(pointer(right)),
                    )
                    count=Int(_partial_native(op, args, work, team))
                end
            end
            order=sortperm(abs.(values[1:count]); rev = true)
            vectors=vectors[:, order]
            values=values[order]
            if kind==2
                right=right[:, order]
                singular=real.(values)
                residuals=[
                    max(
                        LinearAlgebra.norm(
                            _partial_apply(op, right[:, j], work, team)-singular[j]*vectors[
                                :,
                                j,
                            ],
                        ),
                        LinearAlgebra.norm(
                            _partial_apply(adjoint(op), vectors[:, j], work, team)-singular[j]*right[
                                :,
                                j,
                            ],
                        ),
                    ) for j = 1:count
                ]
                return (
                    U = vectors,
                    S = singular,
                    Vt = Matrix(adjoint(right)),
                    nconverged = count,
                    requested = k,
                    residuals,
                    algorithm,
                )
            end
            residuals=[
                LinearAlgebra.norm(
                    _partial_apply(op, vectors[:, j], work, team)-values[j]*vectors[:, j],
                ) for j = 1:count
            ]
            return (
                values = kind==1 ? real.(values) : values,
                vectors,
                nconverged = count,
                requested = k,
                residuals,
                algorithm,
            )
        finally
            isnothing(scratch) && close(work)
        end
    end
end
partial_eigen(op::FaerLinearOperator, k::Integer; kwargs...) =
    _partial_spectrum(op, k, Int32(0); kwargs...)
partial_self_adjoint_eigen(op::FaerLinearOperator, k::Integer; kwargs...) =
    _partial_spectrum(op, k, Int32(1); kwargs...)
partial_svd(op::FaerLinearOperator, k::Integer; kwargs...) =
    _partial_spectrum(op, k, Int32(2); kwargs...)
