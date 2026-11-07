"""
    partial_eigen(op::FaerLinearOperator, nev; v0, tol, min_dim=0,
                  max_dim=0, max_restarts=1000, scratch=nothing, team=nothing)
    partial_self_adjoint_eigen(op::FaerLinearOperator, nev; kwargs...)

Estimate largest-magnitude eigenpairs through faer's restarted partial spectrum
algorithms. The self-adjoint variant assumes the operator is Hermitian.
Return values, vectors, nconverged, requested, residuals and algorithm. Only
converged columns are returned; max_restarts=0 may return none. Residuals are
independently recomputed natively as LinearAlgebra.norm(A*v-lambda*v) (a real
operator acts on the real and imaginary parts of complex eigenvectors, so it is
never promoted). Eigenpairs come out of faer sorted by decreasing magnitude.
v0 defaults to a deterministic nonzero vector; use another start if it is
orthogonal to a desired invariant subspace. A `FaerVector`, a one-dimensional
`FaerView`, a `faerview` borrow or a strided Julia vector of the operator's
scalar type is read in place; other `v0` are converted to it once.

Small/full-subspace self-adjoint requests (max(max_dim, 64, 2nev) >= n) take
faer's native full-subspace branch, which materializes the operator and runs a
dense self-adjoint eigendecomposition reading its lower triangle; algorithm
reports :dense_fallback in that case. Results are defined only for Hermitian
operators. If that branch fails (for example a non-finite materialized
operator) it reports nconverged == 0 instead of throwing.
Projection dimension controls are lower bounds in faer (at least 64 and 2nev).
`scratch` reuses caller-owned native kernel/materialization and final residual
workspace; `team` supplies the scoped native team. Query with
`scratch_requirements(partial_eigen,op,nev;min_dim,max_dim,max_restarts,team)`.
faer writes the converged results, already in their final order, straight into
the returned Julia arrays; truncation to the converged count resizes them in
place (they keep the memory of the requested count). Results stay Julia
`Vector`/`Matrix` by contract rather than native owners. Julia callback operators reuse their per-operator marshalling buffers.
"""
function partial_eigen end
function partial_self_adjoint_eigen end
"""
    partial_svd(op::FaerLinearOperator, nsv; kwargs...)

Estimate largest singular triplets using faer's partial SVD. Keywords match
[`partial_eigen`](@ref). Return U, S, Vt, nconverged, requested, residuals and
algorithm. Residuals check both A*v=s*u and A'*u=s*v. Only reported converged
triplets are returned. faer writes the right singular vectors directly into
`Vt`'s rows (conjugated in place), so `Vt` is a plain `Matrix` built without a
transpose copy. Full-projection requests (max(max_dim, 64, 2nsv) >=
min(size(op)...)) take faer's native full-subspace dense SVD branch; algorithm
reports :dense_fallback, and a failure there reports nconverged == 0. Other
requests do not materialize the operator.
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
            R=_realtype(T)
            E=kind==0 ? Complex{R} : T
            # Rust writes sorted results, residuals and (for SVD) the conjugated,
            # compacted Vt rows straight into these buffers; truncating to the
            # converged count is an in-place resize, not a copy.
            vectors=Vector{E}(undef, m*k)
            values=Vector{E}(undef, k)
            right=Vector{T}(undef, kind==2 ? n*k : 0)
            residuals=Vector{R}(undef, k)
            algorithm=:partial
            count=0
            if k>0 && max_restarts>0
                # Mirrors faer's full-subspace trigger max(max_dim, 2MIN_DIM, 2k) >= limit.
                algorithm=kind in (1, 2) && max(Int(max_dim), 64, 2k)>=min(m, n) ?
                          :dense_fallback : :partial
                source, desc=isnothing(start) ? (Ptr{UInt8}(C_NULL), nothing) :
                             _view_source(start)
                GC.@preserve start desc vectors values right residuals begin
                    args=(
                        kind,
                        UInt(k),
                        UInt(m),
                        UInt(n),
                        _tag(T),
                        source,
                        isnothing(desc) ? Ptr{Int}(C_NULL) : pointer(desc),
                        Float64(tolerance),
                        UInt(min_dim),
                        UInt(max_dim),
                        UInt(max_restarts),
                        Ptr{UInt8}(pointer(vectors)),
                        Ptr{UInt8}(pointer(values)),
                        Ptr{UInt8}(pointer(right)),
                        Ptr{UInt8}(pointer(residuals)),
                    )
                    count=Int(_partial_native(op, args, work, team))
                end
            end
            vectors=reshape(resize!(vectors, m*count), m, count)
            resize!(values, count)
            resize!(residuals, count)
            if kind==2
                return (
                    U = vectors,
                    S = T<:Real ? values : real.(values),
                    Vt = reshape(resize!(right, count*n), count, n),
                    nconverged = count,
                    requested = k,
                    residuals,
                    algorithm,
                )
            end
            return (
                values = kind==1 && T<:Complex ? real.(values) : values,
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
