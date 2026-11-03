"""
    partial_eigen(op::FaerLinearOperator, nev; v0, tol, min_dim=0,
                  max_dim=0, max_restarts=1000)
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
) where {T}
    lock(_matrix_lock) do
        _handle(op)
        m, n=size(op)
        kind!=2 && m!=n && throw(DimensionMismatch("eigen operator must be square"))
        0<=k<=min(m, n) ||
            throw(ArgumentError("requested count must be in 0:min(size(op)...)"))
        min_dim>=0 && max_dim>=0 && max_restarts>=0 ||
            throw(ArgumentError("subspace dimensions and restarts must be nonnegative"))
        isfinite(tol) && tol>=0 ||
            throw(ArgumentError("tol must be finite and nonnegative"))
        tolerance=convert(_realtype(T), tol)
        isfinite(tolerance) || throw(ArgumentError("tol exceeds native precision range"))
        k=Int(k)
        limit=kind==2 ? min(m, n) : n
        effective_min=min(max(min_dim, 32, k), limit)
        effective_max=min(max(max_dim, 64, 2k), limit)
        effective_min<=effective_max || throw(
            ArgumentError("min_dim exceeds the effective maximum projection dimension"),
        )
        start=isnothing(v0) ? T[1+i/max(n, 1) for i = 1:n] : Vector{T}(v0)
        length(start)==n ||
            throw(DimensionMismatch("v0 length must equal operator column count"))
        all(isfinite, start) || throw(ArgumentError("v0 must be finite"))
        n>0 &&
            !(0<LinearAlgebra.norm(start)<Inf) &&
            throw(ArgumentError("v0 must have finite nonzero norm"))
        n>0 && (start ./= LinearAlgebra.norm(start))
        E=kind==0 ? Complex{_realtype(T)} : T
        vectors=zeros(E, m, k)
        values=zeros(E, k)
        right=zeros(T, n, kind==2 ? k : 0)
        algorithm=:partial
        count=0
        if k>0 && kind in (1, 2) && max(Int(max_dim), 64, 2k)>=min(m, n) && max_restarts>0
            # Upstream panics in this branch. Materialization is explicit and
            # limited to full projection requests; iterative calls stay native.
            dense=FaerMatrix(Matrix(op))
            if kind==1
                F=LinearAlgebra.eigen(LinearAlgebra.Hermitian(dense))
                order=sortperm(abs.(F.values); rev = true)[1:k]
                vectors .= F.vectors[:, order]
                values .= F.values[order]
            else
                F=LinearAlgebra.svd(dense)
                vectors .= F.U[:, 1:k]
                right .= F.V[:, 1:k]
                values .= F.S[1:k]
            end
            count=k
            algorithm=:dense_fallback
        elseif k>0 && max_restarts>0
            GC.@preserve start vectors values right begin
                count=Int(
                    _native_call(
                        RustCrate.operator_partial,
                        op,
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
                    ),
                )
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
                    LinearAlgebra.norm(op*right[:, j]-singular[j]*vectors[:, j]),
                    LinearAlgebra.norm(adjoint(op)*vectors[:, j]-singular[j]*right[:, j]),
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
            LinearAlgebra.norm(op*vectors[:, j]-values[j]*vectors[:, j]) for j = 1:count
        ]
        return (
            values = kind==1 ? real.(values) : values,
            vectors,
            nconverged = count,
            requested = k,
            residuals,
            algorithm,
        )
    end
end
partial_eigen(op::FaerLinearOperator, k::Integer; kwargs...) =
    _partial_spectrum(op, k, Int32(0); kwargs...)
partial_self_adjoint_eigen(op::FaerLinearOperator, k::Integer; kwargs...) =
    _partial_spectrum(op, k, Int32(1); kwargs...)
partial_svd(op::FaerLinearOperator, k::Integer; kwargs...) =
    _partial_spectrum(op, k, Int32(2); kwargs...)
