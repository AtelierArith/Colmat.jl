"""
    scratch_requirements(ldiv!, F::FaerSparseFactorization, B::AbstractMatrix)

Query native solve workspace for retained auto/supernodal sparse QR. Use
`ldiv!(X,F,B; scratch=FaerScratch(ldiv!,F,B))` to reuse it. Factorization,
RHS/result storage and destination copying allocate outside scratch. RHS must
have the factor's scalar type; transposed and explicit simplicial QR are not
supported by this workspace interface. Numerically rank-deficient factors are
rejected; inconclusive sparse rank bounds require an allocating SVD validation.
The buffer never grows implicitly.
"""
function scratch_requirements(
    ::typeof(LinearAlgebra.ldiv!),
    f::FaerSparseFactorization{T,:qr},
    b::AbstractMatrix,
) where {T}
    lock(_matrix_lock) do
        _handle(f)
        size(b, 1)==size(f, 1) || throw(DimensionMismatch("factor and RHS rows must match"))
        eltype(b)===T ||
            throw(ArgumentError("workspace solve requires matching RHS scalar type"))
        if _native_call(RustCrate.sparse_qr_needs_spectral_solve, f) &&
           LinearAlgebra.rank(f)<min(size(f)...)
            throw(
                ArgumentError(
                    "rank-deficient QR solve requires SVD workspace; omit scratch",
                ),
            )
        end
        info=zeros(UInt, 2)
        GC.@preserve info _native_call(
            RustCrate.sparse_qr_solve_requirements,
            f,
            UInt(size(b, 2)),
            pointer(info),
        )
        return (bytes = Int(info[1]), alignment = Int(info[2]))
    end
end
function FaerScratch(
    fun::typeof(LinearAlgebra.ldiv!),
    f::FaerSparseFactorization,
    b::AbstractMatrix,
)
    req=scratch_requirements(fun, f, b)
    FaerScratch(req.bytes; alignment = req.alignment)
end
function reserve!(
    s::FaerScratch,
    fun::typeof(LinearAlgebra.ldiv!),
    f::FaerSparseFactorization,
    b::AbstractMatrix,
)
    req=scratch_requirements(fun, f, b)
    reserve!(s, req.bytes; alignment = req.alignment)
end
function _sparse_qr_workspace_solve(
    f::FaerSparseFactorization{T,:qr},
    b::AbstractMatrix,
    s::FaerScratch,
) where {T}
    lock(_matrix_lock) do
        req=scratch_requirements(LinearAlgebra.ldiv!, f, b)
        cap=scratch_capacity(s)
        cap.bytes>=req.bytes && cap.alignment>=req.alignment ||
            throw(ArgumentError("insufficient scratch; call reserve!"))
        rhs=b isa FaerMatrix ? b : FaerMatrix{T}(b)
        dims=(size(f, 2), size(b, 2))
        T===Float64 ? _matrix_result(RustCrate.sparse_qr_solve_workspace, dims, f, rhs, s) :
        _typed_result(T, RustCrate.sparse_qr_solve_workspace, dims, f, rhs, s)
    end
end
_sparse_qr_workspace_solve(f::_SparseSolveFactor, b::AbstractMatrix, scratch) = throw(
    ArgumentError(
        "scratch requires FaerScratch and a normal auto/supernodal sparse QR factor",
    ),
)

"""
    scratch_requirements(factorize, plan::FaerSparseSymbolic, A::FaerSparseMatrix)

Query numeric factor-construction workspace for auto/supernodal QR. Pass a
reserved buffer to `factorize(plan,A; scratch=s)`. Symbolic analysis and owned
numeric factor storage allocate outside scratch. The stored pattern must match.
"""
function scratch_requirements(
    ::typeof(LinearAlgebra.factorize),
    p::FaerSparseSymbolic{:qr},
    a::FaerSparseMatrix{T},
) where {T}
    lock(_matrix_lock) do
        _handle(p)
        _handle(a)
        size(p)==size(a) || throw(DimensionMismatch("plan and matrix shapes must match"))
        info=zeros(UInt, 2)
        GC.@preserve info _native_call(
            RustCrate.sparse_qr_factor_requirements,
            p,
            _tag(T),
            pointer(info),
        )
        (bytes = Int(info[1]), alignment = Int(info[2]))
    end
end
function FaerScratch(
    fun::typeof(LinearAlgebra.factorize),
    p::FaerSparseSymbolic,
    a::FaerSparseMatrix,
)
    req=scratch_requirements(fun, p, a)
    FaerScratch(req.bytes; alignment = req.alignment)
end
function reserve!(
    s::FaerScratch,
    fun::typeof(LinearAlgebra.factorize),
    p::FaerSparseSymbolic,
    a::FaerSparseMatrix,
)
    req=scratch_requirements(fun, p, a)
    reserve!(s, req.bytes; alignment = req.alignment)
end
function _sparse_qr_factor_workspace(h, p, a, s::FaerScratch)
    req=scratch_requirements(LinearAlgebra.factorize, p, a)
    cap=scratch_capacity(s)
    cap.bytes>=req.bytes && cap.alignment>=req.alignment ||
        throw(ArgumentError("insufficient factorization scratch; call reserve!"))
    Int(_native_call(RustCrate.sparse_qr_factor_workspace, h, p, a, s))
end
