"""
    scratch_requirements(factorize, plan::FaerSparseSymbolic, A::FaerSparseMatrix)

Query numeric factor-construction workspace for auto/supernodal QR. Pass a
reserved buffer to `factorize(plan,A; scratch=s)`. The numeric QR kernel and
source-precision rank-certificate preparation share this buffer sequentially.
Symbolic analysis, input normalization/snapshot/finite checks and owned numeric
factor storage allocate separately. The stored pattern must match.
"""
function scratch_requirements(
    ::typeof(LinearAlgebra.factorize),
    p::FaerSparseSymbolic{:qr},
    a::FaerSparseMatrix{T},
) where {T}
    lock(_matrix_lock) do
        _sparse_factor_inputs(p, a)
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
