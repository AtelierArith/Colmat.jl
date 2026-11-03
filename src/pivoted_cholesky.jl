const _PivotedNativeCholesky=LinearAlgebra.CholeskyPivoted{T,<:FaerMatrix} where {T}
"""
    cholesky(A::FaerMatrix, RowMaximum(); tol=0.0, check=true)

Pivoted Cholesky with independently owned native factors. Hermitian/Symmetric
wrappers select the stored triangle. Negative tol uses faer's fixed automatic
threshold eps(T)*n*maximum(diag(A)); nonnegative tol uses an absolute cutoff in
the Rust adapter's complete diagonal-pivot algorithm. Rank-deficient factors
require check=false; solves require full rank. F.L, F.U, F.p, F.rank and Matrix(F)
follow LinearAlgebra's CholeskyPivoted conventions.
"""
function _pivoted_cholesky(a; tol::Real = 0.0, check::Bool = true)
    lock(_matrix_lock) do
        source=a isa FaerMatrix ? a : parent(a)
        _handle(source)
        size(a, 1)==size(a, 2) ||
            throw(DimensionMismatch("pivoted Cholesky requires a square matrix"))
        isfinite(tol) || throw(ArgumentError("tol must be finite"))
        a isa FaerMatrix &&
            !LinearAlgebra.ishermitian(a) &&
            throw(ArgumentError("matrix must be Hermitian"))
        T=eltype(a)
        a isa LinearAlgebra.Symmetric &&
            T<:Complex &&
            throw(ArgumentError("complex symmetric input must be Hermitian"))
        x=Matrix{T}(a)
        n=size(x, 1)
        out=zeros(T, n, n)
        p=zeros(UInt, n)
        rank=GC.@preserve x out p Int(
            _native_call(
                RustCrate.pivoted_llt,
                Ptr{UInt8}(pointer(x)),
                UInt(n),
                _tag(T),
                Ptr{UInt8}(pointer(out)),
                pointer(p),
                Float64(tol),
            ),
        )
        F=LinearAlgebra.CholeskyPivoted(
            FaerMatrix(out),
            'L',
            Int.(p) .+ 1,
            rank,
            tol,
            rank==n ? 0 : 1,
        )
        check && rank<n && throw(LinearAlgebra.RankDeficientException(rank))
        return F
    end
end
LinearAlgebra.cholesky(a::FaerMatrix, ::LinearAlgebra.RowMaximum; kwargs...) =
    _pivoted_cholesky(a; kwargs...)
for W in (:Hermitian, :Symmetric)
    @eval LinearAlgebra.cholesky(
        a::LinearAlgebra.$W{T,<:FaerMatrix},
        ::LinearAlgebra.RowMaximum;
        kwargs...,
    ) where {T} = _pivoted_cholesky(a; kwargs...)
end
function _pivoted_apply(F::_PivotedNativeCholesky, b, op::Int32)
    lock(_matrix_lock) do
        _handle(F.factors)
        n=size(F, 1)
        op!=2 && F.rank!=n && throw(LinearAlgebra.RankDeficientException(Int(F.rank)))
        size(b, 1)==n || throw(DimensionMismatch("pivoted Cholesky RHS size must match"))
        P=promote_type(eltype(F), eltype(b))
        l=F.uplo=='L' ? Matrix{P}(F.factors) : Matrix{P}(adjoint(F.factors))
        rhs=b isa AbstractVector ? reshape(Vector{P}(b), :, 1) : Matrix{P}(b)
        k=op==0 ? size(rhs, 2) : n
        result=zeros(P, n, k)
        p=UInt.(F.p .- 1)
        length(p)==n ||
            throw(DimensionMismatch("pivot permutation length must match factors"))
        GC.@preserve l rhs result p _native_call(
            RustCrate.pivoted_llt_apply,
            Ptr{UInt8}(pointer(l)),
            pointer(p),
            UInt(n),
            _tag(P),
            Ptr{UInt8}(pointer(rhs)),
            UInt(k),
            Ptr{UInt8}(pointer(result)),
            op,
        )
        return b isa AbstractVector ? vec(result) : result
    end
end
_pivoted_solve(F::_PivotedNativeCholesky, b) = _pivoted_apply(F, b, Int32(0))
LinearAlgebra.ldiv!(F::_PivotedNativeCholesky, b::AbstractVector) =
    copyto!(b, _pivoted_solve(F, b))
LinearAlgebra.ldiv!(F::_PivotedNativeCholesky, b::AbstractMatrix) =
    copyto!(b, _pivoted_solve(F, b))
Base.inv(F::_PivotedNativeCholesky) =
    FaerMatrix(_pivoted_apply(F, zeros(eltype(F), size(F, 1), 0), Int32(1)))
Base.Matrix(F::_PivotedNativeCholesky) =
    _pivoted_apply(F, zeros(eltype(F), size(F, 1), 0), Int32(2))
FaerMatrix(F::_PivotedNativeCholesky) = FaerMatrix(Matrix(F))
Base.close(F::_PivotedNativeCholesky) = close(F.factors)
Base.isopen(F::_PivotedNativeCholesky) = isopen(F.factors)
