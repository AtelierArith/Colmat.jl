"""
    hessenberg(A::FaerMatrix; compact=false, scratch=nothing, team=nothing, par_threshold=FaerParThreshold(), blocking_threshold=65536)

A native Hessenberg factorization with explicit Rust-owned `H` and orthogonal
or unitary `Q`, satisfying `A ≈ Q*H*Q'`. Inputs are copied. Unlike Julia's compact
Householder storage, `F.Q` is an explicit matrix and `F.H` contains no reflector
workspace. Closing the factor closes its two owned matrices. Optional `scratch`
and `team` control declared kernel workspace and execution; owned outputs and
view copies allocate separately.
With `compact=true`, return `FaerCompactHessenberg` with implicit native
`FaerHessenbergQ` instead of materializing Q. The same keyword selects its
construction workspace in `scratch_requirements`, `FaerScratch` and `reserve!`.
`par_threshold` selects derived/fixed/never parallelism. A fixed floor counts
squared trailing order. `:never` keeps reduction and explicit Q formation
sequential, while still requiring any supplied team to be open.
`blocking_threshold` is a nonnegative integer: reduction is unblocked when
`n*n < blocking_threshold`, otherwise blocked. Zero always permits blocking;
`typemax(UInt)` selects unblocked reduction for representable dimensions.
Use the same options for workspace queries and computation. These options
control this reduction; saved-factor Schur iteration has its own defaults.
"""
struct FaerHessenberg{T,M<:FaerMatrix{T}} <: LinearAlgebra.Factorization{T}
    H::M
    Q::M
end
Base.size(f::FaerHessenberg) = size(f.H)
Base.size(f::FaerHessenberg, d::Integer) = size(f.H, d)
Base.isopen(f::FaerHessenberg) = isopen(f.H) && isopen(f.Q)
function Base.close(f::FaerHessenberg)
    lock(_matrix_lock) do ;
        close(f.H)
        return close(f.Q)
    end
    return nothing
end
function Base.copy(f::FaerHessenberg)
    return lock(() -> FaerHessenberg(copy(f.H), copy(f.Q)), _matrix_lock)
end
function Base.deepcopy_internal(f::FaerHessenberg, d::IdDict)
    get!(d, f) do
        lock(_matrix_lock) do
            return FaerHessenberg(
                Base.deepcopy_internal(f.H, d), Base.deepcopy_internal(f.Q, d)
            )
        end
    end
end
FaerMatrix(f::FaerHessenberg) = lock(() -> f.Q * f.H * f.Q', _matrix_lock)
Base.Matrix(f::FaerHessenberg) = Matrix(FaerMatrix(f))
Base.Array(f::FaerHessenberg) = Matrix(f)

"""
    schur(A::FaerMatrix; scratch=nothing, team=nothing)

Compute a real quasi-triangular or complex triangular Schur factorization in
faer, returning `LinearAlgebra.Schur` with Rust-owned `T` and `Z`. The input is
preserved and `A ≈ F.Z*F.T*F.Z'`. Eigenvalues are independent Julia vectors.
Optional `scratch` and `team` also apply to saved explicit or compact Hessenberg factors;
query with `scratch_requirements`, construct `FaerScratch`, or call `reserve!`.
"""
function LinearAlgebra.schur(a::FaerMatrix; scratch=nothing, team=nothing)
    return _native_schur(a; scratch, team)
end
function LinearAlgebra.schur(a::FaerMatrixView; scratch=nothing, team=nothing)
    return _native_schur(a; scratch, team)
end
function LinearAlgebra.hessenberg(
    a::Union{FaerMatrix,FaerMatrixView};
    scratch=nothing,
    team=nothing,
    compact::Bool=false,
    par_threshold=FaerParThreshold(),
    blocking_threshold::Integer=256 * 256,
)
    return _tuned_hessenberg(a; scratch, team, compact, par_threshold, blocking_threshold)
end
function LinearAlgebra.schur(f::FaerHessenberg; scratch=nothing, team=nothing)
    return _saved_schur(f; scratch, team)
end
"""
    hessenberg!(A::FaerMatrix; compact=false, scratch=nothing, team=nothing)

Compute a native Hessenberg reduction, then overwrite A with explicit H.
Return `FaerHessenberg(A,Q)`. The result shares H storage with A; existing views
remain valid. Independent native work results allocate separately from declared
caller scratch.
With `compact=true`, return `FaerCompactHessenberg(A,Q)` whose implicit Q owns
packed reflectors independently of A.
"""
function LinearAlgebra.hessenberg!(
    a::FaerMatrix;
    scratch=nothing,
    team=nothing,
    compact::Bool=false,
    par_threshold=FaerParThreshold(),
    blocking_threshold::Integer=256 * 256,
)
    lock(_matrix_lock) do
        f = LinearAlgebra.hessenberg(
            a; scratch, team, compact, par_threshold, blocking_threshold
        )
        try
            copyto!(a, f.H)
            return compact ? FaerCompactHessenberg(a, f.Q) : FaerHessenberg(a, f.Q)
        finally
            compact && close(f.H)
        end
    end
end
"""
    schur!(H::FaerHessenberg; scratch=nothing, team=nothing)

Reuse the Hessenberg reduction and update both H.H and H.Q after successful
Schur QR iteration. The returned Schur factor shares these owned matrices.
"""
function LinearAlgebra.schur!(f::FaerHessenberg; scratch=nothing, team=nothing)
    lock(_matrix_lock) do
        f.H === f.Q && throw(ArgumentError("H and Q must have distinct storage"))
        result = LinearAlgebra.schur(f; scratch, team)
        copyto!(f.H, result.T)
        copyto!(f.Q, result.Z)
        return LinearAlgebra.Schur(f.H, f.Q, result.values)
    end
end
const _NativeSchur = LinearAlgebra.Schur{T,<:FaerMatrix{T}} where {T<:_NativeScalar}
function _check_schur(f::_NativeSchur)
    _handle(f.T)
    _handle(f.Z)
    n = LinearAlgebra.checksquare(f.T)
    size(f.Z) == (n, n) && length(f.values) == n ||
        throw(DimensionMismatch("Schur T, Z and values must have matching dimensions"))
    return n
end
Base.size(f::_NativeSchur) = size(f.T)
Base.size(f::_NativeSchur, d::Integer) = size(f.T, d)
Base.isopen(f::_NativeSchur) = isopen(f.T) && isopen(f.Z)
function Base.close(f::_NativeSchur)
    lock(_matrix_lock) do ;
        close(f.T)
        return close(f.Z)
    end
    return nothing
end
function Base.copy(f::_NativeSchur)
    lock(_matrix_lock) do
        _check_schur(f)
        return LinearAlgebra.Schur(copy(f.T), copy(f.Z), copy(f.values))
    end
end
function Base.deepcopy_internal(f::_NativeSchur, d::IdDict)
    get!(d, f) do
        lock(_matrix_lock) do
            _check_schur(f)
            return LinearAlgebra.Schur(
                Base.deepcopy_internal(f.T, d),
                Base.deepcopy_internal(f.Z, d),
                Base.deepcopy_internal(f.values, d),
            )
        end
    end
end
FaerMatrix(f::_NativeSchur) = lock(() -> f.Z * f.T * f.Z', _matrix_lock)
Base.Matrix(f::_NativeSchur) = Matrix(FaerMatrix(f))
function _schur_selection(f::_NativeSchur, select)
    n = _check_schur(f)
    length(select) == n ||
        throw(DimensionMismatch("Schur selection must match the matrix dimension"))
    Base.require_one_based_indexing(select)
    data = UInt8.(select)
    if eltype(f) <: Real
        i = 1
        while i < n
            if !iszero(f.T[i + 1, i])
                data[i] == data[i + 1] || throw(
                    ArgumentError("select both or neither member of a real conjugate pair"),
                )
                i += 2
            else
                i += 1
            end
        end
    end
    return data
end
"""
    ordschur(F::Schur{T,<:FaerMatrix}, select; scratch=nothing, team=nothing)

Move selected Schur blocks to the leading positions, returning independent native
factors. Select both members of each real conjugate pair. Optional scratch covers
native flags and block swaps; working factors and Julia arrays allocate separately.
A supplied team lends its token, but selected faer's reorder kernels are sequential.
The bang form preserves existing owners and views and commits only after successful
reordering and eigenvalue conversion.
"""
function LinearAlgebra.ordschur(
    f::_NativeSchur, select::Union{Vector{Bool},BitVector}; scratch=nothing, team=nothing
)
    return _ordschur(f, select; scratch, team)
end
function LinearAlgebra.ordschur!(
    f::_NativeSchur, select::Union{Vector{Bool},BitVector}; scratch=nothing, team=nothing
)
    lock(_matrix_lock) do
        f.T === f.Z && throw(ArgumentError("Schur T and Z must have distinct storage"))
        f.values isa StridedVector || throw(
            ArgumentError(
                "mutating native Schur requires mutable strided eigenvalue storage"
            ),
        )
        result = _ordschur(f, select; scratch, team)
        try
            values = Vector{eltype(f.values)}(result.values)
            copyto!(f.T, result.T)
            copyto!(f.Z, result.Z)
            copyto!(f.values, values)
        finally
            close(result)
        end
    end
    return f
end
"""
    schur!(A::FaerMatrix; scratch=nothing, team=nothing)

Compute a native Schur factorization and overwrite A with its Schur form.
The result's T is A itself. Independent work results are allocated, and A is
updated after convergence; its shape and existing views are preserved.
"""
function LinearAlgebra.schur!(a::FaerMatrix; scratch=nothing, team=nothing)
    lock(_matrix_lock) do
        f = LinearAlgebra.schur(a; scratch, team)
        copyto!(a, f.T)
        return LinearAlgebra.Schur(a, f.Z, f.values)
    end
end
function LinearAlgebra.eigvals(f::_NativeSchur)
    lock(_matrix_lock) do
        _check_schur(f)
        return copy(f.values)
    end
end
"""
    eigvecs(F::Schur{T,<:FaerMatrix}; scratch=nothing, team=nothing)
    eigen(F::Schur{T,<:FaerMatrix}; scratch=nothing, team=nothing)

Compute native unit-norm right eigenvectors in stored Schur diagonal order.
Real conjugate columns are returned as complex vectors. Caller scratch covers
real packing and faer's declared kernel temporary storage; owned output and
Julia eigenvalue copies allocate separately. The selected faer may still allocate
an internal scaled Schur copy for extreme entry magnitudes.
"""
function LinearAlgebra.eigvecs(f::_NativeSchur; scratch=nothing, team=nothing)
    return _saved_schur_eigenvectors(f; scratch, team)
end
function LinearAlgebra.eigen(f::_NativeSchur; scratch=nothing, team=nothing)
    lock(_matrix_lock) do
        vectors = _saved_schur_eigenvectors(f; scratch, team)
        try
            LinearAlgebra.Eigen(LinearAlgebra.eigvals(f), vectors)
        catch
            close(vectors)
            rethrow()
        end
    end
end

Base.iterate(f::FaerHessenberg) = (f.H, Val(:Q))
Base.iterate(f::FaerHessenberg, ::Val{:Q}) = (f.Q, Val(:done))
Base.iterate(f::FaerHessenberg, ::Val{:done}) = nothing
