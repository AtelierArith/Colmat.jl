"""
    balance(A) -> FaerBalance

Native permutation and power-of-two balancing of a square matrix. The result
owns `B`, 1-based `perm`, positive real `scale`, and a Julia `active` range.
`B[i,j] ≈ A[perm[i],perm[j]] * scale[j] / scale[i]`. The input is unchanged.
For right eigenvectors use [`balance_eigenvectors`](@ref) to undo balancing.
Metadata arrays are independent copies and validated before back-transforms.
"""
struct FaerBalance{T,M<:FaerMatrix{T},R}
    B::M
    perm::Vector{Int}
    scale::Vector{R}
    active::UnitRange{Int}
end
Base.eltype(::Type{<:FaerBalance{T}}) where {T} = T
Base.eltype(f::FaerBalance) = eltype(f.B)
Base.size(f::FaerBalance) = size(f.B)
Base.size(f::FaerBalance, d::Integer) = size(f.B, d)
Base.isopen(f::FaerBalance) = isopen(f.B)
Base.close(f::FaerBalance) = close(f.B)
function _balance_check(f::FaerBalance)
    _handle(f.B)
    n = size(f.B, 1)
    length(f.perm) == length(f.scale) == n ||
        throw(DimensionMismatch("balance metadata lengths must match B"))
    sort(f.perm) == collect(1:n) ||
        throw(ArgumentError("balance indices must be a permutation of 1:n"))
    all(x -> isfinite(x) && x > 0, f.scale) ||
        throw(ArgumentError("balance scales must be finite and positive"))
    return nothing
end
function Base.copy(f::FaerBalance)
    lock(_matrix_lock) do
        _balance_check(f)
        return FaerBalance(copy(f.B), copy(f.perm), copy(f.scale), f.active)
    end
end
Base.deepcopy_internal(f::FaerBalance, d::IdDict) = get!(() -> copy(f), d, f)
function balance(a::AbstractMatrix{<:Number})
    lock(_matrix_lock) do
        a isa _RotationArray && _rotation_open(a)
        size(a, 1) == size(a, 2) ||
            throw(DimensionMismatch("balancing requires a square matrix"))
        T = _reflector_type(eltype(a))
        R = _realtype(T)
        data = Matrix{T}(a)
        all(isfinite, data) || throw(ArgumentError("balance input must be finite"))
        n = size(a, 1)
        perm = zeros(UInt, n)
        scale = zeros(R, n)
        active = zeros(UInt, 2)
        GC.@preserve data perm scale active _native_call(
            RustCrate.balance_data,
            Ptr{UInt8}(pointer(data)),
            UInt(n),
            perm,
            Ptr{UInt8}(pointer(scale)),
            active,
            _tag(T),
        )
        return FaerBalance(
            FaerMatrix(data), Int.(perm) .+ 1, scale, (Int(active[1]) + 1):Int(active[2])
        )
    end
end
"""
    balance!(A::FaerMatrix)

Compute balancing, then replace A after success. The returned factor's B shares
A's native storage. Closing either closes that shared owner. Computation stages
an independent matrix; invalid inputs leave A unchanged.
"""
function balance!(a::FaerMatrix)
    lock(_matrix_lock) do
        result = balance(a)
        copyto!(a, result.B)
        return FaerBalance(a, result.perm, result.scale, result.active)
    end
end
"""
    balance_eigenvectors(F::FaerBalance, X)

Back-transform right eigenvector columns by `Y[F.perm[i],j] =
F.scale[i] * X[i,j]` using faer. Columns are not renormalized. Inputs and factor
are unchanged. Real balancing metadata and complex vectors promote normally.
Native vector inputs return native vectors, Julia vectors return Julia vectors,
and matrix inputs return native matrices. Values must remain finite.
"""
function balance_eigenvectors(f::FaerBalance, x::AbstractVecOrMat{<:Number})
    lock(_matrix_lock) do
        _balance_check(f)
        x isa _RotationArray && _rotation_open(x)
        size(x, 1) == size(f, 1) ||
            throw(DimensionMismatch("eigenvector rows must match balanced matrix"))
        T = promote_type(eltype(f.B), _reflector_type(eltype(x)))
        R = _realtype(T)
        data = x isa AbstractVector ? reshape(Vector{T}(x), length(x), 1) : Matrix{T}(x)
        all(isfinite, data) || throw(ArgumentError("eigenvector inputs must be finite"))
        perm = UInt.(f.perm .- 1)
        scale = Vector{R}(f.scale)
        all(x -> isfinite(x) && x > 0, scale) || throw(
            ArgumentError("scales must remain finite and positive in the working type")
        )
        GC.@preserve data perm scale _native_call(
            RustCrate.balance_vectors_data,
            Ptr{UInt8}(pointer(data)),
            UInt(size(data, 1)),
            UInt(size(data, 2)),
            perm,
            Ptr{UInt8}(pointer(scale)),
            _tag(T),
        )
        all(isfinite, data) || throw(
            ArgumentError("back-transformed eigenvectors overflowed the working type")
        )
        if x isa _BorrowVector
            return _vector_from_source(FaerMatrix(data))
        elseif x isa AbstractVector
            return vec(data)
        end
        return FaerMatrix(data)
    end
end
"""Back-transform X after successful staging; conversion or metadata failure leaves X unchanged."""
function balance_eigenvectors!(f::FaerBalance, x::_HouseholderArray)
    lock(_matrix_lock) do
        return _operator_copy_result!(x, balance_eigenvectors(f, x))
    end
end
