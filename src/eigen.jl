# t4a-faer returns complex eigenpairs for general real matrices and real pairs
# for self-adjoint matrices. The Julia result types follow those conventions.
_eigensort(x::Real) = x
_eigensort(x::Complex) = (real(x), imag(x))
function _eigen(
    a::FaerMatrix{Float64},
    mode;
    vectors = true,
    lower = true,
    b = nothing,
    other_lower = true,
    sortby = _eigensort,
)
    n=LinearAlgebra.checksquare(a)
    if !isnothing(b)
        size(b)==size(a) ||
            throw(DimensionMismatch("generalized eigen inputs must have the same shape"))
    end
    T=mode in (1, 3) ? Float64 : ComplexF64
    values=Vector{T}(undef, n)
    eigenvectors=Matrix{T}(undef, n, vectors ? n : 0)
    lock(_matrix_lock) do
        other = isnothing(b) ? a : b
        GC.@preserve values eigenvectors begin
            _native_call(
                RustCrate.eigen_buffers,
                a,
                Ptr{UInt8}(pointer(values)),
                Ptr{UInt8}(pointer(eigenvectors)),
                vectors,
                Int32(mode),
                lower,
                other,
                other_lower,
            )
        end
    end
    if !isnothing(sortby)
        order=sortperm(values; by = sortby)
        values=values[order]
        vectors && (eigenvectors=eigenvectors[:, order])
    end
    return vectors ? LinearAlgebra.Eigen(values, eigenvectors) : values
end

"""
    eigen(A::FaerMatrix{Float64}; sortby)
    eigen(A::Union{Hermitian,Symmetric}; sortby)

Compute eigenvalues and right eigenvectors using t4a-faer. Self-adjoint
wrappers read only the specified triangle. Results own independent Julia
arrays. General real matrices produce complex eigenpairs.
"""
LinearAlgebra.eigen(a::FaerMatrix{Float64}; sortby = _eigensort) = _eigen(a, 0; sortby)
LinearAlgebra.eigvals(a::FaerMatrix{Float64}; sortby = _eigensort) =
    _eigen(a, 0; vectors = false, sortby)
LinearAlgebra.eigen(a::_FaerSelfAdjoint; sortby = _eigensort) =
    _eigen(parent(a), 1; lower = _uplo(a)===:L, sortby)
LinearAlgebra.eigvals(a::_FaerSelfAdjoint; sortby = _eigensort) =
    _eigen(parent(a), 1; vectors = false, lower = _uplo(a)===:L, sortby)

"""Compute right generalized eigenvectors satisfying `A*V ≈ B*V*Diagonal(values)`."""
function LinearAlgebra.eigen(
    a::FaerMatrix{Float64},
    b::FaerMatrix{Float64};
    sortby = _eigensort,
)
    result=_eigen(a, 2; b, sortby)
    return LinearAlgebra.GeneralizedEigen(result.values, result.vectors)
end
LinearAlgebra.eigvals(a::FaerMatrix{Float64}, b::FaerMatrix{Float64}; sortby = _eigensort) =
    _eigen(a, 2; b, vectors = false, sortby)
function LinearAlgebra.eigen(a::_FaerSelfAdjoint, b::_FaerSelfAdjoint; sortby = _eigensort)
    result=_eigen(
        parent(a),
        3;
        b = parent(b),
        lower = _uplo(a)===:L,
        other_lower = _uplo(b)===:L,
        sortby,
    )
    return LinearAlgebra.GeneralizedEigen(result.values, result.vectors)
end
LinearAlgebra.eigvals(a::_FaerSelfAdjoint, b::_FaerSelfAdjoint; sortby = _eigensort) =
    _eigen(
        parent(a),
        3;
        b = parent(b),
        lower = _uplo(a)===:L,
        other_lower = _uplo(b)===:L,
        vectors = false,
        sortby,
    )

"""
    pfaffian(A::FaerMatrix{Float64})

Compute the Pfaffian of an even-dimensional skew-symmetric matrix with
 t4a-faer. Non-square, odd-dimensional and invalid inputs raise an exception.
"""
pfaffian(a::FaerMatrix{Float64}; scratch = nothing, team = nothing) =
    isnothing(scratch) && isnothing(team) ? _native_call(RustCrate.pfaffian_value, a) :
    _pfaffian_options(a, scratch, team)
