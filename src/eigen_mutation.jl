# Inside `with_evd_tuning`, the Schur form written back follows the scope's
# Hessenberg settings, like the driver's own reduction; the multishift QR step
# follows `with_schur_tuning` in either case.
function _writeback_schur(a)
    iszero(_evd_tuning_depth[]) && return LinearAlgebra.schur(a)
    values = zeros(UInt, 5)
    flags = zeros(UInt8, 5)
    _native_call(RustCrate.evd_tuning_state, values, flags)
    par = if iszero(flags[1]) || iszero(values[1])
        FaerParThreshold()
    elseif values[1] == typemax(UInt)
        FaerParThreshold(:never)
    else
        FaerParThreshold(values[1])
    end
    blocking = iszero(flags[2]) ? 256 * 256 : values[2]
    h = LinearAlgebra.hessenberg(a; par_threshold=par, blocking_threshold=blocking)
    try
        return LinearAlgebra.schur(h)
    finally
        close(h)
    end
end
function _general_eigen_writeback!(
    a::FaerMatrix, vectors::Bool, sortby; scratch=nothing, team=nothing
)
    lock(_matrix_lock) do
        _handle(a)
        LinearAlgebra.checksquare(a)
        if !_general_evd_default_path(scratch, team)
            result = if vectors
                LinearAlgebra.eigen(a; sortby, scratch, team)
            else
                LinearAlgebra.eigvals(a; sortby, scratch, team)
            end
            form = _writeback_schur(a)
            try
                copyto!(a, form.T)
                return result
            finally
                close(form)
            end
        end
        schur = LinearAlgebra.schur(a)
        try
            values = copy(schur.values)
            eigenvectors = nothing
            if vectors
                native = LinearAlgebra.eigvecs(schur)
                try
                    eigenvectors = Matrix(native)
                finally
                    close(native)
                end
                for j in axes(eigenvectors, 2)
                    column = view(eigenvectors, :, j)
                    length = LinearAlgebra.norm(column)
                    isfinite(length) && length > 0 ||
                        throw(ArgumentError("native eigenvector has invalid norm"))
                    column ./= length
                end
            end
            if !isnothing(sortby)
                order = sortperm(values; by=sortby)
                values = values[order]
                vectors && (eigenvectors = eigenvectors[:, order])
            end
            result = vectors ? LinearAlgebra.Eigen(values, eigenvectors) : values
            copyto!(a, schur.T)
            return result
        finally
            close(schur)
        end
    end
end
"""
    eigen!(A::FaerMatrix; sortby)
    eigvals!(A::FaerMatrix; sortby)

Compute native eigenpairs/values, then overwrite A by its real quasi-triangular
or complex triangular Schur form. General eigenpairs use complex Julia arrays;
right eigenvectors are normalized. Returned outputs remain independent of A.
With `scratch`/`team`, or inside [`with_evd_tuning`](@ref) or
[`with_schur_tuning`](@ref), the outputs come from the same workspace driver as
`eigen(A; ...)` and the written Schur form is computed separately (sequential,
internal workspace) with the scope's Hessenberg and multishift QR settings;
`evd_recursion_threshold` only affects the returned eigenvectors. Self-adjoint
`eigen!`/`eigvals!` return what `eigen`/`eigvals` of the wrapper return in the
same scope, so `with_evd_tuning` applies to them as well.
Sort/convergence errors before writeback preserve A. Factorization and writeback
allocate; this is not a zero-allocation LAPACK work-buffer interface.

For native Hermitian/real Symmetric wrappers, eigen! writes eigenvectors to the
parent and eigvals! writes the diagonal eigenvalue matrix. Only the selected
triangle is read. Generalized pairs use QZ S/T writeback, or eigenvectors and a
diagonal eigenvalue matrix for self-adjoint pairs. Pair scalar types must match
and their parents must have distinct storage.
"""
function LinearAlgebra.eigen!(
    a::FaerMatrix; sortby=_eigensort, scratch=nothing, team=nothing
)
    return _general_eigen_writeback!(a, true, sortby; scratch, team)
end
function LinearAlgebra.eigvals!(
    a::FaerMatrix; sortby=_eigensort, scratch=nothing, team=nothing
)
    return _general_eigen_writeback!(a, false, sortby; scratch, team)
end
function _selfadjoint_eigen_writeback!(
    a::_AnyFaerSelfAdjoint,
    vectors::Bool,
    sortby;
    scratch=nothing,
    team=nothing,
    alg=nothing,
)
    lock(_matrix_lock) do
        source = parent(a)
        _handle(source)
        result = if vectors
            LinearAlgebra.eigen(a; sortby, scratch, team, alg)
        else
            LinearAlgebra.eigvals(a; sortby, scratch, team, alg)
        end
        packed = if vectors
            FaerMatrix(result.vectors)
        else
            FaerMatrix(Matrix{eltype(source)}(LinearAlgebra.Diagonal(result)))
        end
        copyto!(source, packed)
        return result
    end
end
function LinearAlgebra.eigen!(
    a::_AnyFaerSelfAdjoint; sortby=_eigensort, scratch=nothing, team=nothing, alg=nothing
)
    return _selfadjoint_eigen_writeback!(a, true, sortby; scratch, team, alg)
end
function LinearAlgebra.eigvals!(
    a::_AnyFaerSelfAdjoint; sortby=_eigensort, scratch=nothing, team=nothing, alg=nothing
)
    return _selfadjoint_eigen_writeback!(a, false, sortby; scratch, team, alg)
end
function _eigen_pair_check(a, b)
    a === b && throw(ArgumentError("generalized eigen inputs must have distinct storage"))
    eltype(a) === eltype(b) || throw(
        ArgumentError("destructive generalized eigen inputs require matching scalar types"),
    )
    size(a) == size(b) ||
        throw(DimensionMismatch("generalized eigen input sizes must match"))
    LinearAlgebra.checksquare(a)
    _handle(a)
    _handle(b)
    return nothing
end
function _generalized_eigen_writeback!(
    a::FaerMatrix, b::FaerMatrix, vectors::Bool, sortby; scratch=nothing, team=nothing
)
    lock(_matrix_lock) do
        _eigen_pair_check(a, b)
        result = if vectors
            LinearAlgebra.eigen(a, b; sortby, scratch, team)
        else
            LinearAlgebra.eigvals(a, b; sortby, scratch, team)
        end
        qz = LinearAlgebra.schur(a, b)
        try
            copyto!(a, qz.S)
            copyto!(b, qz.T)
            return result
        finally
            close(qz)
        end
    end
end
function LinearAlgebra.eigen!(
    a::FaerMatrix, b::FaerMatrix; sortby=_eigensort, scratch=nothing, team=nothing
)
    return _generalized_eigen_writeback!(a, b, true, sortby; scratch, team)
end
function LinearAlgebra.eigvals!(
    a::FaerMatrix, b::FaerMatrix; sortby=_eigensort, scratch=nothing, team=nothing
)
    return _generalized_eigen_writeback!(a, b, false, sortby; scratch, team)
end
function _selfadjoint_pair_writeback!(
    a::_AnyFaerSelfAdjoint,
    b::_AnyFaerSelfAdjoint,
    vectors::Bool,
    sortby;
    scratch=nothing,
    team=nothing,
)
    lock(_matrix_lock) do
        x, y = parent(a), parent(b)
        _eigen_pair_check(x, y)
        result = if vectors
            LinearAlgebra.eigen(a, b; sortby, scratch, team)
        else
            LinearAlgebra.eigvals(a, b; sortby, scratch, team)
        end
        values = vectors ? result.values : result
        diagonal = FaerMatrix(Matrix{eltype(y)}(LinearAlgebra.Diagonal(values)))
        left = vectors ? FaerMatrix(result.vectors) : copy(diagonal)
        copyto!(x, left)
        copyto!(y, diagonal)
        return result
    end
end
function LinearAlgebra.eigen!(
    a::_AnyFaerSelfAdjoint,
    b::_AnyFaerSelfAdjoint;
    sortby=_eigensort,
    scratch=nothing,
    team=nothing,
)
    return _selfadjoint_pair_writeback!(a, b, true, sortby; scratch, team)
end
function LinearAlgebra.eigvals!(
    a::_AnyFaerSelfAdjoint,
    b::_AnyFaerSelfAdjoint;
    sortby=_eigensort,
    scratch=nothing,
    team=nothing,
)
    return _selfadjoint_pair_writeback!(a, b, false, sortby; scratch, team)
end
