function _general_eigen_writeback!(
    a::FaerMatrix,
    vectors::Bool,
    sortby;
    scratch = nothing,
    team = nothing,
)
    lock(_matrix_lock) do
        _handle(a)
        LinearAlgebra.checksquare(a)
        if !isnothing(scratch) || !isnothing(team)
            result=vectors ? LinearAlgebra.eigen(a; sortby, scratch, team) :
                   LinearAlgebra.eigvals(a; sortby, scratch, team)
            form=LinearAlgebra.schur(a)
            try
                copyto!(a, form.T)
                return result
            finally
                close(form)
            end
        end
        schur=LinearAlgebra.schur(a)
        try
            values=copy(schur.values)
            eigenvectors=nothing
            if vectors
                native=LinearAlgebra.eigvecs(schur)
                try
                    eigenvectors=Matrix(native)
                finally
                    close(native)
                end
                for j in axes(eigenvectors, 2)
                    column=view(eigenvectors, :, j)
                    length=LinearAlgebra.norm(column)
                    isfinite(length) && length>0 ||
                        throw(ArgumentError("native eigenvector has invalid norm"))
                    column ./= length
                end
            end
            if !isnothing(sortby)
                order=sortperm(values; by = sortby)
                values=values[order]
                vectors && (eigenvectors=eigenvectors[:, order])
            end
            result=vectors ? LinearAlgebra.Eigen(values, eigenvectors) : values
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
Sort/convergence errors before writeback preserve A. Factorization and writeback
allocate; this is not a zero-allocation LAPACK work-buffer interface.

For native Hermitian/real Symmetric wrappers, eigen! writes eigenvectors to the
parent and eigvals! writes the diagonal eigenvalue matrix. Only the selected
triangle is read. Generalized pairs use QZ S/T writeback, or eigenvectors and a
diagonal eigenvalue matrix for self-adjoint pairs. Pair scalar types must match
and their parents must have distinct storage.
"""
LinearAlgebra.eigen!(
    a::FaerMatrix;
    sortby = _eigensort,
    scratch = nothing,
    team = nothing,
) = _general_eigen_writeback!(a, true, sortby; scratch, team)
LinearAlgebra.eigvals!(
    a::FaerMatrix;
    sortby = _eigensort,
    scratch = nothing,
    team = nothing,
) = _general_eigen_writeback!(a, false, sortby; scratch, team)
function _selfadjoint_eigen_writeback!(
    a::_AnyFaerSelfAdjoint,
    vectors::Bool,
    sortby;
    scratch = nothing,
    team = nothing,
)
    lock(_matrix_lock) do
        source=parent(a)
        _handle(source)
        result=vectors ? LinearAlgebra.eigen(a; sortby, scratch, team) :
               LinearAlgebra.eigvals(a; sortby, scratch, team)
        packed=vectors ? FaerMatrix(result.vectors) :
               FaerMatrix(Matrix{eltype(source)}(LinearAlgebra.Diagonal(result)))
        copyto!(source, packed)
        return result
    end
end
LinearAlgebra.eigen!(
    a::_AnyFaerSelfAdjoint;
    sortby = _eigensort,
    scratch = nothing,
    team = nothing,
) = _selfadjoint_eigen_writeback!(a, true, sortby; scratch, team)
LinearAlgebra.eigvals!(
    a::_AnyFaerSelfAdjoint;
    sortby = _eigensort,
    scratch = nothing,
    team = nothing,
) = _selfadjoint_eigen_writeback!(a, false, sortby; scratch, team)
function _eigen_pair_check(a, b)
    a===b && throw(ArgumentError("generalized eigen inputs must have distinct storage"))
    eltype(a)===eltype(b) || throw(
        ArgumentError("destructive generalized eigen inputs require matching scalar types"),
    )
    size(a)==size(b) || throw(DimensionMismatch("generalized eigen input sizes must match"))
    LinearAlgebra.checksquare(a)
    _handle(a)
    _handle(b)
    return nothing
end
function _generalized_eigen_writeback!(a::FaerMatrix, b::FaerMatrix, vectors::Bool, sortby)
    lock(_matrix_lock) do
        _eigen_pair_check(a, b)
        result=vectors ? LinearAlgebra.eigen(a, b; sortby) :
               LinearAlgebra.eigvals(a, b; sortby)
        qz=LinearAlgebra.schur(a, b)
        try
            copyto!(a, qz.S)
            copyto!(b, qz.T)
            return result
        finally
            close(qz)
        end
    end
end
LinearAlgebra.eigen!(a::FaerMatrix, b::FaerMatrix; sortby = _eigensort) =
    _generalized_eigen_writeback!(a, b, true, sortby)
LinearAlgebra.eigvals!(a::FaerMatrix, b::FaerMatrix; sortby = _eigensort) =
    _generalized_eigen_writeback!(a, b, false, sortby)
function _selfadjoint_pair_writeback!(
    a::_AnyFaerSelfAdjoint,
    b::_AnyFaerSelfAdjoint,
    vectors::Bool,
    sortby,
)
    lock(_matrix_lock) do
        x, y=parent(a), parent(b)
        _eigen_pair_check(x, y)
        result=vectors ? LinearAlgebra.eigen(a, b; sortby) :
               LinearAlgebra.eigvals(a, b; sortby)
        values=vectors ? result.values : result
        diagonal=FaerMatrix(Matrix{eltype(y)}(LinearAlgebra.Diagonal(values)))
        left=vectors ? FaerMatrix(result.vectors) : copy(diagonal)
        copyto!(x, left)
        copyto!(y, diagonal)
        return result
    end
end
LinearAlgebra.eigen!(a::_AnyFaerSelfAdjoint, b::_AnyFaerSelfAdjoint; sortby = _eigensort) =
    _selfadjoint_pair_writeback!(a, b, true, sortby)
LinearAlgebra.eigvals!(
    a::_AnyFaerSelfAdjoint,
    b::_AnyFaerSelfAdjoint;
    sortby = _eigensort,
) = _selfadjoint_pair_writeback!(a, b, false, sortby)
