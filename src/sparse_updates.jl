const _FixedSparseInput=Union{
    FaerSparseMatrix,
    _SparseTransform,
    SparseArrays.AbstractSparseMatrixCSC,
}
_fixed_sparse_input(a::FaerSparseMatrix, ::Type{T}, format) where {T} =
    FaerSparseMatrix{T}(a; format)
_fixed_sparse_input(a::_SparseTransform, ::Type{T}, format) where {T} =
    FaerSparseMatrix{T}(copy(a); format)
_fixed_sparse_input(a::SparseArrays.AbstractSparseMatrixCSC, ::Type{T}, format) where {T} =
    FaerSparseMatrix{T}(a; format)
function _sparse_fixed_update!(
    dest::FaerSparseMatrix{T},
    lhs::_FixedSparseInput,
    rhs::_FixedSparseInput,
    op,
) where {T}
    lock(_matrix_lock) do
        _handle(dest)
        size(dest)==size(lhs)==size(rhs) ||
            throw(DimensionMismatch("fixed sparse update dimensions must match"))
        # Independent snapshots make every destination/source alias safe and
        # finish all conversions before mutating the destination's values.
        a=_fixed_sparse_input(lhs, T, dest.format)
        b=op<2 || lhs===rhs ? a : _fixed_sparse_input(rhs, T, dest.format)
        valid=_native_call(RustCrate.sparse_fixed_update, dest, a, b, Int32(op))
        valid || throw(
            ArgumentError(
                "source stored positions must be contained in the destination pattern, including stored zeros",
            ),
        )
        return dest
    end
end
"""
    add_assign!(dest::FaerSparseMatrix, src)

Compute `dest += src` through faer while retaining dest's stored pattern.
Every stored source position, including stored zeros, must already exist in
dest. Shapes must agree. Inputs are copied/converted to dest's scalar type and
storage orientation before arithmetic; destination type, indices, nnz and live
value views are retained. Aliases are supported. Validation/conversion failures
leave dest unchanged. Accepts native CSC/CSR, native transpose/adjoint, or Julia
CSC sources. No coefficient matrix is densified.
"""
add_assign!(dest::FaerSparseMatrix, src::_FixedSparseInput) =
    _sparse_fixed_update!(dest, src, src, 0)
"""
    sub_assign!(dest::FaerSparseMatrix, src)

Compute `dest -= src` using faer's fixed-pattern kernel. The conversion,
validation, aliasing and value-view contract is the same as `add_assign!`.
"""
sub_assign!(dest::FaerSparseMatrix, src::_FixedSparseInput) =
    _sparse_fixed_update!(dest, src, src, 1)
"""
    add_into!(dest::FaerSparseMatrix, lhs, rhs)

Overwrite existing destination values with `lhs + rhs`, without changing
its pattern. Both input patterns must be contained in dest, even when values
cancel. Unused destination positions are set to zero and retained. Other
conversion and aliasing rules are the same as `add_assign!`.
"""
add_into!(dest::FaerSparseMatrix, lhs::_FixedSparseInput, rhs::_FixedSparseInput) =
    _sparse_fixed_update!(dest, lhs, rhs, 2)
"""
    sub_into!(dest::FaerSparseMatrix, lhs, rhs)

Overwrite existing destination values with `lhs - rhs` through faer's
fixed-pattern kernel. The contract is the same as `add_into!`.
"""
sub_into!(dest::FaerSparseMatrix, lhs::_FixedSparseInput, rhs::_FixedSparseInput) =
    _sparse_fixed_update!(dest, lhs, rhs, 3)
