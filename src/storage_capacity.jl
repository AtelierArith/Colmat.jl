_capacity_int(n::UInt) =
    n<=UInt(typemax(Int)) ? Int(n) :
    throw(OverflowError("native capacity does not fit in Int"))

"""
    capacity(A::FaerMatrix) -> (rows, cols)
    capacity(x::Union{FaerVector,FaerRow,FaerDiagonal}) -> Int

Query the current Rust-owned allocation capacity in scalar elements. Matrix
capacity is a pair; collection capacity counts stored vector/row/diagonal entries.
Alignment padding may make capacity larger than the requested reservation.
Querying capacity does not change native dimensions, values or allocation.

Views do not expose allocation capacity. Diagonal storage has fixed dimensions;
its reported capacity does not enable resizing. An unrepresentable native capacity
raises `OverflowError`. The matrix query uses a two-entry Julia output buffer.
"""
function capacity(a::FaerMatrix)
    lock(_matrix_lock) do
        # RustCall mutable primitive slices require initialized elements.
        out=zeros(UInt, 2)
        _native_call(RustCrate.storage_capacities, a, out)
        return (_capacity_int(out[1]), _capacity_int(out[2]))
    end
end
function capacity(x::_OwnedCollection)
    lock(_matrix_lock) do
        return _capacity_int(_native_call(RustCrate.collection_capacity, x))
    end
end
