# Histories retain residual diagnostics, rather than full iterate snapshots.
const _IterationEntry{R} = NamedTuple{
    (:iteration, :abs_residual, :rel_residual, :normal_abs_residual, :normal_rel_residual),
    Tuple{Int,R,R,Union{Nothing,R},Union{Nothing,R}},
}

function _iteration_product(op, x::Matrix{P}) where {P}
    eltype(op) === P || throw(ArgumentError("observer scalar types must match"))
    dims = (size(op, 1), size(x, 2))
    args = (op, Ptr{UInt8}(pointer(x)), UInt(size(x, 1)), UInt(size(x, 2)))
    y = GC.@preserve x (
        if P === Float64
            _matrix_result(RustCrate.operator_observer_product, dims, args...)
        else
            _typed_result(P, RustCrate.operator_observer_product, dims, args...)
        end
    )
    try
        return Matrix(y)
    finally
        close(y)
    end
end

function _iteration_snapshot(
    ::Type{P}, pointer, rows, cols, row_stride, col_stride, dims
) where {P}
    (Int(rows), Int(cols)) == dims ||
        throw(DimensionMismatch("native iterate shape mismatch"))
    row_stride == 1 && col_stride >= rows ||
        throw(ArgumentError("invalid native iterate strides"))
    result = Matrix{P}(undef, dims...)
    isempty(result) && return result
    pointer == C_NULL && throw(ArgumentError("native iterate pointer is null"))
    values = Ptr{P}(pointer)
    # The Rust observer borrows its live output MatRef synchronously. Its owned
    # column-major allocation, dimensions and strides remain valid until return.
    for j in 0:(dims[2] - 1), i in 0:(dims[1] - 1)
        result[i + 1, j + 1] = unsafe_load(values, 1 + i + j * Int(col_stride))
    end
    return result
end

function _iteration_observer(
    ::Type{P},
    op,
    rhs,
    callback,
    history;
    normal=false,
    preconditioner=nothing,
    vector=false,
) where {P}
    R = _realtype(P)
    entries = _IterationEntry{R}[]
    observing = history || !isnothing(callback)
    observing || return (entries, false, (_...) -> true, nothing)
    B = Matrix(rhs)
    bnorm = LinearAlgebra.norm(B)
    normalop = nothing
    if normal
        a = adjoint(op)
        if isnothing(preconditioner)
            normalop = a
        else
            m = nothing
            try
                m = adjoint(preconditioner)
                normalop = m * a
            finally
                close(a)
                isnothing(m) || close(m)
            end
        end
    end
    normalref = try
        if isnothing(normalop)
            nothing
        else
            LinearAlgebra.norm(_iteration_product(normalop, B))
        end
    catch
        isnothing(normalop) || close(normalop)
        rethrow()
    end
    dims = (size(op, 2), size(B, 2))
    count = 0
    observer = function (pointer, rows, cols, row_stride, col_stride)
        snapshot = _iteration_snapshot(P, pointer, rows, cols, row_stride, col_stride, dims)
        count += 1
        residual = B - _iteration_product(op, snapshot)
        absolute = R(LinearAlgebra.norm(residual))
        relative = iszero(bnorm) ? absolute : R(absolute / bnorm)
        normalabs = if isnothing(normalop)
            nothing
        else
            R(LinearAlgebra.norm(_iteration_product(normalop, residual)))
        end
        normalrel = if isnothing(normalop)
            nothing
        elseif iszero(normalref)
            normalabs
        else
            R(normalabs / normalref)
        end
        entry = _IterationEntry{R}((count, absolute, relative, normalabs, normalrel))
        history && push!(entries, entry)
        isnothing(callback) && return true
        state = merge(entry, (x=vector ? vec(snapshot) : snapshot,))
        action = callback(state)
        (isnothing(action) || action === true) && return true
        action === false && return false
        return throw(
            ArgumentError(
                "iteration callback must return nothing, true (continue), or false (stop)",
            ),
        )
    end
    return entries, observing, observer, normalop
end
