# Balanced general EVD: faer's EVD driver has no balancing step, so
# `eigen(A; permute=true, scale=true)` balances with faer's `balance_in_place`
# (through `balance`), runs the native EVD on `B = S⁻¹PᵀAPS` and maps the
# eigenvectors back. faer always permutes and scales together.
function _balanced_check(permute::Bool, scale::Bool)
    permute==scale || throw(
        ArgumentError(
            "faer balancing applies permutation and scaling together; pass permute == scale",
        ),
    )
    return permute
end
# Right vectors of A are P*S*v, left vectors P*S⁻¹*w; columns get unit 2-norm.
function _balance_back(f::FaerBalance, v::AbstractMatrix, right::Bool)
    x=similar(v)
    for j in axes(v, 2), i in axes(v, 1)
        x[f.perm[i], j]=right ? v[i, j]*f.scale[i] : v[i, j]/f.scale[i]
    end
    for j in axes(x, 2)
        s=LinearAlgebra.norm(view(x, :, j))
        iszero(s) || (x[:, j]./=s)
    end
    return x
end
function _balanced_eigen(a, vectors; side = :right, sortby, kwargs...)
    f=balance(a)
    try
        vectors || return LinearAlgebra.eigvals(f.B; sortby, kwargs...)
        side in (:right, :left, :both) ||
            throw(ArgumentError("side must be :right, :left or :both"))
        r=LinearAlgebra.eigen(f.B; side, sortby = nothing, kwargs...)
        values=r.values
        order=isnothing(sortby) ? (1:length(values)) : sortperm(values; by = sortby)
        values=values[order]
        side===:right &&
            return LinearAlgebra.Eigen(values, _balance_back(f, r.vectors, true)[:, order])
        side===:left &&
            return (values = values, vectors = _balance_back(f, r.vectors, false)[:, order])
        return (
            values = values,
            left = _balance_back(f, r.left, false)[:, order],
            right = _balance_back(f, r.right, true)[:, order],
        )
    finally
        close(f)
    end
end
