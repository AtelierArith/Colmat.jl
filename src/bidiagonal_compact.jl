"""
    FaerBidiagonalBasis

Implicit thin or full U/V returned by `bidiagonalize(A; compact=true)`.
Both wrappers share owned native packed reflectors. Products apply those
reflectors through embedding/projection without forming the full square basis.
`Matrix(Q)` materializes exactly `size(Q)`. Copy/deepcopy duplicate storage;
transpose/adjoint/conj share it and can be closed independently.
"""
mutable struct FaerBidiagonalBasis{T,H} <: AbstractMatrix{T}
    handle::Union{Nothing,H}
    m::Int
    n::Int
    blocksize::Int
    left::Bool
    cols::Int
    transposed::Bool
    conjugated::Bool
end
function _handle(q::FaerBidiagonalBasis)
    h=getfield(q, :handle)
    isnothing(h)&&throw(ArgumentError("compact bidiagonal basis has been closed"))
    return h
end
_native_argument(q::FaerBidiagonalBasis) = _handle(q)
_is_native_owner(::FaerBidiagonalBasis) = true
_basis_dim(q) = q.left ? q.m : q.n
_basis_metadata(q) = (UInt(q.m), UInt(q.n), UInt(q.blocksize), _tag(eltype(q)))
Base.size(q::FaerBidiagonalBasis) =
    q.transposed ? (q.cols, _basis_dim(q)) : (_basis_dim(q), q.cols)
Base.size(q::FaerBidiagonalBasis, d::Integer) =
    d<1 ? throw(ArgumentError("dimension must be positive")) : d<=2 ? size(q)[d] : 1
Base.isopen(q::FaerBidiagonalBasis) =
    lock(()->!isnothing(getfield(q, :handle)), _matrix_lock)
function Base.close(q::FaerBidiagonalBasis)
    lock(_matrix_lock) do
        q.handle=nothing
    end
    return nothing
end
function _basis_validate(q::FaerBidiagonalBasis)
    _native_call(RustCrate.compact_bidiagonal_validate, q, _basis_metadata(q)...)
    q.cols in (min(q.m, q.n), _basis_dim(q))||throw(
        DimensionMismatch("compact basis column count mismatch"),
    )
    return nothing
end
_dense_rhs_validate(q::FaerBidiagonalBasis) = _basis_validate(q)
function _compact_basis(
    ::Type{T},
    h,
    left,
    cols;
    transposed = false,
    conjugated = false,
) where {T}
    info=zeros(UInt, 4)
    GC.@preserve info _native_call(RustCrate.compact_bidiagonal_info, h, info)
    info[4]==UInt(_tag(T))||throw(
        ArgumentError("compact basis returned an incompatible scalar"),
    )
    q=FaerBidiagonalBasis{T,typeof(h)}(
        h,
        Int(info[1]),
        Int(info[2]),
        Int(info[3]),
        left,
        cols,
        transposed,
        conjugated,
    )
    _basis_validate(q)
    return q
end
function _basis_copy(
    q::FaerBidiagonalBasis{T};
    transposed = q.transposed,
    conjugated = q.conjugated,
    duplicate = false,
    cols = q.cols,
) where {T}
    lock(_matrix_lock) do
        _basis_validate(q)
        h=duplicate ? _native_call(RustCrate.OwnedCompactBidiagonal, q) : _handle(q)
        return _compact_basis(T, h, q.left, cols; transposed, conjugated)
    end
end
Base.copy(q::FaerBidiagonalBasis) = _basis_copy(q; duplicate = true)
Base.deepcopy_internal(q::FaerBidiagonalBasis, d::IdDict) = get!(() -> copy(q), d, q)
Base.transpose(q::FaerBidiagonalBasis) = _basis_copy(q; transposed = !q.transposed)
Base.conj(q::FaerBidiagonalBasis) = _basis_copy(q; conjugated = !q.conjugated)
Base.adjoint(q::FaerBidiagonalBasis) =
    _basis_copy(q; transposed = !q.transposed, conjugated = !q.conjugated)
function Base.inv(q::FaerBidiagonalBasis)
    size(q, 1)==size(q, 2)||throw(
        DimensionMismatch("a rectangular basis has no two-sided inverse"),
    )
    return adjoint(q)
end
function _basis_coerce(q::FaerBidiagonalBasis, ::Type{T}) where {T}
    _basis_validate(q)
    T===eltype(q)&&return q
    h=_native_call(RustCrate.OwnedCompactBidiagonal, q, _tag(T))
    return _compact_basis(
        T,
        h,
        q.left,
        q.cols;
        transposed = q.transposed,
        conjugated = q.conjugated,
    )
end
function Base.getproperty(q::FaerBidiagonalBasis{T}, name::Symbol) where {T}
    if name in (:offset, :basis_conjugated)
        return lock(_matrix_lock) do
            _basis_validate(q)
            left=q.left==(q.m>=q.n)
            name===:offset ? (left ? 0 : 1) : !left
        end
    end
    if name in (:basis, :factor)
        return lock(_matrix_lock) do
            _basis_validate(q)
            left=q.left==(q.m>=q.n)
            k=min(q.m, q.n)
            count=left ? k : max(0, k-1)
            rows=name===:factor ? q.blocksize : left ? _basis_dim(q) : count
            out=Matrix{T}(undef, rows, count)
            GC.@preserve out _native_call(
                RustCrate.compact_bidiagonal_component,
                q,
                _basis_metadata(q)...,
                q.left,
                name===:factor ? Int32(1) : Int32(0),
                Ptr{UInt8}(pointer(out)),
                UInt(length(out)),
            )
            return out
        end
    end
    return getfield(q, name)
end
Base.propertynames(q::FaerBidiagonalBasis, private::Bool = false) =
    private ? (fieldnames(typeof(q))..., :basis, :factor, :offset, :basis_conjugated) :
    (:basis, :factor, :offset, :basis_conjugated, :blocksize, :transposed, :conjugated)
function _basis_check(q, a, right)
    _basis_validate(q)
    _dense_rhs_validate(a)
    size(a, right ? 2 : 1)==size(q, right ? 1 : 2)||throw(
        DimensionMismatch("compact basis product dimensions must match"),
    )
    return nothing
end
function _basis_requirement(q, T, extent, right; team = nothing)
    _iterative_resource_check(nothing, team)
    info=zeros(UInt, 2)
    GC.@preserve info _native_call(
        RustCrate.compact_bidiagonal_apply_requirements,
        q,
        _basis_metadata(q)...,
        _tag(T),
        q.left,
        UInt(q.cols),
        UInt(extent),
        right,
        q.transposed,
        info,
    )
    return (bytes = Int(info[1]), alignment = Int(info[2]))
end
function _basis_apply(
    q,
    a,
    right;
    T = promote_type(eltype(q), _reflector_type(eltype(a))),
    scratch = nothing,
    team = nothing,
)
    _iterative_resource_check(scratch, team)
    _basis_check(q, a, right)
    T<:_NativeScalar||throw(ArgumentError("unsupported compact basis scalar"))
    extent=right ? size(a, 1) : ndims(a)==1 ? 1 : size(a, 2)
    req=_basis_requirement(q, T, extent, right; team)
    work=isnothing(scratch) ? FaerScratch(req.bytes; alignment = req.alignment) : scratch
    try
        cap=scratch_capacity(work)
        cap.bytes>=req.bytes&&cap.alignment>=req.alignment||throw(
            ArgumentError(
                "compact basis scratch capacity/alignment is insufficient; call reserve!",
            ),
        )
        h=_basis_coerce(q, T)
        data=a isa AbstractVector ? reshape(Vector{T}(a), :, 1) : Matrix{T}(a)
        expected=a isa AbstractVector ? (length(a), 1) : size(a)
        data isa Matrix{T}&&size(data)==expected||throw(
            DimensionMismatch(
                "compact basis input conversion changed scalar storage or shape",
            ),
        )
        shape=right ? (extent, size(q, 2)) : (size(q, 1), extent)
        out=Matrix{T}(undef, shape)
        args=(
            h,
            work,
            _basis_metadata(h)...,
            h.left,
            UInt(h.cols),
            UInt(extent),
            right,
            h.transposed,
            h.conjugated,
            Ptr{UInt8}(pointer(data)),
            Ptr{UInt8}(pointer(out)),
        )
        GC.@preserve data out begin
            _native_call(RustCrate.compact_bidiagonal_apply, args..., team)
        end
        return out
    finally
        isnothing(scratch)&&close(work)
    end
end
function _basis_product(q, a, right)
    lock(_matrix_lock) do
        out=_basis_apply(q, a, right)
        return a isa StridedVector ? vec(out) :
               a isa AbstractVector ? FaerVector(vec(out)) : FaerMatrix(out)
    end
end
Base.:*(q::FaerBidiagonalBasis, a::_HouseholderArray) = _basis_product(q, a, false)
Base.:*(a::_HouseholderMatrix, q::FaerBidiagonalBasis) = _basis_product(q, a, true)
Base.:*(a::FaerBidiagonalBasis, b::FaerBidiagonalBasis) = a*Matrix(b)
Base.:\(q::FaerBidiagonalBasis, a::_HouseholderArray) = q'*a
Base.:/(a::_HouseholderMatrix, q::FaerBidiagonalBasis) = a*q'
function Base.Matrix(q::FaerBidiagonalBasis{T}) where {T}
    lock(_matrix_lock) do
        _basis_validate(q)
        base=q.transposed ? _basis_copy(q; transposed = false) : q
        out=_basis_apply(base, Matrix{T}(LinearAlgebra.I, base.cols, base.cols), false)
        return q.transposed ? copy(transpose(out)) : out
    end
end
Base.Matrix{T}(q::FaerBidiagonalBasis) where {T} = Matrix{T}(Matrix(q))
Base.Array(q::FaerBidiagonalBasis) = Matrix(q)
FaerMatrix(q::FaerBidiagonalBasis) = FaerMatrix(Matrix(q))
function Base.getindex(q::FaerBidiagonalBasis{T}, i::Int, j::Int) where {T}
    checkbounds(q, i, j)
    x=zeros(T, size(q, 2))
    x[j]=one(T)
    return (q*x)[i]
end
# `dest = alpha*Q*a + beta*dest` read from and written to view descriptors by
# one native call (Rust `compact_bidiagonal_mul_into`) when `a` and `dest`
# share the working scalar. The embedding/projection work matrix is the
# declared scratch; the source is snapshot when it overlaps `dest`; inputs and
# the combined result are checked finite before the first write.
function _basis_mul_native!(dest, q, a, alpha, beta, right; scratch, team)
    P=promote_type(eltype(q), eltype(a), eltype(dest))
    eltype(dest)===eltype(a)===P &&
        promote_type(P, typeof(alpha), typeof(beta))<:_NativeScalar ||
        return nothing
    target=_gemm_destination(dest)
    src=_gemm_input(a)
    (isnothing(target) || isnothing(src)) && return nothing
    _iterative_resource_check(scratch, team)
    extent=right ? size(a, 1) : ndims(a)==1 ? 1 : size(a, 2)
    req=_basis_requirement(q, P, extent, right; team)
    work=isnothing(scratch) ? FaerScratch(req.bytes; alignment = req.alignment) : scratch
    try
        cap=scratch_capacity(work)
        cap.bytes>=req.bytes&&cap.alignment>=req.alignment||throw(
            ArgumentError(
                "compact basis scratch capacity/alignment is insufficient; call reserve!",
            ),
        )
        h=_basis_coerce(q, P)
        al, be=Ref{P}(P(alpha)), Ref{P}(P(beta))
        p, d=_view_source(target)
        r, e=_view_source(src)
        GC.@preserve d e al be _native_call(
            RustCrate.compact_bidiagonal_mul_into,
            h,
            work,
            UInt[h.m, h.n, h.blocksize, h.cols, extent],
            _tag(P),
            h.left,
            right,
            h.transposed,
            h.conjugated,
            p,
            pointer(d),
            r,
            pointer(e),
            Ptr{UInt8}(Base.unsafe_convert(Ptr{P}, al)),
            Ptr{UInt8}(Base.unsafe_convert(Ptr{P}, be)),
            iszero(al[]),
            iszero(be[]),
            team,
        )
    finally
        isnothing(scratch)&&close(work)
    end
    return dest
end
# Working scalar of a basis product: operands and destination only; the
# coefficients convert to it.
function _basis_mul_scalar(dest, q, a, alpha, beta)
    P=promote_type(eltype(q), _reflector_type(eltype(a)), eltype(dest))
    P<:_NativeScalar && promote_type(P, typeof(alpha), typeof(beta))<:_NativeScalar ||
        throw(ArgumentError("unsupported compact basis product scalar"))
    return P
end
function _basis_mul!(dest, q, a, alpha, beta, right; scratch = nothing, team = nothing)
    lock(_matrix_lock) do
        _check_writable(dest)
        _basis_check(q, a, right)
        _dense_rhs_validate(dest)
        expected=right ? (size(a, 1), size(q, 2)) :
                 a isa AbstractVector ? (size(q, 1),) : (size(q, 1), size(a, 2))
        size(dest)==expected||throw(
            DimensionMismatch("compact basis destination shape mismatch"),
        )
        native=_basis_mul_native!(dest, q, a, alpha, beta, right; scratch, team)
        isnothing(native)||return dest
        P=_basis_mul_scalar(dest, q, a, alpha, beta)
        out=_basis_apply(q, a, right; T = P, scratch, team)
        result=dest isa AbstractVector ? vec(out) : out
        combined=iszero(beta) ? alpha .* result : alpha .* result .+ beta .* Array(dest)
        buffer=dest isa AbstractVector ? Vector{eltype(dest)}(combined) :
               Matrix{eltype(dest)}(combined)
        all(isfinite, buffer)||throw(
            ArgumentError("compact basis product overflowed the destination scalar"),
        )
        copyto!(dest isa FaerBorrowedArray ? parent(dest) : dest, buffer)
        return dest
    end
end
LinearAlgebra.mul!(
    dest::_HouseholderArray,
    q::FaerBidiagonalBasis,
    a::_HouseholderArray,
    alpha::Number,
    beta::Number;
    scratch = nothing,
    team = nothing,
) = _basis_mul!(dest, q, a, alpha, beta, false; scratch, team)
LinearAlgebra.mul!(
    dest::_HouseholderMatrix,
    a::_HouseholderMatrix,
    q::FaerBidiagonalBasis,
    alpha::Number,
    beta::Number;
    scratch = nothing,
    team = nothing,
) = _basis_mul!(dest, q, a, alpha, beta, true; scratch, team)
LinearAlgebra.mul!(
    dest::_HouseholderArray,
    q::FaerBidiagonalBasis,
    a::_HouseholderArray;
    scratch = nothing,
    team = nothing,
) = mul!(dest, q, a, 1, 0; scratch, team)
LinearAlgebra.mul!(
    dest::_HouseholderMatrix,
    a::_HouseholderMatrix,
    q::FaerBidiagonalBasis;
    scratch = nothing,
    team = nothing,
) = mul!(dest, a, q, 1, 0; scratch, team)
LinearAlgebra.lmul!(
    q::FaerBidiagonalBasis,
    a::_HouseholderArray;
    scratch = nothing,
    team = nothing,
) = mul!(a, q, a; scratch, team)
LinearAlgebra.rmul!(
    a::_HouseholderMatrix,
    q::FaerBidiagonalBasis;
    scratch = nothing,
    team = nothing,
) = mul!(a, a, q; scratch, team)
LinearAlgebra.ldiv!(
    q::FaerBidiagonalBasis,
    a::_HouseholderArray;
    scratch = nothing,
    team = nothing,
) = mul!(a, q', a; scratch, team)
LinearAlgebra.ldiv!(
    dest::_HouseholderArray,
    q::FaerBidiagonalBasis,
    a::_HouseholderArray;
    scratch = nothing,
    team = nothing,
) = mul!(dest, q', a; scratch, team)
LinearAlgebra.rdiv!(
    a::_HouseholderMatrix,
    q::FaerBidiagonalBasis;
    scratch = nothing,
    team = nothing,
) = mul!(a, a, q'; scratch, team)

function _compact_bidiagonal_svd(
    f;
    full::Bool = false,
    scratch = nothing,
    team = nothing,
    recursion_threshold::Integer = 128,
)
    req=scratch_requirements(LinearAlgebra.svd, f; full, team, recursion_threshold)
    work=isnothing(scratch) ? FaerScratch(req.bytes; alignment = req.alignment) : scratch
    core=nothing
    try
        cap=scratch_capacity(work)
        cap.bytes>=req.bytes&&cap.alignment>=req.alignment||throw(
            ArgumentError(
                "saved compact bidiagonal SVD scratch capacity/alignment is insufficient; call reserve!",
            ),
        )
        core=bidiagonal_svd(
            f.B.dv,
            f.B.ev;
            uplo = f.B.uplo,
            scratch = work,
            team,
            recursion_threshold,
        )
        k=size(f.B, 1)
        m, n=size(f)
        T=eltype(f)
        u=_basis_copy(f.U; cols = full ? m : k)
        v=_basis_copy(f.V; cols = full ? n : k)
        if full
            left=Matrix{T}(LinearAlgebra.I, m, m)
            left[1:k, 1:k]=Matrix(core.U)
            right=Matrix{T}(LinearAlgebra.I, n, n)
            right[1:k, 1:k]=Matrix(core.Vt)
        else
            left=Matrix(core.U)
            right=Matrix(core.Vt)
        end
        l=_basis_apply(u, left, false; scratch = work, team)
        r=_basis_apply(v', right, true; scratch = work, team)
        return LinearAlgebra.SVD(FaerMatrix(l), core.S, FaerMatrix(r))
    finally
        if !isnothing(core)
            close(core.U)
            close(core.Vt)
        end
        isnothing(scratch)&&close(work)
    end
end
function _basis_mul_requirement(dest, q, a, alpha, beta, right; team = nothing)
    lock(_matrix_lock) do
        _basis_check(q, a, right)
        _dense_rhs_validate(dest)
        expected=right ? (size(a, 1), size(q, 2)) :
                 a isa AbstractVector ? (size(q, 1),) : (size(q, 1), size(a, 2))
        size(dest)==expected||throw(
            DimensionMismatch("compact basis destination shape mismatch"),
        )
        P=_basis_mul_scalar(dest, q, a, alpha, beta)
        P<:_NativeScalar||throw(ArgumentError("unsupported compact basis product scalar"))
        return _basis_requirement(
            q,
            P,
            right ? size(a, 1) : ndims(a)==1 ? 1 : size(a, 2),
            right;
            team,
        )
    end
end
scratch_requirements(
    ::typeof(LinearAlgebra.mul!),
    dest::_HouseholderArray,
    q::FaerBidiagonalBasis,
    a::_HouseholderArray,
    alpha::Number = 1,
    beta::Number = 0;
    team = nothing,
) = _basis_mul_requirement(dest, q, a, alpha, beta, false; team)
scratch_requirements(
    ::typeof(LinearAlgebra.mul!),
    dest::_HouseholderMatrix,
    a::_HouseholderMatrix,
    q::FaerBidiagonalBasis,
    alpha::Number = 1,
    beta::Number = 0;
    team = nothing,
) = _basis_mul_requirement(dest, q, a, alpha, beta, true; team)
scratch_requirements(
    ::typeof(LinearAlgebra.lmul!),
    q::FaerBidiagonalBasis,
    a::_HouseholderArray;
    team = nothing,
) = scratch_requirements(LinearAlgebra.mul!, a, q, a; team)
scratch_requirements(
    ::typeof(LinearAlgebra.rmul!),
    a::_HouseholderMatrix,
    q::FaerBidiagonalBasis;
    team = nothing,
) = scratch_requirements(LinearAlgebra.mul!, a, a, q; team)
scratch_requirements(
    ::typeof(LinearAlgebra.ldiv!),
    q::FaerBidiagonalBasis,
    a::_HouseholderArray;
    team = nothing,
) = scratch_requirements(LinearAlgebra.mul!, a, q', a; team)
scratch_requirements(
    ::typeof(LinearAlgebra.ldiv!),
    dest::_HouseholderArray,
    q::FaerBidiagonalBasis,
    a::_HouseholderArray;
    team = nothing,
) = scratch_requirements(LinearAlgebra.mul!, dest, q', a; team)
scratch_requirements(
    ::typeof(LinearAlgebra.rdiv!),
    a::_HouseholderMatrix,
    q::FaerBidiagonalBasis;
    team = nothing,
) = scratch_requirements(LinearAlgebra.mul!, a, a, q'; team)
for (fun, signature, args) in (
    (
        :mul!,
        :(
            dest::_HouseholderArray,
            q::FaerBidiagonalBasis,
            a::_HouseholderArray,
            scales::Number...,
        ),
        :(dest, q, a, scales...),
    ),
    (
        :mul!,
        :(
            dest::_HouseholderMatrix,
            a::_HouseholderMatrix,
            q::FaerBidiagonalBasis,
            scales::Number...,
        ),
        :(dest, a, q, scales...),
    ),
    (:lmul!, :(q::FaerBidiagonalBasis, a::_HouseholderArray), :(q, a)),
    (:rmul!, :(a::_HouseholderMatrix, q::FaerBidiagonalBasis), :(a, q)),
    (:ldiv!, :(q::FaerBidiagonalBasis, a::_HouseholderArray), :(q, a)),
    (
        :ldiv!,
        :(dest::_HouseholderArray, q::FaerBidiagonalBasis, a::_HouseholderArray),
        :(dest, q, a),
    ),
    (:rdiv!, :(a::_HouseholderMatrix, q::FaerBidiagonalBasis), :(a, q)),
)
    parameters=signature.args
    callargs=args.args
    @eval begin
        function FaerScratch(::typeof(LinearAlgebra.$fun), $(parameters...); kwargs...)
            req=scratch_requirements(LinearAlgebra.$fun, $(callargs...); kwargs...)
            return FaerScratch(req.bytes; alignment = req.alignment)
        end
        function reserve!(
            s::FaerScratch,
            ::typeof(LinearAlgebra.$fun),
            $(parameters...);
            kwargs...,
        )
            req=scratch_requirements(LinearAlgebra.$fun, $(callargs...); kwargs...)
            return reserve!(s, req.bytes; alignment = req.alignment)
        end
    end
end
