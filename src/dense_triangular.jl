# Direct dense triangular solves and destructive inverses
# (deps/rust/src/dense_triangular.rs).
const _DenseTriangularView{T}=FaerView{T,2,<:FaerMatrix{T}}
const _DenseTriangularBase{T}=Union{FaerMatrix{T},_DenseTriangularView{T}}
const _DenseTriangularParent{T}=Union{
    _DenseTriangularBase{T},
    LinearAlgebra.Transpose{T,<:_DenseTriangularBase{T}},
    LinearAlgebra.Adjoint{T,<:_DenseTriangularBase{T}},
    # Elementwise conjugates, e.g. `transpose(LowerTriangular(A'))`.
    LinearAlgebra.Transpose{T,<:LinearAlgebra.Adjoint{T,<:_DenseTriangularBase{T}}},
    LinearAlgebra.Adjoint{T,<:LinearAlgebra.Transpose{T,<:_DenseTriangularBase{T}}},
}
const _DenseTriangular=Union{
    LinearAlgebra.LowerTriangular{<:Any,<:_DenseTriangularParent},
    LinearAlgebra.UpperTriangular{<:Any,<:_DenseTriangularParent},
    LinearAlgebra.UnitLowerTriangular{<:Any,<:_DenseTriangularParent},
    LinearAlgebra.UnitUpperTriangular{<:Any,<:_DenseTriangularParent},
}
# Parents the destructive inverse overwrites (a conjugated parent would need a
# second pass, so adjoint parents keep using `inv`).
const _DenseInvertParent{T}=Union{
    _DenseTriangularBase{T},
    LinearAlgebra.Transpose{T,<:_DenseTriangularBase{T}},
}
# Native destinations written in place through their view descriptor.
const _DenseTriangularNative=Union{
    FaerMatrix,
    FaerVector,
    FaerView{<:Any,<:Any,<:Union{FaerMatrix,FaerVector}},
}
_dense_triangular_direct(x::_DenseTriangularNative, ::Type{P}) where {P} = eltype(x)===P
_dense_triangular_direct(x::FaerBorrowedArray, ::Type{P}) where {P} = eltype(x)===P
_dense_triangular_direct(x::StridedVecOrMat, ::Type{P}) where {P} =
    eltype(x)===P && _faerview_valid(x)
_dense_triangular_direct(x, ::Type) = false

_dense_triangular_owner(x::_DenseTriangularNative) = first(_view_source(x))
_dense_triangular_owner(x) = nothing
function _dense_triangular_check_team(team)
    isnothing(team) || team isa FaerTeam || throw(ArgumentError("team must be a FaerTeam"))
    isnothing(team) || _handle(team)
    return nothing
end
# Rows (left solve) or columns (right solve) of the RHS that must match.
function _dense_triangular_prepare(a, b, right::Bool, team)
    _dense_triangular_check_team(team)
    _view_source(parent(a))
    b isa Union{_DenseTriangularNative,FaerBorrowedArray} && _spectral_rhs_open(b)
    right && ndims(b)!=2 && throw(ArgumentError("right triangular solves need a matrix"))
    size(a, 1)==size(b, right ? 2 : 1) ||
        throw(DimensionMismatch("triangular solve dimensions must match"))
    return promote_type(eltype(a), _native_input_type(eltype(b)))
end
function _dense_triangular_query(a, ::Type{P}) where {P}
    p, d=_view_source(parent(a))
    info=zeros(UInt, 2)
    GC.@preserve d info _native_call(
        RustCrate.dense_triangular_requirements,
        p,
        pointer(d),
        _tag(P),
        pointer(info),
    )
    return (bytes = Int(info[1]), alignment = Int(info[2]))
end
function _dense_triangular_check_singular(a)
    _isunit(a) && return nothing
    p, d=_view_source(parent(a))
    i=GC.@preserve d Int(_native_call(RustCrate.dense_triangular_zero_diagonal, p, pointer(d)))
    iszero(i) || throw(LinearAlgebra.SingularException(i))
    return nothing
end
function _with_dense_triangular_scratch(fun, req, scratch)
    isnothing(scratch) && iszero(req.bytes) && return fun(_sparse_triangular_empty())
    temporary=isnothing(scratch)
    work=temporary ? FaerScratch(req.bytes; alignment = req.alignment) : scratch
    work isa FaerScratch || throw(ArgumentError("scratch must be a FaerScratch"))
    try
        cap=scratch_capacity(work)
        cap.bytes>=req.bytes && cap.alignment>=req.alignment || throw(
            ArgumentError("insufficient dense triangular scratch; call reserve!"),
        )
        return fun(work)
    finally
        temporary && close(work)
    end
end
# Solve op(A) x = x (or x op(A) = x) in place in `x`, which holds P values.
function _dense_triangular_native!(x, a, right::Bool, work, team)
    p, d=_view_source(parent(a))
    lower, unit=_islower(a), _isunit(a)
    code=if x isa _DenseTriangularNative
        q, e=_view_source(x)
        GC.@preserve d e _native_call(
            RustCrate.dense_triangular_into,
            q,
            pointer(e),
            p,
            pointer(d),
            lower,
            unit,
            right,
            work,
            team,
        )
    else
        y=x isa FaerBorrowedArray ? parent(x) : x
        m=size(y, 1)
        n=ndims(y)==1 ? 1 : size(y, 2)
        rs=stride(y, 1)
        cs=ndims(y)==1 ? m*rs : stride(y, 2)
        GC.@preserve d y _native_call(
            RustCrate.dense_triangular_raw,
            p,
            pointer(d),
            Ptr{UInt8}(pointer(y)),
            UInt(m),
            UInt(n),
            rs,
            cs,
            _tag(eltype(y)),
            lower,
            unit,
            right,
            work,
            team,
        )
    end
    iszero(code) || throw(LinearAlgebra.SingularException(Int(code)))
    return x
end
# A fresh RHS copy in the promoted type; it never aliases `b`.
_dense_triangular_copy(b::AbstractMatrix, ::Type{P}) where {P} =
    b isa FaerMatrix && eltype(b)===P ? copy(b) : FaerMatrix{P}(b)
_dense_triangular_copy(b::_BorrowVector, ::Type{P}) where {P} =
    eltype(b)===P ? copy(b) : FaerVector{P}(b)
_dense_triangular_copy(b::AbstractVector, ::Type{P}) where {P} = Vector{P}(b)
# The triangle is read through a raw pointer while the destination owner is
# borrowed mutably; a destination sharing its owner solves from a snapshot.
function _dense_triangular_unaliased(a, dest)
    owner=_dense_triangular_owner(dest)
    isnothing(owner) && return a
    first(_view_source(parent(a)))===owner || return a
    return _triangular_wrap(a, copy(parent(a)))
end

function _dense_triangular_solve(a, b, right::Bool; scratch = nothing, team = nothing)
    lock(_matrix_lock) do
        P=_dense_triangular_prepare(a, b, right, team)
        req=_dense_triangular_query(a, P)
        _with_dense_triangular_scratch(req, scratch) do work
            x=_dense_triangular_copy(b, P)
            try
                return _dense_triangular_native!(x, a, right, work, team)
            catch
                x isa Union{FaerMatrix,FaerVector} && close(x)
                rethrow()
            end
        end
    end
end
# Write op(A) \ b (or b / op(A)) into dest. A destination holding the promoted
# scalar that is a native owner/view or an injective Julia strided array is
# solved in place; with dest === b nothing else is copied. Other destinations
# stage one promoted native copy. All validation (singularity, scratch, team,
# closed resources) precedes the first write to dest.
function _dense_triangular_write!(
    dest,
    a,
    b,
    right::Bool;
    scratch = nothing,
    team = nothing,
)
    lock(_matrix_lock) do
        P=_dense_triangular_prepare(a, b, right, team)
        dest isa Union{_DenseTriangularNative,FaerBorrowedArray} && _spectral_rhs_open(dest)
        size(dest)==size(b) ||
            throw(DimensionMismatch("triangular output size must match RHS"))
        direct=_dense_triangular_direct(dest, P)
        req=_dense_triangular_query(a, P)
        _with_dense_triangular_scratch(req, scratch) do work
            if direct
                _dense_triangular_check_singular(a)
                src=_dense_triangular_unaliased(a, dest)
                try
                    dest===b || copyto!(dest, b)
                    _dense_triangular_native!(dest, src, right, work, team)
                finally
                    src===a || close(parent(src))
                end
            else
                x=_dense_triangular_copy(b, P)
                try
                    _dense_triangular_native!(x, a, right, work, team)
                    copyto!(dest, x)
                finally
                    x isa Union{FaerMatrix,FaerVector} && close(x)
                end
            end
            return dest
        end
    end
end

for C in (:LowerTriangular, :UpperTriangular, :UnitLowerTriangular, :UnitUpperTriangular)
    @eval begin
        Base.:\(
            a::LinearAlgebra.$C{T,<:_DenseTriangularParent{T}},
            b::AbstractVecOrMat{<:Number};
            kwargs...,
        ) where {T<:_NativeScalar} = _dense_triangular_solve(a, b, false; kwargs...)
        Base.:/(
            b::AbstractMatrix{<:Number},
            a::LinearAlgebra.$C{T,<:_DenseTriangularParent{T}};
            kwargs...,
        ) where {T<:_NativeScalar} = _dense_triangular_solve(a, b, true; kwargs...)
        LinearAlgebra.ldiv!(
            a::LinearAlgebra.$C{T,<:_DenseTriangularParent{T}},
            b::AbstractVecOrMat;
            kwargs...,
        ) where {T<:_NativeScalar} = _dense_triangular_write!(b, a, b, false; kwargs...)
        LinearAlgebra.ldiv!(
            dest::AbstractVecOrMat,
            a::LinearAlgebra.$C{T,<:_DenseTriangularParent{T}},
            b::AbstractVecOrMat;
            kwargs...,
        ) where {T<:_NativeScalar} = _dense_triangular_write!(dest, a, b, false; kwargs...)
        LinearAlgebra.rdiv!(
            b::AbstractMatrix,
            a::LinearAlgebra.$C{T,<:_DenseTriangularParent{T}};
            kwargs...,
        ) where {T<:_NativeScalar} = _dense_triangular_write!(b, a, b, true; kwargs...)
    end
end

# Resolve structured RHS methods provided by LinearAlgebra and SparseArrays.
for C in (:LowerTriangular, :UpperTriangular, :UnitLowerTriangular, :UnitUpperTriangular)
    for R in (
        LinearAlgebra.LowerTriangular,
        LinearAlgebra.UpperTriangular,
        LinearAlgebra.UnitLowerTriangular,
        LinearAlgebra.UnitUpperTriangular,
        LinearAlgebra.Bidiagonal,
        LinearAlgebra.UpperHessenberg,
        SparseArrays.AbstractSparseMatrixCSC,
        SparseArrays.AbstractCompressedVector,
    )
        @eval Base.:\(
            a::LinearAlgebra.$C{T,<:_DenseTriangularParent{T}},
            b::$R{<:Number};
            kwargs...,
        ) where {T<:_NativeScalar} = _dense_triangular_solve(a, b, false; kwargs...)
    end
    for R in (
        LinearAlgebra.LowerTriangular,
        LinearAlgebra.UpperTriangular,
        LinearAlgebra.UnitLowerTriangular,
        LinearAlgebra.UnitUpperTriangular,
        LinearAlgebra.Bidiagonal,
        LinearAlgebra.UpperHessenberg,
    )
        @eval Base.:/(
            b::$R{<:Number},
            a::LinearAlgebra.$C{T,<:_DenseTriangularParent{T}};
            kwargs...,
        ) where {T<:_NativeScalar} = _dense_triangular_solve(a, b, true; kwargs...)
    end
    # Row vectors keep LinearAlgebra's wrapped-vector result.
    @eval begin
        Base.:/(
            u::LinearAlgebra.Transpose{<:Number,<:AbstractVector},
            a::LinearAlgebra.$C{T,<:_DenseTriangularParent{T}},
        ) where {T<:_NativeScalar} = transpose(transpose(a)\parent(u))
        Base.:/(
            u::LinearAlgebra.Adjoint{<:Number,<:AbstractVector},
            a::LinearAlgebra.$C{T,<:_DenseTriangularParent{T}},
        ) where {T<:_NativeScalar} = adjoint(adjoint(a)\parent(u))
    end
end

@doc """
    ldiv!(A::Triangular, B; scratch=nothing, team=nothing)
    ldiv!(Y, A::Triangular, B; scratch=nothing, team=nothing)
    rdiv!(B, A::Triangular; scratch=nothing, team=nothing)
    A \\ B, B / A

Solve with a `LowerTriangular`/`UpperTriangular`/`UnitLowerTriangular`/
`UnitUpperTriangular` wrapper of a `FaerMatrix`, a native matrix view, or the
`transpose`/`adjoint` of either. The triangle is borrowed through its view
descriptor (no copy for transposed, adjoint or strided parents), and faer's
`solve_{unit_}{lower,upper}_triangular_in_place_with_conj` runs directly on the
destination when it is a `FaerMatrix`/`FaerVector`, a native view, a
[`faerview`](@ref) borrow or an injective Julia strided array whose element type
is `promote_type(eltype(A), eltype(B))`. `rdiv!` solves `X * A = B` as
`transpose(A) * transpose(X) = transpose(B)` on the same storage.

faer's dense triangular kernels take no workspace: `scratch` only holds the
promoted copy of the triangle when `eltype(A)` is narrower than the promoted
type. Query with `scratch_requirements(ldiv!, A, B)` (also `\\`, `rdiv!` and the
three-argument form), create with `FaerScratch(ldiv!, A, B)` or grow with
`reserve!(work, ldiv!, A, B)`; it is zero bytes without promotion. With
`Y === B` (or `rdiv!`) and that scratch, the Colmat side allocates nothing;
faer's blocked kernels still allocate matmul packing buffers above their base
case (measured: none at n = 70, 12 at n = 300). `team` runs the kernel on a
caller-owned [`FaerTeam`](@ref). `Y !== B` copies `B` into `Y`; narrower or
non-strided destinations stage one promoted copy. A destination sharing the
triangle's native owner is solved from a snapshot of the triangle. Zero
diagonals throw `SingularException` and closed or undersized resources throw
before any write.
""" LinearAlgebra.ldiv!(
    ::LinearAlgebra.LowerTriangular{T,<:_DenseTriangularParent{T}},
    ::AbstractVecOrMat,
) where {T<:_NativeScalar}

function _dense_triangular_requirements(a, b, right::Bool; team = nothing)
    lock(_matrix_lock) do
        P=_dense_triangular_prepare(a, b, right, team)
        return _dense_triangular_query(a, P)
    end
end
for (F, right) in ((:(typeof(LinearAlgebra.ldiv!)), false), (:(typeof(\)), false))
    @eval begin
        scratch_requirements(
            ::$F,
            a::_DenseTriangular,
            b::AbstractVecOrMat;
            team = nothing,
        ) = _dense_triangular_requirements(a, b, $right; team)
        FaerScratch(fun::$F, a::_DenseTriangular, b::AbstractVecOrMat; team = nothing) =
            _new_operator_workspace(fun, a, b; team)
        reserve!(
            s::FaerScratch,
            fun::$F,
            a::_DenseTriangular,
            b::AbstractVecOrMat;
            team = nothing,
        ) = _reserve_operator_workspace(s, fun, a, b; team)
    end
end
scratch_requirements(
    ::typeof(LinearAlgebra.rdiv!),
    b::AbstractMatrix,
    a::_DenseTriangular;
    team = nothing,
) = _dense_triangular_requirements(a, b, true; team)
FaerScratch(
    fun::typeof(LinearAlgebra.rdiv!),
    b::AbstractMatrix,
    a::_DenseTriangular;
    team = nothing,
) = _new_operator_workspace(fun, b, a; team)
reserve!(
    s::FaerScratch,
    fun::typeof(LinearAlgebra.rdiv!),
    b::AbstractMatrix,
    a::_DenseTriangular;
    team = nothing,
) = _reserve_operator_workspace(s, fun, b, a; team)
function scratch_requirements(
    ::typeof(LinearAlgebra.ldiv!),
    dest::AbstractVecOrMat,
    a::_DenseTriangular,
    b::AbstractVecOrMat;
    team = nothing,
)
    size(dest)==size(b) ||
        throw(DimensionMismatch("triangular output size must match RHS"))
    return _dense_triangular_requirements(a, b, false; team)
end
FaerScratch(
    fun::typeof(LinearAlgebra.ldiv!),
    dest::AbstractVecOrMat,
    a::_DenseTriangular,
    b::AbstractVecOrMat;
    team = nothing,
) = _new_operator_workspace(fun, dest, a, b; team)
reserve!(
    s::FaerScratch,
    fun::typeof(LinearAlgebra.ldiv!),
    dest::AbstractVecOrMat,
    a::_DenseTriangular,
    b::AbstractVecOrMat;
    team = nothing,
) = _reserve_operator_workspace(s, fun, dest, a, b; team)

# Destructive inverse: faer's `invert_*` take a separate source, staged in scratch.
function _dense_invert_requirements(a; team = nothing)
    lock(_matrix_lock) do
        _dense_triangular_check_team(team)
        _view_source(parent(a))
        info=zeros(UInt, 2)
        GC.@preserve info _native_call(
            RustCrate.dense_triangular_invert_requirements,
            UInt(size(a, 1)),
            _tag(eltype(a)),
            pointer(info),
        )
        return (bytes = Int(info[1]), alignment = Int(info[2]))
    end
end
function _dense_invert!(a; scratch = nothing, team = nothing)
    lock(_matrix_lock) do
        _dense_triangular_check_team(team)
        p, d=_view_source(parent(a))
        req=_dense_invert_requirements(a; team)
        _with_dense_triangular_scratch(req, scratch) do work
            code=GC.@preserve d _native_call(
                RustCrate.dense_triangular_invert,
                p,
                pointer(d),
                _islower(a),
                _isunit(a),
                work,
                team,
            )
            iszero(code) || throw(LinearAlgebra.SingularException(Int(code)))
            return a
        end
    end
end
for C in (:LowerTriangular, :UpperTriangular, :UnitLowerTriangular, :UnitUpperTriangular)
    @eval begin
        LinearAlgebra.inv!(
            a::LinearAlgebra.$C{T,<:_DenseInvertParent{T}};
            kwargs...,
        ) where {T<:_NativeScalar} = _dense_invert!(a; kwargs...)
        scratch_requirements(
            ::typeof(LinearAlgebra.inv!),
            a::LinearAlgebra.$C{T,<:_DenseInvertParent{T}};
            team = nothing,
        ) where {T<:_NativeScalar} = _dense_invert_requirements(a; team)
        FaerScratch(
            fun::typeof(LinearAlgebra.inv!),
            a::LinearAlgebra.$C{T,<:_DenseInvertParent{T}};
            team = nothing,
        ) where {T<:_NativeScalar} = _new_operator_workspace(fun, a; team)
        reserve!(
            s::FaerScratch,
            fun::typeof(LinearAlgebra.inv!),
            a::LinearAlgebra.$C{T,<:_DenseInvertParent{T}};
            team = nothing,
        ) where {T<:_NativeScalar} = _reserve_operator_workspace(s, fun, a; team)
    end
end
@doc """
    LinearAlgebra.inv!(A::Triangular; scratch=nothing, team=nothing)

Overwrite the selected triangle of the native parent of `A` (a `FaerMatrix`, a
native matrix view or the `transpose` of either) with the triangle of `inv(A)`
and return `A`, like LinearAlgebra's `inv!` for strided triangles. The opposite
strict triangle is left untouched (and the diagonal for unit wrappers). faer's
`triangular_inverse::invert_*` kernels write into a separate destination, so the
triangle is first copied into `scratch`: query with
`scratch_requirements(inv!, A)`, create with `FaerScratch(inv!, A)` or grow with
`reserve!(work, inv!, A)`. A zero diagonal throws `SingularException` before
any write. `team` runs the kernel on a caller-owned [`FaerTeam`](@ref).
"""
LinearAlgebra.inv!(
    ::LinearAlgebra.LowerTriangular{T,<:_DenseInvertParent{T}},
) where {T<:_NativeScalar}
