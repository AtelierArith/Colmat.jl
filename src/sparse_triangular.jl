const _SparseTriangularParent{T}=Union{
    FaerSparseMatrix{T},
    LinearAlgebra.Transpose{T,<:FaerSparseMatrix{T}},
    LinearAlgebra.Adjoint{T,<:FaerSparseMatrix{T}},
    FaerBorrowedSparseMatrix{T},
    LinearAlgebra.Transpose{T,<:FaerBorrowedSparseMatrix{T}},
    LinearAlgebra.Adjoint{T,<:FaerBorrowedSparseMatrix{T}},
}
const _SparseTriangular=Union{
    LinearAlgebra.LowerTriangular{<:Any,<:_SparseTriangularParent},
    LinearAlgebra.UpperTriangular{<:Any,<:_SparseTriangularParent},
    LinearAlgebra.UnitLowerTriangular{<:Any,<:_SparseTriangularParent},
    LinearAlgebra.UnitUpperTriangular{<:Any,<:_SparseTriangularParent},
}
# The stored owner and the operation applied to it; Rust maps CSR storage and
# transpose/adjoint onto faer's column-major kernels without copying the owner.
_sparse_triangular_owner(p::FaerSparseMatrix) = (p, Int32(0))
# A borrowed Julia CSC matrix is read through the per-call 0-based buffers.
_sparse_triangular_owner(p::FaerBorrowedSparseMatrix) = (p, Int32(0))
_sparse_triangular_open(source::FaerSparseMatrix) = _handle(source)
_sparse_triangular_open(::FaerBorrowedSparseMatrix) = nothing
# Rust reads the borrowed sparse arrays while solving in the destination.
_sparse_triangular_alias(dest, ::FaerSparseMatrix) = nothing
function _sparse_triangular_alias(dest, source::FaerBorrowedSparseMatrix)
    y=dest isa FaerBorrowedArray ? parent(dest) : dest
    y isa AbstractArray && Base.mightalias(y, source) && throw(
        ArgumentError("sparse triangular destination must not alias the borrowed sparse matrix"),
    )
    return nothing
end
_sparse_triangular_owner(p::LinearAlgebra.Transpose) = (parent(p), Int32(1))
_sparse_triangular_owner(p::LinearAlgebra.Adjoint) = (parent(p), Int32(2))
# Destinations the native kernel writes into directly.
const _SparseTriangularNative=Union{FaerMatrix,FaerVector,FaerView}
_sparse_triangular_direct(x::_SparseTriangularNative, ::Type{P}) where {P} = eltype(x)===P
_sparse_triangular_direct(x::FaerBorrowedArray, ::Type{P}) where {P} =
    eltype(x)===P && _borrowed_writable(x)
_sparse_triangular_direct(x::StridedVecOrMat, ::Type{P}) where {P} =
    eltype(x)===P && _faerview_valid(x)
_sparse_triangular_direct(x, ::Type) = false

function _sparse_triangular_prepare(a, b, team)
    isnothing(team) || team isa FaerTeam || throw(ArgumentError("team must be a FaerTeam"))
    isnothing(team) || _handle(team)
    source, op=_sparse_triangular_owner(parent(a))
    _sparse_triangular_open(source)
    b isa Union{_SparseTriangularNative,FaerBorrowedArray} && _spectral_rhs_open(b)
    size(a, 1)==size(b, 1) ||
        throw(DimensionMismatch("sparse triangular solve dimensions must match"))
    P=promote_type(eltype(a), _sparse_input_type(eltype(b)))
    return source, op, P
end
function _sparse_triangular_query(
    source::FaerBorrowedSparseMatrix,
    a,
    op,
    ::Type{P},
) where {P}
    info=zeros(UInt, 2)
    GC.@preserve info _with_borrowed_sparse(source) do raw...
        _native_call(
            RustCrate.borrowed_sparse_triangular_requirements,
            raw...,
            _islower(a),
            _isunit(a),
            op,
            _tag(P),
            info,
        )
    end
    return (bytes = Int(info[1]), alignment = Int(info[2]))
end
function _sparse_triangular_query(source, a, op, ::Type{P}) where {P}
    info=zeros(UInt, 2)
    GC.@preserve info _native_call(
        RustCrate.sparse_triangular_requirements,
        source,
        _islower(a),
        _isunit(a),
        op,
        _tag(P),
        info,
    )
    return (bytes = Int(info[1]), alignment = Int(info[2]))
end
function _sparse_triangular_check_singular(source, a)
    _isunit(a) && return nothing
    i=Int(
        source isa FaerBorrowedSparseMatrix ?
        _with_borrowed_sparse((raw...) -> _native_call(RustCrate.borrowed_sparse_zero_diagonal, raw...), source) :
        _native_call(RustCrate.sparse_zero_diagonal, source),
    )
    iszero(i) || throw(LinearAlgebra.SingularException(i))
    return nothing
end
# Solve in place in `x`, which already holds the RHS in scalar type P.
function _sparse_triangular_native!(x, source::FaerBorrowedSparseMatrix, a, op, work)
    lower, unit=_islower(a), _isunit(a)
    code=_with_borrowed_sparse(source) do raw...
        if x isa _SparseTriangularNative
            p, d=_view_source(x)
            GC.@preserve d _native_call(
                RustCrate.borrowed_sparse_triangular_into,
                p,
                pointer(d),
                raw...,
                lower,
                unit,
                op,
                work,
            )
        else
            y=x isa FaerBorrowedArray ? parent(x) : x
            m=size(y, 1)
            n=ndims(y)==1 ? 1 : size(y, 2)
            rs=stride(y, 1)
            cs=ndims(y)==1 ? m*rs : stride(y, 2)
            GC.@preserve y _native_call(
                RustCrate.borrowed_sparse_triangular_raw,
                raw...,
                Ptr{UInt8}(pointer(y)),
                UInt(m),
                UInt(n),
                rs,
                cs,
                _tag(eltype(y)),
                lower,
                unit,
                op,
                work,
            )
        end
    end
    iszero(code) || throw(LinearAlgebra.SingularException(Int(code)))
    return x
end
function _sparse_triangular_native!(x, source, a, op, work)
    lower, unit=_islower(a), _isunit(a)
    code=if x isa _SparseTriangularNative
        p, d=_view_source(x)
        GC.@preserve d _native_call(
            RustCrate.sparse_triangular_into,
            p,
            pointer(d),
            source,
            lower,
            unit,
            op,
            work,
        )
    else
        y=x isa FaerBorrowedArray ? parent(x) : x
        m=size(y, 1)
        n=ndims(y)==1 ? 1 : size(y, 2)
        rs=stride(y, 1)
        cs=ndims(y)==1 ? m*rs : stride(y, 2)
        GC.@preserve y _native_call(
            RustCrate.sparse_triangular_raw,
            source,
            Ptr{UInt8}(pointer(y)),
            UInt(m),
            UInt(n),
            rs,
            cs,
            _tag(eltype(y)),
            lower,
            unit,
            op,
            work,
        )
    end
    iszero(code) || throw(LinearAlgebra.SingularException(Int(code)))
    return x
end
# Solves borrowing a conforming owner need no workspace. Without caller scratch
# they share one empty native workspace; every use holds `_matrix_lock`.
const _SPARSE_TRIANGULAR_EMPTY=Ref{Any}(nothing)
function _sparse_triangular_empty()
    work=_SPARSE_TRIANGULAR_EMPTY[]
    if isnothing(work) || !isopen(work)
        work=FaerScratch(0)
        _SPARSE_TRIANGULAR_EMPTY[]=work
    end
    return work
end
function _with_sparse_triangular_scratch(fun, req, scratch)
    isnothing(scratch) && iszero(req.bytes) && return fun(_sparse_triangular_empty())
    temporary=isnothing(scratch)
    work=temporary ? FaerScratch(req.bytes; alignment = req.alignment) : scratch
    work isa FaerScratch || throw(ArgumentError("scratch must be a FaerScratch"))
    try
        cap=scratch_capacity(work)
        cap.bytes>=req.bytes && cap.alignment>=req.alignment || throw(
            ArgumentError("insufficient sparse triangular scratch; call reserve!"),
        )
        return fun(work)
    finally
        temporary && close(work)
    end
end
# A fresh native RHS copy in the promoted type; it never aliases `b`.
function _sparse_triangular_copy(b::AbstractMatrix, ::Type{P}) where {P}
    b isa FaerMatrix && return eltype(b)===P ? copy(b) : FaerMatrix{P}(b)
    return FaerMatrix{P}(b)
end
_sparse_triangular_copy(b::_BorrowVector, ::Type{P}) where {P} =
    eltype(b)===P ? copy(b) : FaerVector{P}(b)
_sparse_triangular_copy(b::AbstractVector, ::Type{P}) where {P} = Vector{P}(b)

function _sparse_triangular_solve(
    a::_SparseTriangular,
    b::AbstractVecOrMat;
    scratch = nothing,
    team = nothing,
)
    lock(_matrix_lock) do
        source, op, P=_sparse_triangular_prepare(a, b, team)
        req=_sparse_triangular_query(source, a, op, P)
        _with_sparse_triangular_scratch(req, scratch) do work
            # Rust rejects a zero diagonal before writing the fresh copy.
            x=_sparse_triangular_copy(b, P)
            try
                return _sparse_triangular_native!(x, source, a, op, work)
            catch
                x isa Union{FaerMatrix,FaerVector} && close(x)
                rethrow()
            end
        end
    end
end
_sparse_triangular_vector(a, b; kwargs...) = _sparse_triangular_solve(a, b; kwargs...)

# Write op(A) \ b into dest. When dest holds the promoted scalar and is a native
# owner/view or an injective Julia strided array, the solve runs in dest itself;
# with dest === b nothing else is copied. Narrower or non-strided destinations
# stage one promoted native copy and convert on writeback. All validation,
# including the singularity check and scratch capacity, precedes the first write
# to dest: Rust checks the diagonal before solving, so only a copy into dest
# needs the separate check first.
function _sparse_triangular_write!(dest, a, b; scratch = nothing, team = nothing)
    lock(_matrix_lock) do
        source, op, P=_sparse_triangular_prepare(a, b, team)
        dest isa Union{_SparseTriangularNative,FaerBorrowedArray} && _spectral_rhs_open(dest)
        size(dest)==size(b) ||
            throw(DimensionMismatch("triangular output size must match RHS"))
        _sparse_triangular_alias(dest, source)
        direct=_sparse_triangular_direct(dest, P)
        direct && dest!==b && _sparse_triangular_check_singular(source, a)
        req=_sparse_triangular_query(source, a, op, P)
        _with_sparse_triangular_scratch(req, scratch) do work
            if direct
                dest===b || copyto!(dest, b)
                _sparse_triangular_native!(dest, source, a, op, work)
            else
                x=_sparse_triangular_copy(b, P)
                try
                    _sparse_triangular_native!(x, source, a, op, work)
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
            a::LinearAlgebra.$C{T,<:_SparseTriangularParent{T}},
            b::AbstractMatrix{<:Number};
            kwargs...,
        ) where {T<:_NativeScalar} = _sparse_triangular_solve(a, b; kwargs...)
        Base.:\(
            a::LinearAlgebra.$C{T,<:_SparseTriangularParent{T}},
            b::AbstractVector{<:Number};
            kwargs...,
        ) where {T<:_NativeScalar} = _sparse_triangular_vector(a, b; kwargs...)
        Base.:\(
            a::LinearAlgebra.$C{T,<:_SparseTriangularParent{T}},
            b::SparseArrays.AbstractCompressedVector{<:Number};
            kwargs...,
        ) where {T<:_NativeScalar} = _sparse_triangular_vector(a, b; kwargs...)
    end
end
"""
    ldiv!(A::SparseTriangular, B; scratch=nothing, team=nothing)
    ldiv!(Y, A::SparseTriangular, B; scratch=nothing, team=nothing)

Solve with a lower/upper/unit triangular wrapper of a `FaerSparseMatrix` (CSC or
CSR, optionally `transpose`/`adjoint`) or of a borrowed Julia CSC matrix
`faerview(S)` (optionally `transpose`/`adjoint`, read in place through the
per-call 0-based index buffers) in place. A destination sharing memory with a
borrowed `S` throws `ArgumentError`. When the destination is a
`FaerMatrix`/`FaerVector`, a native view, a [`faerview`](@ref) borrow or an
injective Julia strided array whose element type is
`promote_type(eltype(A), eltype(B))`, faer's
`sp_solve_{unit_}{lower,upper}_triangular_in_place` runs directly on that memory.
The sparse owner is borrowed as is when every column already has faer's layout
(only the selected half stored, diagonal first for lower / last for upper,
including unit wrappers); otherwise the triangle is rebuilt, and promoted
values are converted, in `scratch`. Query with
`scratch_requirements(ldiv!, A, B)`, create with `FaerScratch(ldiv!, A, B)` or
grow with `reserve!(work, ldiv!, A, B)`; without `scratch` a temporary of that
size is used. With caller scratch and `Y === B`, the native call allocates
nothing. `Y !== B` copies `B` into `Y` first; narrower or non-strided
destinations stage one promoted copy and convert on writeback. `team` is
validated and accepted for uniformity, but faer's sparse triangular kernels are
sequential. Zero diagonals throw `SingularException` and closed or undersized
resources throw before any write.
"""
function LinearAlgebra.ldiv!(
    a::_SparseTriangular,
    b::Union{AbstractVector,AbstractMatrix};
    kwargs...,
)
    return _sparse_triangular_write!(b, a, b; kwargs...)
end
function LinearAlgebra.ldiv!(
    dest::Union{AbstractVector,AbstractMatrix},
    a::_SparseTriangular,
    b::Union{AbstractVector,AbstractMatrix};
    kwargs...,
)
    return _sparse_triangular_write!(dest, a, b; kwargs...)
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
    )
        @eval Base.:\(
            a::LinearAlgebra.$C{T,<:_SparseTriangularParent{T}},
            b::$R{<:Number};
            kwargs...,
        ) where {T<:_NativeScalar} = _sparse_triangular_solve(a, b; kwargs...)
    end
end
