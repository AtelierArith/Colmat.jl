"""
    FaerSparseProductPlan(A, B)

Reusable plan for the sparse product `A * B`. Construction runs faer's symbolic
phase (`sparse_sparse_matmul_symbolic`) once: it computes the result pattern
and the work partition, and keeps independent copies of the stored patterns of
`A` and `B` (stored zeros included). Values are not read.

`A` and `B` are `FaerSparseMatrix` operands (`:csc` or `:csr`; a CSR plan is
the column-major plan of `Bᵀ Aᵀ`), or borrowed Julia CSC matrices `faerview(S)`
(`FaerBorrowedSparseMatrix`), read in place. The plan follows `A`'s storage
format. A native `B` of the other format is transposed once at planning time:
the plan keeps that transposed pattern and a value permutation, and each
numeric call gathers `B`'s values through it into the caller scratch (no
conversion per call). Borrowed and native operands may be mixed; both are read
in place (a native CSR operand through its column-major transpose).

`FaerSparseMatrix{T}(plan)` allocates a zero-valued destination with the
planned pattern and format; `A * B` has the same pattern. A borrowed Julia CSC
matrix `faerview(C)` with exactly that pattern is also accepted as destination
(values written in place). Reuse the plan with
`mul!(C, plan, A, B, α=1, β=0; scratch, team)` while the patterns stay fixed.
`size(plan)`, `nnz(plan)`, `plan.format`, `copy`, `close` and `isopen` are
supported; a copy reruns the symbolic phase from the saved patterns.
"""
mutable struct FaerSparseProductPlan{H}
    handle::Union{Nothing,H}
    dims::Tuple{Int,Int}
    stored::Int
    format::Symbol
    bformat::Symbol
    FaerSparseProductPlan{H}(handle::H, dims, stored, format, bformat) where {H} =
        new{H}(handle, dims, stored, format, bformat)
end
function _handle(p::FaerSparseProductPlan)
    h=p.handle
    isnothing(h) && throw(ArgumentError("sparse product plan has been closed"))
    return h
end
_native_argument(p::FaerSparseProductPlan) = _handle(p)
_is_native_owner(::FaerSparseProductPlan) = true

const _PlanOperand=Union{FaerSparseMatrix,FaerBorrowedSparseMatrix}

function _sparse_plan_result(f, args...)
    lock(_matrix_lock) do
        h=RustCrate.SparseProductPlan()
        _native_call(f, h, args...)
        info=zeros(UInt, 5)
        GC.@preserve info _native_call(RustCrate.sparse_product_plan_info, h, pointer(info))
        return FaerSparseProductPlan{typeof(h)}(
            h,
            (Int(info[1]), Int(info[2])),
            Int(info[3]),
            info[4]==0 ? :csc : :csr,
            info[5]==0 ? :csc : :csr,
        )
    end
end
function _sparse_plan_dims(a, b)
    size(a, 2)==size(b, 1) ||
        throw(DimensionMismatch("sparse product dimensions must match"))
end
function FaerSparseProductPlan(a::FaerSparseMatrix, b::FaerSparseMatrix)
    lock(_matrix_lock) do
        _handle(a)
        _handle(b)
        _sparse_plan_dims(a, b)
        return _sparse_plan_result(RustCrate.sparse_product_plan_build, a, b)
    end
end
_raw_format(x::FaerSparseMatrix) = x.format
_raw_format(::FaerBorrowedSparseMatrix) = :csc
function _raw_sparse_plan(a, b)
    lock(_matrix_lock) do
        for x in (a, b)
            x isa FaerSparseMatrix && _handle(x)
        end
        _sparse_plan_dims(a, b)
        _with_raw_sparse(a) do m, n, p, i, _, k
            _with_raw_sparse(b) do m2, n2, p2, i2, _, k2
                _sparse_plan_result(
                    RustCrate.sparse_product_plan_build_borrowed,
                    m,
                    n,
                    p,
                    i,
                    k,
                    m2,
                    n2,
                    p2,
                    i2,
                    k2,
                    _raw_format(a)===:csr,
                    _raw_format(b)===:csr,
                )
            end
        end
    end
end
FaerSparseProductPlan(a::FaerBorrowedSparseMatrix, b::FaerBorrowedSparseMatrix) =
    _raw_sparse_plan(a, b)
FaerSparseProductPlan(a::FaerBorrowedSparseMatrix, b::FaerSparseMatrix) =
    _raw_sparse_plan(a, b)
FaerSparseProductPlan(a::FaerSparseMatrix, b::FaerBorrowedSparseMatrix) =
    _raw_sparse_plan(a, b)

Base.size(p::FaerSparseProductPlan) = p.dims
Base.size(p::FaerSparseProductPlan, i::Integer) = i<=2 ? p.dims[i] : 1
SparseArrays.nnz(p::FaerSparseProductPlan) = p.stored
Base.isopen(p::FaerSparseProductPlan) = lock(() -> !isnothing(p.handle), _matrix_lock)
function Base.close(p::FaerSparseProductPlan)
    lock(_matrix_lock) do
        if !isnothing(p.handle)
            _native_call(RustCrate.sparse_product_plan_close, p)
            p.handle=nothing
        end
    end
    return nothing
end
function Base.copy(p::FaerSparseProductPlan)
    lock(_matrix_lock) do
        _handle(p)
        return _sparse_plan_result(RustCrate.sparse_product_plan_clone_from, p)
    end
end
Base.deepcopy_internal(p::FaerSparseProductPlan, d::IdDict) = get!(() -> copy(p), d, p)
function Base.show(io::IO, p::FaerSparseProductPlan)
    isopen(p) || return print(io, "FaerSparseProductPlan(closed)")
    print(io, "FaerSparseProductPlan(", p.dims[1], "×", p.dims[2], ", nnz=", p.stored)
    print(io, ", format=:", p.format, ")")
end

"""
    FaerSparseMatrix{T}(plan::FaerSparseProductPlan)

A zero-valued sparse matrix of scalar `T` with the planned result pattern and
storage format: a destination for `mul!(C, plan, A, B)`.
"""
function FaerSparseMatrix{T}(p::FaerSparseProductPlan) where {T<:_NativeScalar}
    lock(_matrix_lock) do
        _handle(p)
        return _sparse_result(
            T,
            p.dims,
            p.format,
            RustCrate.sparse_from_product_plan,
            p,
            _tag(T),
        )
    end
end

function _plan_team(team)
    isnothing(team) || team isa FaerTeam || throw(ArgumentError("team must be a FaerTeam"))
    !isnothing(team) && _handle(team)
    return team
end
function _plan_requirements(p::FaerSparseProductPlan, ::Type{T}, team) where {T}
    lock(_matrix_lock) do
        _handle(p)
        _plan_team(team)
        info=zeros(UInt, 2)
        GC.@preserve info begin
            isnothing(team) ?
            _native_call(
                RustCrate.sparse_product_plan_requirements,
                p,
                _tag(T),
                pointer(info),
            ) :
            _native_call(
                RustCrate.sparse_product_plan_team_requirements,
                team,
                p,
                _tag(T),
                pointer(info),
            )
        end
        return (bytes = Int(info[1]), alignment = Int(info[2]))
    end
end

"""
    scratch_requirements(mul!, C, plan::FaerSparseProductPlan, A, B; team=nothing)

Bytes and alignment of the caller workspace for the numeric phase of
`mul!(C, plan, A, B)`: one dense work column of `eltype(C)` per thread
(`faer::sparse_sparse_matmul_numeric_scratch`), plus `nnz(B)` values of
`eltype(C)` when `B`'s storage format differs from `A`'s. It depends on the plan, the
scalar type and the team, not on the values. `FaerScratch(mul!, C, plan, A, B;
team)` and `reserve!(scratch, mul!, C, plan, A, B; team)` size a workspace.
"""
function scratch_requirements(
    ::typeof(LinearAlgebra.mul!),
    c::_PlanOperand,
    p::FaerSparseProductPlan,
    a::_PlanOperand,
    b::_PlanOperand;
    team = nothing,
)
    return _plan_requirements(p, eltype(c), team)
end
FaerScratch(
    fun::typeof(LinearAlgebra.mul!),
    c::_PlanOperand,
    p::FaerSparseProductPlan,
    a::_PlanOperand,
    b::_PlanOperand;
    team = nothing,
) = _new_operator_workspace(fun, c, p, a, b; team)
reserve!(
    s::FaerScratch,
    fun::typeof(LinearAlgebra.mul!),
    c::_PlanOperand,
    p::FaerSparseProductPlan,
    a::_PlanOperand,
    b::_PlanOperand;
    team = nothing,
) = _reserve_operator_workspace(s, fun, c, p, a, b; team)

const _PLAN_ERRORS=(
    "left operand pattern does not match the sparse product plan",
    "right operand pattern does not match the sparse product plan",
    "destination pattern does not match the sparse product plan; allocate it with FaerSparseMatrix{T}(plan)",
    "operand and destination storage formats must match the sparse product plan",
    "sparse product plan scalar types must match",
    "sparse product scratch capacity/alignment is insufficient; call reserve!(scratch, mul!, C, plan, A, B; team)",
)
_plan_operand(a::FaerSparseMatrix, ::Type{T}) where {T} = _sparse_coerce(a, T)
_plan_operand(a::FaerBorrowedSparseMatrix, ::Type{T}) where {T} =
    _borrowed_sparse_promote(a, T)
_plan_values(a::FaerBorrowedSparseMatrix) = SparseArrays.nonzeros(a.parent)
_plan_values(::FaerSparseMatrix) = nothing

function _plan_mul!(c::_PlanOperand, p, a, b, alpha, beta, scratch, team)
    lock(_matrix_lock) do
        T=eltype(c)
        c isa FaerSparseMatrix && _handle(c)
        _handle(p)
        a isa FaerSparseMatrix && _handle(a)
        b isa FaerSparseMatrix && _handle(b)
        _plan_team(team)
        size(a, 2)==size(b, 1) && size(c)==(size(a, 1), size(b, 2))==p.dims ||
            throw(DimensionMismatch("product, plan and destination dimensions must match"))
        P=promote_type(eltype(a), eltype(b))
        promote_type(P, T)===T || throw(
            ArgumentError("destination scalar $T cannot hold the operand scalar $P"),
        )
        alias=ArgumentError("the destination of a planned sparse product must not alias an operand")
        (c===a || c===b) && throw(alias)
        # Any borrowed matrix reads every matrix through raw column-major parts
        # in place. Those carry no format, so formats are checked here.
        raw=any(x -> x isa FaerBorrowedSparseMatrix, (c, a, b))
        raw &&
            (_raw_format(a)!==p.format || _raw_format(b)!==p.bformat) &&
            throw(ArgumentError(_PLAN_ERRORS[4]))
        aa=_plan_operand(a, T)
        bb=_plan_operand(b, T)
        if c isa FaerBorrowedSparseMatrix
            dst=_plan_values(c)
            for x in (aa, bb)
                v=_plan_values(x)
                isnothing(v) || !Base.mightalias(dst, v) || throw(alias)
            end
        end
        temp=isnothing(scratch)
        temp || scratch isa FaerScratch || throw(ArgumentError("scratch must be a FaerScratch"))
        work=if temp
            req=_plan_requirements(p, T, team)
            FaerScratch(req.bytes; alignment = req.alignment)
        else
            _handle(scratch)
            scratch
        end
        try
            x=Ref{T}(alpha)
            y=Ref{T}(beta)
            accumulate=!iszero(beta)
            code=GC.@preserve x y begin
                xp=Ptr{UInt8}(Base.unsafe_convert(Ptr{T}, x))
                yp=Ptr{UInt8}(Base.unsafe_convert(Ptr{T}, y))
                if !raw
                    args=(p, aa, bb, xp, yp, accumulate, work)
                    _native_call(RustCrate.sparse_plan_mul, c, args..., team)
                else
                    _with_raw_sparse(aa) do ra...
                        _with_raw_sparse(bb) do rb...
                            tail=(_tag(T), xp, yp, accumulate, work)
                            if c isa FaerSparseMatrix
                                _native_call(RustCrate.sparse_plan_mul_borrowed, c, p, ra..., rb..., tail..., team)
                            else
                                _with_raw_sparse(c) do rc...
                                    _native_call(RustCrate.sparse_plan_mul_into_borrowed, p, rc..., ra..., rb..., tail..., team)
                                end
                            end
                        end
                    end
                end
            end
            code==0 || throw(ArgumentError(_PLAN_ERRORS[Int(code)]))
        finally
            temp && close(work)
        end
        return c
    end
end
"""
    mul!(C, plan::FaerSparseProductPlan, A, B, α=1, β=0; scratch=nothing, team=nothing)

Compute `C = α*A*B + β*C` with faer's numeric phase
(`sparse_sparse_matmul_numeric`), reusing the symbolic analysis of `plan`. The
stored patterns of `A` and `B` (and their storage formats) must equal those the
plan was built from, and `C` must have exactly the planned result pattern
(`FaerSparseMatrix{T}(plan)` or `A*B`); otherwise `ArgumentError` is thrown
before any value of `C` is written. Values of `A`/`B` may change freely between
calls. `C`'s pattern, nnz, `epoch` and live `nonzeros(C)` views are retained;
`β == 0` overwrites, otherwise `C` is scaled by `β` and accumulated.

`C` is a `FaerSparseMatrix` or a borrowed Julia CSC destination `faerview(S)`:
then `nonzeros(S)` is written in place (same vector, same pattern) and `S` must
not share stored values with a borrowed operand (`ArgumentError`). `A`/`B` may
mix borrowed and native operands; all of them are read in place.

`eltype(C)` must hold `promote_type(eltype(A), eltype(B))`; an operand of
another scalar type is converted to `eltype(C)` (allocating a copy with the same
pattern). `C` must not be
`A` or `B`. With `scratch` (see `scratch_requirements(mul!, C, plan, A, B; team)`)
and operands already of `eltype(C)` no native buffer is allocated; without it a
temporary workspace is allocated and released. A `FaerTeam` supplies parallel
execution (faer's cost model may still run small products sequentially).
"""
LinearAlgebra.mul!(
    c::_PlanOperand,
    p::FaerSparseProductPlan,
    a::_PlanOperand,
    b::_PlanOperand,
    alpha::Number = true,
    beta::Number = false;
    scratch = nothing,
    team = nothing,
) = _plan_mul!(c, p, a, b, alpha, beta, scratch, team)
