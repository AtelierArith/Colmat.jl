"""
    bidiagonalize(A; compact=false, full=false, blocksize=0, scratch=nothing, team=nothing, par_threshold=FaerParThreshold())

Native faer reduction `A ≈ U * B * V'`, where B is a square `Bidiagonal` of
order min(size(A)...). Tall/square inputs give upper B; wide inputs give lower B.
The result owns native U/V with thin orthonormal columns by default; full=true
retains complete bases. Inputs are preserved; complex phases stay in B.
`compact=true` returns implicit [`FaerBidiagonalBasis`](@ref) U/V with the same
thin/full shapes. Products and saved-factor SVD apply packed reflectors without
forming square bases. Zero blocksize selects faer's recommendation.
Use `scratch_requirements`, `FaerScratch(bidiagonalize,A)` and `reserve!` for the
canonical work matrix, compact factors and declared reduction/U/V workspace.
The actual scoped team token is used by all stages. Conversion, returned owners
and staged bang writeback allocate separately. In compact mode the returned
owner retains the packed matrix and both factors; scratch covers only the
reduction kernel. Use the same `compact` option in queries and computation.
`par_threshold` selects the native reduction's derived/fixed/never policy;
`FaerParThreshold(:never)` also makes U/V formation sequential, while validating
any supplied team.
"""
struct FaerBidiagonalization{T,M<:Union{FaerMatrix{T},FaerBidiagonalBasis{T}}} <:
       LinearAlgebra.Factorization{T}
    U::M
    B::LinearAlgebra.Bidiagonal{T,Vector{T}}
    V::M
end
Base.size(f::FaerBidiagonalization) = (size(f.U, 1), size(f.V, 1))
Base.size(f::FaerBidiagonalization, d::Integer) =
    d<1 ? throw(ArgumentError("dimension must be positive")) : d<=2 ? size(f)[d] : 1
Base.isopen(f::FaerBidiagonalization) = isopen(f.U)&&isopen(f.V)
function Base.close(f::FaerBidiagonalization)
    lock(_matrix_lock) do ;
        close(f.U)
        close(f.V)
    end
    return nothing
end
function Base.copy(f::FaerBidiagonalization)
    lock(_matrix_lock) do
        _handle(f.U)
        _handle(f.V)
        if f.U isa FaerBidiagonalBasis && _handle(f.U)===_handle(f.V)
            _basis_validate(f.U)
            _basis_validate(f.V)
            h=_native_call(RustCrate.OwnedCompactBidiagonal, f.U)
            u=_compact_basis(
                eltype(f),
                h,
                f.U.left,
                f.U.cols;
                transposed = f.U.transposed,
                conjugated = f.U.conjugated,
            )
            v=_compact_basis(
                eltype(f),
                h,
                f.V.left,
                f.V.cols;
                transposed = f.V.transposed,
                conjugated = f.V.conjugated,
            )
            return FaerBidiagonalization(u, copy(f.B), v)
        end
        return FaerBidiagonalization(copy(f.U), copy(f.B), copy(f.V))
    end
end
Base.deepcopy_internal(f::FaerBidiagonalization, d::IdDict) = get!(() -> copy(f), d, f)
function _bidiagonal_bases(f)
    _handle(f.U)
    _handle(f.V)
    m, n=size(f)
    k=size(f.B, 1)
    f.U isa FaerBidiagonalBasis &&
        return _basis_copy(f.U; cols = k), _basis_copy(f.V; cols = k)
    return copy(view(f.U, 1:m, 1:k)), copy(view(f.V, 1:n, 1:k))
end
function FaerMatrix(f::FaerBidiagonalization)
    lock(_matrix_lock) do
        U, V=_bidiagonal_bases(f)
        return U*FaerMatrix(Matrix(f.B))*V'
    end
end
Base.Matrix(f::FaerBidiagonalization) = Matrix(FaerMatrix(f))
Base.Array(f::FaerBidiagonalization) = Matrix(f)
function bidiagonalize(
    a::AbstractMatrix{<:Number};
    full::Bool = false,
    blocksize::Integer = 0,
    scratch = nothing,
    team = nothing,
    par_threshold = FaerParThreshold(),
    compact::Bool = false,
)
    lock(_matrix_lock) do
        _iterative_resource_check(scratch, team)
        policy=_par_threshold(par_threshold)
        m, n, T, bs=_bidiagonalize_plan(a, blocksize)
        k=min(m, n)
        req=_bidiagonalize_requirement(m, n, T, full, bs, team, policy; compact)
        work=isnothing(scratch) ? FaerScratch(req.bytes; alignment = req.alignment) :
             scratch
        try
            cap=scratch_capacity(work)
            cap.bytes>=req.bytes && cap.alignment>=req.alignment || throw(
                ArgumentError(
                    "bidiagonal reduction scratch capacity/alignment is insufficient; call reserve!",
                ),
            )
            data=Matrix{T}(a)
            data isa Matrix{T} || throw(
                ArgumentError("bidiagonal conversion returned incompatible scalar storage"),
            )
            size(data)==(m, n) ||
                throw(DimensionMismatch("bidiagonal conversion changed matrix shape"))
            all(isfinite, data) ||
                throw(ArgumentError("bidiagonalization input must be finite"))
            if compact
                d=Vector{T}(undef, k)
                e=Vector{T}(undef, max(0, k-1))
                h=GC.@preserve data d e begin
                    args=(
                        work,
                        Ptr{UInt8}(pointer(data)),
                        UInt(m),
                        UInt(n),
                        Ptr{UInt8}(pointer(d)),
                        Ptr{UInt8}(pointer(e)),
                        _tag(T),
                        UInt(bs),
                        policy.value,
                    )
                    isnothing(team) ?
                    _native_call(RustCrate.OwnedCompactBidiagonal, args...) :
                    _native_call(RustCrate.OwnedCompactBidiagonal, args..., team)
                end
                return FaerBidiagonalization(
                    _compact_basis(T, h, true, full ? m : k),
                    LinearAlgebra.Bidiagonal(d, e, m<n ? :L : :U),
                    _compact_basis(T, h, false, full ? n : k),
                )
            end
            U=Matrix{T}(undef, m, full ? m : k)
            V=Matrix{T}(undef, n, full ? n : k)
            d=Vector{T}(undef, k)
            e=Vector{T}(undef, max(0, k-1))
            GC.@preserve data U V d e begin
                args=(
                    work,
                    Ptr{UInt8}(pointer(data)),
                    UInt(m),
                    UInt(n),
                    Ptr{UInt8}(pointer(U)),
                    Ptr{UInt8}(pointer(V)),
                    Ptr{UInt8}(pointer(d)),
                    Ptr{UInt8}(pointer(e)),
                    _tag(T),
                    full,
                    UInt(bs),
                    policy.value,
                )
                isnothing(team) ?
                _native_call(RustCrate.bidiagonalize_tuned_workspace, args...) :
                _native_call(RustCrate.bidiagonalize_tuned_workspace_team, args..., team)
            end
            all(isfinite, U)&&all(isfinite, V)&&all(isfinite, d)&&all(isfinite, e) ||
                throw(ArgumentError("bidiagonalization overflowed the working type"))
            return FaerBidiagonalization(
                FaerMatrix(U),
                LinearAlgebra.Bidiagonal(d, e, m<n ? :L : :U),
                FaerMatrix(V),
            )
        finally
            isnothing(scratch) && close(work)
        end
    end
end
"""Stage native bidiagonalization, then replace A by the rectangular embedding of B."""
function bidiagonalize!(a::FaerMatrix; kwargs...)
    lock(_matrix_lock) do
        f=bidiagonalize(a; kwargs...)
        try
            m, n=size(a)
            k=size(f.B, 1)
            buffer=zeros(eltype(a), m, n)
            buffer[1:k, 1:k]=Matrix(f.B)
            copyto!(a, buffer)
            return f
        catch
            close(f)
            rethrow()
        end
    end
end
"""
    bidiagonal_svd(d, e; uplo=:U, recursion_threshold=128, scratch=nothing, team=nothing)

Call faer's square bidiagonal SVD directly. d and e are the diagonal and selected
offdiagonal (`:U` superdiagonal, `:L` subdiagonal). All four native scalar types
and promotion are supported. Inputs remain unchanged. Returns a Julia SVD with
native U/Vt and nonnegative descending real S. Use `scratch_requirements`,
`FaerScratch(bidiagonal_svd,d,e)` and `reserve!` for the coefficient copies and
declared native workspace; U/V are written directly into preserved staging.
Actual team shape/token is used for the query/kernel. Julia conversion, result
owners and saved-factor back-transform products allocate separately. The native
kernel still computes dense factors when the caller only needs singular values.
"""
function bidiagonal_svd(
    d::AbstractVector{<:Number},
    e::AbstractVector{<:Number};
    uplo = :U,
    recursion_threshold::Integer = 128,
    scratch = nothing,
    team = nothing,
)
    lock(_matrix_lock) do
        _iterative_resource_check(scratch, team)
        n, T, threshold=_bidiagonal_svd_plan(d, e, uplo, recursion_threshold)
        R=_realtype(T)
        req=_bidiagonal_svd_requirement(n, T, threshold, team)
        work=isnothing(scratch) ? FaerScratch(req.bytes; alignment = req.alignment) :
             scratch
        try
            cap=scratch_capacity(work)
            cap.bytes>=req.bytes && cap.alignment>=req.alignment || throw(
                ArgumentError(
                    "bidiagonal SVD scratch capacity/alignment is insufficient; call reserve!",
                ),
            )
            diag=Vector{T}(d)
            off=Vector{T}(e)
            diag isa Vector{T} && off isa Vector{T} || throw(
                ArgumentError("bidiagonal conversion returned incompatible scalar storage"),
            )
            length(diag)==n && length(off)==max(0, n-1) || throw(
                DimensionMismatch("bidiagonal conversion changed coefficient lengths"),
            )
            all(isfinite, diag)&&all(isfinite, off) ||
                throw(ArgumentError("bidiagonal coefficients must be finite"))
            U=Matrix{T}(undef, n, n)
            V=Matrix{T}(undef, n, n)
            values=Vector{R}(undef, n)
            GC.@preserve diag off U V values begin
                args=(
                    work,
                    Ptr{UInt8}(pointer(diag)),
                    Ptr{UInt8}(pointer(off)),
                    UInt(n),
                    Ptr{UInt8}(pointer(values)),
                    Ptr{UInt8}(pointer(U)),
                    Ptr{UInt8}(pointer(V)),
                    _tag(T),
                    uplo in (:U, 'U'),
                    UInt(threshold),
                )
                isnothing(team) ?
                _native_call(RustCrate.bidiagonal_svd_workspace, args...) :
                _native_call(RustCrate.bidiagonal_svd_workspace_team, args..., team)
            end
            all(isfinite, U)&&all(isfinite, V)&&all(isfinite, values) ||
                throw(ArgumentError("bidiagonal SVD overflowed the working type"))
            return LinearAlgebra.SVD(FaerMatrix(U), values, FaerMatrix(copy(adjoint(V))))
        finally
            isnothing(scratch) && close(work)
        end
    end
end
function LinearAlgebra.svd(
    f::FaerBidiagonalization;
    full::Bool = false,
    scratch = nothing,
    team = nothing,
    kwargs...,
)
    lock(_matrix_lock) do
        _iterative_resource_check(scratch, team)
        _bidiagonal_factor_validate(f; full)
        m, n=size(f)
        k=size(f.B, 1)
        T=eltype(f)
        f.U isa FaerBidiagonalBasis &&
            return _compact_bidiagonal_svd(f; full, scratch, team, kwargs...)
        core=bidiagonal_svd(f.B.dv, f.B.ev; uplo = f.B.uplo, scratch, team, kwargs...)
        try
            if full
                left=Matrix{T}(LinearAlgebra.I, m, m)
                left[1:k, 1:k]=Matrix(core.U)
                right=Matrix{T}(LinearAlgebra.I, n, n)
                right[1:k, 1:k]=Matrix(core.Vt)
                l, r=FaerMatrix(left), FaerMatrix(right)
                try
                    return LinearAlgebra.SVD(
                        _bidiagonal_product(team, f.U, l),
                        core.S,
                        _bidiagonal_product(team, r, f.V'),
                    )
                finally
                    close(l)
                    close(r)
                end
            end
            U, V=_bidiagonal_bases(f)
            try
                return LinearAlgebra.SVD(
                    _bidiagonal_product(team, U, core.U),
                    core.S,
                    _bidiagonal_product(team, core.Vt, V'),
                )
            finally
                close(U)
                close(V)
            end
        finally
            close(core.U)
            close(core.Vt)
        end
    end
end
function LinearAlgebra.svdvals(
    f::FaerBidiagonalization;
    scratch = nothing,
    team = nothing,
    kwargs...,
)
    lock(_matrix_lock) do
        _iterative_resource_check(scratch, team)
        _bidiagonal_factor_validate(f)
        return _bidiagonal_values(f.B.dv, f.B.ev; uplo = f.B.uplo, scratch, team, kwargs...)
    end
end
const _NativeBidiagonal=LinearAlgebra.Bidiagonal{
    T,
    V,
} where {T<:_NativeScalar,V<:FaerVector{T}}
LinearAlgebra.svd(a::_NativeBidiagonal; full::Bool = false, kwargs...) =
    bidiagonal_svd(a.dv, a.ev; uplo = a.uplo, kwargs...)
LinearAlgebra.svdvals(a::_NativeBidiagonal; kwargs...) =
    _bidiagonal_values(a.dv, a.ev; uplo = a.uplo, kwargs...)
