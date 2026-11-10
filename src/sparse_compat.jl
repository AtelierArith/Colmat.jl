# Intersections introduced when SparseArrays loads its triangular/vector and
# CHOLMOD copy methods. Keep our explicit native solve/copy paths unambiguous.
for C in (:LowerTriangular, :UpperTriangular, :UnitLowerTriangular, :UnitUpperTriangular)
    @eval begin
        Base.:\(
            a::LinearAlgebra.$C{Float64,<:FaerMatrix{Float64}},
            b::SparseArrays.AbstractCompressedVector{<:Real},
        ) = a \ Vector(b)
        Base.:\(
            a::LinearAlgebra.$C{T,<:FaerMatrix{T}},
            b::SparseArrays.AbstractCompressedVector{<:Number},
        ) where {T<:_ExtraScalar} = a \ Vector(b)
    end
end
function Base.copyto!(v::FaerMatrixView, src::SparseArrays.AbstractSparseMatrixCSC)
    return _copyto_faer_view!(v, src)
end
Base.copyto!(v::FaerView, src::SparseArrays.CHOLMOD.Dense) = _copyto_faer_view!(v, src)
function Base.copyto!(
    v::FaerView{T}, src::SparseArrays.CHOLMOD.Dense{T}
) where {T<:_NativeScalar}
    return _copyto_faer_view!(v, src)
end
function Base.copyto!(
    v::FaerMatrixView{T}, src::SparseArrays.CHOLMOD.Dense{T}
) where {T<:_NativeScalar}
    return _copyto_faer_view!(v, src)
end
function Base.copyto!(
    dest::Union{FaerMatrix,FaerRow}, src::SparseArrays.AbstractSparseMatrixCSC
)
    return _copyto_native_owner!(dest, src)
end
function Base.copyto!(dest::Union{FaerMatrix,FaerRow}, src::SparseArrays.CHOLMOD.Dense)
    return _copyto_native_owner!(dest, src)
end
function Base.copyto!(
    dest::Union{FaerMatrix{T},FaerRow{T}}, src::SparseArrays.CHOLMOD.Dense{T}
) where {T<:_NativeScalar}
    return _copyto_native_owner!(dest, src)
end
