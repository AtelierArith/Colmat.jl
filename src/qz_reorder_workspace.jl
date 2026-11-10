"""
    scratch_requirements(ordschur, F::GeneralizedSchur{T,<:FaerMatrix})
    scratch_requirements(ordschur!, F::GeneralizedSchur{T,<:FaerMatrix})

Return `(bytes, alignment)` for saved-factor generalized Schur reordering.
Includes selection flags, native block-move scratch and real homogeneous-value
extraction. Independent work factors and Julia output/conversion vectors allocate
outside this workspace. Both functions have the same requirement.
"""
function scratch_requirements(::_QzOrderFunction, f::_NativeQZ)
    lock(_matrix_lock) do
        n = _check_pair_factor(f)
        info = zeros(UInt, 2)
        GC.@preserve info _native_call(
            RustCrate.qz_reorder_requirements, UInt(n), _tag(eltype(f)), info
        )
        return (bytes=Int(info[1]), alignment=Int(info[2]))
    end
end
function FaerScratch(fun::_QzOrderFunction, f::_NativeQZ)
    req = scratch_requirements(fun, f)
    return FaerScratch(req.bytes; alignment=req.alignment)
end
function reserve!(s::FaerScratch, fun::_QzOrderFunction, f::_NativeQZ)
    req = scratch_requirements(fun, f)
    return reserve!(s, req.bytes; alignment=req.alignment)
end
