# The numerical test resolves and verifies the named export before this wrapper.
# Qualification alone records nothing: the traced ccall must actually return,
# and its pointer must match the selected name, ABI and route.
function lapacke288_selected_entry(f, name, ::Type{J}, route) where {J<:Union{Int32,Int64}}
    if isdefined(@__MODULE__, :ABIEvidence)
        return ABIEvidence.with_entry(f, String(name), 8sizeof(J), string(route))
    end
    return f()
end
