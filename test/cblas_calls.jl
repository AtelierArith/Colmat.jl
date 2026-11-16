using Test, Colmat, LinearAlgebra, Libdl

const CBLAS_271_NATIVE = Libdl.dlopen(Colmat.backend_library())
const CBLAS_271_TRAMPOLINE = Libdl.dlopen(BLAS.libblastrampoline)
const CBLAS_271_CALLS = Set{Tuple{String,Int,Symbol}}()
const CBLAS_271_CALL_SITES = Set{Tuple{String,Int,Symbol,String,Int,String}}()

function cblas271_pointer(name, ::Type{J}, route) where {J<:Union{Int32,Int64}}
    suffix = J===Int64 ? "64_" : ""
    expected = Libdl.dlsym(CBLAS_271_NATIVE, name*suffix)
    interface = J===Int64 ? :ilp64 : :lp64
    @test BLAS.lbt_get_forward(name, interface)==expected
    handle = if route===:dll
        CBLAS_271_NATIVE
    elseif route===:lbt_trampoline
        CBLAS_271_TRAMPOLINE
    else
        error("unknown CBLAS route")
    end
    return Libdl.dlsym(handle, name*suffix)
end

# Record after the concrete foreign call returns. Source hashes and the final
# successful full-suite gate will bind these observations to the tested code.
macro cblas271_call(name, integer, route, result_type, signature, arguments...)
    pointer = gensym(:entry)
    result = gensym(:result)
    call = Expr(
        :call, :ccall, pointer, esc(result_type), esc(signature), esc.(arguments)...
    )
    file = String(__source__.file)
    line = __source__.line
    return quote
        let name = String($(esc(name))), integer = $(esc(integer)), route = $(esc(route))
            local $pointer = cblas271_pointer(name, integer, route)
            local $result = $call
            local bits = 8*sizeof(integer)
            push!(CBLAS_271_CALLS, (name, bits, route))
            push!(
                CBLAS_271_CALL_SITES,
                (name, bits, route, $file, $line, string($(esc(signature)))),
            )
            $result
        end
    end
end

function cblas271_storage(values, inc)
    result=fill(eltype(values)(97), 1+(length(values)-1)*abs(inc))
    for i in eachindex(values)
        index=inc<0 ? 1+(length(values)-i)*(-inc) : 1+(i-1)*inc
        result[index]=values[i]
    end
    return result
end
function cblas271_indices(n, inc)
    return inc<0 ? ((1 + (n - 1) * (-inc)):inc:1) : (1:inc:(1 + (n - 1) * inc))
end
