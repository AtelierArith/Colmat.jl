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

cblas271_value(::Type{T}, i, j, shift) where {T<:Real} = T((i+2j+shift)/7)
function cblas271_value(::Type{T}, i, j, shift) where {T<:Complex}
    return T((i+2j+shift)/7, (2i-j+shift)/11)
end
cblas271_matrix_index(order, ld, i, j) = order==101 ? (i-1)*ld+j : (j-1)*ld+i
function cblas271_pack(matrix, order, ld)
    rows, cols=size(matrix)
    result=fill(eltype(matrix)(97), ld*(order==101 ? rows : cols)+3)
    for j in 1:cols, i in 1:rows
        result[cblas271_matrix_index(order, ld, i, j)]=matrix[i, j]
    end
    return result
end
cblas271_op(matrix, t) =
    if t==111
        matrix
    elseif t==112
        transpose(matrix)
    elseif t==113
        adjoint(matrix)
    else
        conj.(matrix)
    end

cblas271_poison_diagonal(value::Real) = value
cblas271_poison_diagonal(value::Complex{R}) where {R} = Complex{R}(real(value), R(NaN))
function cblas271_symmetric_matrix(::Type{T}, n, u, k, hermitian) where {T}
    return T[
        if abs(i-j)>k
            zero(T)
        elseif i==j
            T(2+i/7)
        elseif (u==121&&i<j)||(u==122&&i>j)
            cblas271_value(T, i, j, 1)
        else
            value=cblas271_value(T, j, i, 1)
            hermitian ? conj(value) : value
        end for i in 1:n, j in 1:n
    ]
end
function cblas271_triangle_storage(
    matrix, order, u, storage, k; hermitian=false, unit=false
)
    n=size(matrix, 1)
    T=eltype(matrix)
    stored(i, j) =
        if unit&&i==j
            T(NaN)
        elseif hermitian&&i==j
            cblas271_poison_diagonal(matrix[i, j])
        else
            matrix[i, j]
        end
    selected(i, j) = ((u==121&&i<=j)||(u==122&&i>=j))&&(storage!==:band||abs(i-j)<=k)
    if storage===:packed
        data=T[]
        if order==101
            for i in 1:n, j in 1:n
                selected(i, j)&&push!(data, stored(i, j))
            end
        else
            for j in 1:n, i in 1:n
                selected(i, j)&&push!(data, stored(i, j))
            end
        end
        append!(data, fill(T(97), 3))
        return data, 0
    end
    ld=storage===:dense ? n+2 : k+3
    data=fill(T(NaN), ld*n+3)
    for i in 1:n, j in 1:n
        if selected(i, j)
            position=if storage===:dense
                cblas271_matrix_index(order, ld, i, j)
            elseif order==101
                (i-1)*ld+(u==121 ? j-i : k+j-i)+1
            else
                (j-1)*ld+(u==121 ? k+i-j : i-j)+1
            end
            data[position]=stored(i, j)
        end
    end
    return data, ld
end
