# Native operators retain only integer callback identities. Julia owns all
# functions and scopes their dispatcher around every native call that uses them.
_operator_callback_owners(op::FaerLinearOperator) = getfield(op, :callbacks)
const _operator_callback_id = Ref{UInt}(0)
const _operator_callback_context_key = :__colmat_operator_callback_context
mutable struct _OperatorCallbackContext
    owners::Dict{UInt,Any}
    exception::Any
    boundary_exception::Any
end
function _merge_operator_callbacks(a, b)
    left=_operator_callback_owners(a)
    right=_operator_callback_owners(b)
    isnothing(left) && return right
    isnothing(right) && return left
    return merge(left, right)
end
function _operator_callback_dispatch(
    id::UInt,
    tag::Int32,
    adj::Int32,
    m::UInt,
    n::UInt,
    k::UInt,
    x::Ptr{UInt8},
    rs::Int,
    cs::Int,
    y::Ptr{UInt8},
    ys::Int,
    yc::Int,
)::Cint
    context=nothing
    try
        context=get(task_local_storage(), _operator_callback_context_key, nothing)
        isnothing(context) && return Cint(0)
        owner=context.owners[id]
        T=tag==0 ? Float32 :
          tag==1 ? Float64 :
          tag==2 ? ComplexF32 :
          tag==3 ? ComplexF64 : throw(ArgumentError("invalid callback scalar tag"))
        adj in (0, 1) || throw(ArgumentError("invalid callback orientation"))
        _operator_callback_apply!(T, owner, adj, m, n, k, x, rs, cs, y, ys, yc)
        return Cint(1)
    catch e
        if !isnothing(context) && isnothing(context.exception)
            context.exception=Some(e)
        end
        # No Julia exception may unwind through Rust's C ABI stack.
        return Cint(0)
    end
end

# Specialize once per scalar/callable pair; buffer loops contain no runtime
# scalar-tag dispatch and execute before/after the single user action call.
function _operator_callback_apply!(
    ::Type{T},
    owner,
    adj,
    m,
    n,
    k,
    x,
    rs,
    cs,
    y,
    ys,
    yc,
) where {T}
    rows, inputs, cols=Int(m), Int(n), Int(k)
    input=Matrix{T}(undef, inputs, cols)
    # Native passes valid MatRef/MatMut spans and signed element strides.
    # Copy every input before calling user code; escaped callback arrays
    # own Julia storage and never alias the borrowed native buffers.
    source=Ptr{T}(x)
    for j = 1:cols, i = 1:inputs
        input[i, j]=unsafe_load(source, 1+(i-1)*rs+(j-1)*cs)
    end
    output=fill(T(NaN), rows, cols)
    (adj==0 ? owner.forward : owner.adjoint)(output, input)
    all(isfinite, output) || throw(
        ArgumentError(
            "operator callback must overwrite every output entry with finite values",
        ),
    )
    destination=Ptr{T}(y)
    for j = 1:cols, i = 1:rows
        unsafe_store!(destination, output[i, j], 1+(i-1)*ys+(j-1)*yc)
    end
    return nothing
end
# Evaluate @cfunction at call time so a precompiled module never serializes an
# address from the precompilation process. The entrypoint has no closure.
_operator_callback_pointer() = @cfunction(
    _operator_callback_dispatch,
    Cint,
    (UInt, Int32, Int32, UInt, UInt, UInt, Ptr{UInt8}, Int, Int, Ptr{UInt8}, Int, Int)
)
function _native_call_with_callbacks(f, args...)
    owners=Dict{UInt,Any}()
    for arg in args
        retained=_operator_callback_owners(arg)
        isnothing(retained) || merge!(owners, retained)
    end
    context=_OperatorCallbackContext(owners, nothing, nothing)
    tls=task_local_storage()
    previous=get(tls, _operator_callback_context_key, nothing)
    result=Ref{Any}()
    body=() -> begin
        try
            result[]=_native_call_plain(f, args...)
            return Int32(1)
        catch e
            context.boundary_exception=Some(e)
            return Int32(0)
        end
    end
    # RustCall borrows callbacks on the calling OS thread. Pin a migratable
    # Julia task only while native frames are live, including callback yields.
    calling_task=current_task()
    old_sticky=calling_task.sticky
    outer_exception=nothing
    try
        tls[_operator_callback_context_key]=context
        calling_task.sticky=true
        _native_call_plain(
            RustCrate.with_operator_callbacks,
            _operator_callback_pointer(),
            body,
        )
    catch e
        outer_exception=Some(e)
    finally
        calling_task.sticky=old_sticky
        if isnothing(previous)
            delete!(tls, _operator_callback_context_key)
        else
            tls[_operator_callback_context_key]=previous
        end
    end
    isnothing(context.exception) || throw(something(context.exception))
    isnothing(context.boundary_exception) || throw(something(context.boundary_exception))
    isnothing(outer_exception) || throw(something(outer_exception))
    return result[]
end
"""
    FaerLinearOperator(apply!, adjoint_apply!, m, n; T=Float64)

Create an m-by-n native LinOp/BiLinOp backed by Julia functions. Each function
receives independent Julia matrices `(Y,X)` and must overwrite every entry of
`Y` with the forward or adjoint product, respectively. Multiple RHS columns and
promoted native scalar types may be supplied. Functions and captured state are
retained by reference; operator copies, transforms and compositions share that
state. User exceptions are rethrown after the native call returns and outputs
are staged before writeback. Escaped callback arrays remain valid Julia arrays.

The native solver still runs in Rust and invokes callbacks synchronously on the
calling Julia thread. Callback marshaling allocates owned Julia arrays outside
native scratch; this is not an allocation-free interface. The calling task is
pinned during the native call and its prior scheduling state is restored afterward.
Same-task recursive Colmat calls are supported; other tasks serialize on the
owner lock until the callback returns. Functions must define
linear, mutually adjoint actions. A nonfinite or incompletely written result is
rejected. Empty-domain/range and zero-column products are handled as zero maps.
"""
function FaerLinearOperator(
    apply!,
    adjoint_apply!,
    m::Integer,
    n::Integer;
    T::Type = Float64,
)
    T<:_NativeScalar ||
        throw(ArgumentError("callback operators require a native scalar type"))
    m>=0 && n>=0 || throw(ArgumentError("operator dimensions must be nonnegative"))
    dims=(Int(m), Int(n))
    lock(_matrix_lock) do
        _operator_callback_id[]==typemax(UInt) &&
            throw(OverflowError("operator callback identity exhausted"))
        _operator_callback_id[]+=1
        id=_operator_callback_id[]
        h=RustCrate.OwnedLinearOperator()
        _native_call(
            RustCrate.operator_callback,
            h,
            id,
            UInt(dims[1]),
            UInt(dims[2]),
            _tag(T),
        )
        owners=Dict{UInt,Any}(id=>(forward = apply!, adjoint = adjoint_apply!))
        return FaerLinearOperator{T,typeof(h)}(h, dims, :callback, false, false, owners)
    end
end
