#=Preserve the original Float32 constrained system and compare independent precisions.

Run on the generalized-QR C adapter branch with Julia 1.13 and its built library.
All variables are constrained, so solution sensitivity is governed by B rather
than the stacked [A; B] matrix. This is a test-conditioning MWE, not a native bug.
=#
using Colmat, LinearAlgebra, Libdl
const original_a=Float32[
    1.3153454 0.38762516 0.51747465;
    -2.105331 0.9465635 -2.5433927;
    -0.4542911 1.1725777 -1.5416673
]
const original_b=Float32[
    1.1991587 -2.2223306 2.4921591;
    0.6084307 0.927926 -0.4180281;
    -0.8986303 -1.2177733 0.49271902
]
const original_c=Float32[-2.0647829, -0.5771696, 2.031597]
const original_d=Float32[0.17026924, -0.2891059, 1.1248497]
const native=Colmat.backend_library()
const reference=only(
    filter(
        l->l.interface==:ilp64&&occursin("openblas", lowercase(l.libname)),
        BLAS.get_config().loaded_libs,
    ),
).libname
for (T, prefix) in ((Float32, "s"), (Float64, "d"))
    @eval function solve_fortran(::Type{$T}, library)
        a=Matrix{$T}(original_a)
        b=Matrix{$T}(original_b)
        c=Vector{$T}(original_c)
        d=Vector{$T}(original_d)
        x=zeros($T, 3)
        w=zeros($T, 64)
        info=Ref{Int64}(-777)
        ccall(
            Libdl.dlsym(Libdl.dlopen(library), $prefix*"gglse_64_"),
            Cvoid,
            (
                Ref{Int64},
                Ref{Int64},
                Ref{Int64},
                Ptr{$T},
                Ref{Int64},
                Ptr{$T},
                Ref{Int64},
                Ptr{$T},
                Ptr{$T},
                Ptr{$T},
                Ptr{$T},
                Ref{Int64},
                Ref{Int64},
            ),
            3,
            3,
            3,
            a,
            3,
            b,
            3,
            c,
            d,
            x,
            w,
            length(w),
            info,
        )
        @assert info[]==0
        x
    end
end
function independent_condition(input)
    a=Matrix{Float64}(input)
    m, n=size(a)
    s=zeros(Float64, min(m, n))
    u=zeros(1)
    vt=zeros(1)
    w=zeros(1024)
    info=Ref{Int64}(-777)
    ccall(
        Libdl.dlsym(Libdl.dlopen(reference), :dgesvd_64_),
        Cvoid,
        (
            Ref{UInt8},
            Ref{UInt8},
            Ref{Int64},
            Ref{Int64},
            Ptr{Float64},
            Ref{Int64},
            Ptr{Float64},
            Ptr{Float64},
            Ref{Int64},
            Ptr{Float64},
            Ref{Int64},
            Ptr{Float64},
            Ref{Int64},
            Ref{Int64},
            Csize_t,
            Csize_t,
        ),
        UInt8('N'),
        UInt8('N'),
        m,
        n,
        a,
        m,
        s,
        u,
        1,
        vt,
        1,
        w,
        length(w),
        info,
        1,
        1,
    )
    @assert info[]==0&&all(isfinite, s)&&last(s)>0
    return first(s)/last(s)
end
function solve_row_c()
    a=fill(777.0f0, 20)
    b=fill(777.0f0, 20)
    for i in 1:3, j in 1:3
        a[1 + (i - 1) * 6 + j]=original_a[i, j]
        b[1 + (i - 1) * 6 + j]=original_b[i, j]
    end
    c=vcat(777.0f0, original_c, 777.0f0)
    d=vcat(777.0f0, original_d, 777.0f0)
    x=Float32[777, 0, 0, 0, 777]
    info=ccall(
        Libdl.dlsym(Libdl.dlopen(native), :LAPACKE_sgglse64_),
        Int64,
        (
            Cint,
            Int64,
            Int64,
            Int64,
            Ptr{Float32},
            Int64,
            Ptr{Float32},
            Int64,
            Ptr{Float32},
            Ptr{Float32},
            Ptr{Float32},
        ),
        101,
        3,
        3,
        3,
        view(a, 2:19),
        6,
        view(b, 2:19),
        6,
        view(c, 2:4),
        view(d, 2:4),
        view(x, 2:4),
    )
    @assert info==0
    @assert a[[1, end]]==b[[1, end]]==c[[1, end]]==d[[1, end]]==x[[1, end]]==Float32[
        777, 777
    ]
    return x[2:4]
end
actual=solve_row_c();
native_fortran=solve_fortran(Float32, native)
reference_single=solve_fortran(Float32, reference);
reference_wide=solve_fortran(Float64, reference)
constraint_condition=independent_condition(original_b);
stack_condition=independent_condition(vcat(original_a, original_b))
relative_error(x) = norm(Float64.(x)-reference_wide)/norm(reference_wide)
function backward_error(x)
    return norm(Float64.(original_b)*Float64.(x)-Float64.(original_d))/(
        norm(Float64.(original_b))*norm(Float64.(x))+norm(Float64.(original_d))
    )
end
@assert actual==native_fortran
@assert constraint_condition>1000stack_condition
@assert relative_error(actual)<20eps(Float32)*constraint_condition
@assert relative_error(reference_single)<20eps(Float32)*constraint_condition
@assert backward_error(actual)<100eps(Float32)&&backward_error(reference_single)<100eps(
    Float32
)
println((;
    constraint_condition,
    stack_condition,
    actual,
    reference_single,
    reference_wide,
    native_relative_error=relative_error(actual),
    reference_relative_error=relative_error(reference_single),
    native_backward_error=backward_error(actual),
    reference_backward_error=backward_error(reference_single),
))
