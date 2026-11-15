using Test, LinearAlgebra, Random, Libdl, Colmat
function ordinary_schur_diagnostic()
    return ccall(
        Libdl.dlsym(Libdl.dlopen(Colmat.backend_library()), :colmat_blas_last_error),
        Cint,
        (),
    )
end
ordinary_schur_diagnostic()
function ordinary_schur_fixture(::Type{T}, n, form=:pairs) where {T}
    t=triu(randn(T, n, n)/max(1, n))
    for j in 1:n
        t[j, j]=form===:repeated ? T(2) : T(j+1)+(T<:Complex ? T(0, 0.1j) : zero(T))
    end
    if form===:zero
        fill!(t, 0)
    elseif form===:pairs&&T<:Real
        for j in 1:3:(n - 1)
            t[j, j]=t[j + 1, j + 1]=j+2
            t[j, j + 1]=(iseven(j) ? -1 : 1)*(j+0.5)
            t[j + 1, j]=-2t[j, j + 1]
        end
    end
    return t
end
function ordinary_schur_layout(t)
    n=size(t, 2)
    blocks=Tuple{Int,Int}[]
    j=1
    while j<=n
        len=eltype(t)<:Real&&j<n&&!iszero(t[j + 1, j]) ? 2 : 1
        push!(blocks, (j, len))
        j+=len
    end
    return blocks
end
function ordinary_schur_values(t)
    R=real(eltype(t))
    n=size(t, 2)
    w=Complex{R}.(diag(t[1:n, :]))
    for (j, len) in ordinary_schur_layout(t)
        if len==2
            beta=sqrt(abs(t[j, j + 1]))*sqrt(abs(t[j + 1, j]))
            re=t[j, j]/2+t[j + 1, j + 1]/2
            w[j]=complex(re, beta)
            w[j + 1]=complex(re, -beta)
        end
    end
    return w
end
for (T, prefix) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    J in (Int32, Int64)

    symbol=Symbol(prefix*"trexc_"*(J===Int64 ? "64_" : ""))
    signature=Any[Ref{UInt8}, Ref{J}, Ptr{T}, Ref{J}, Ptr{T}, Ref{J}, Ptr{J}, Ptr{J}]
    args=Any[:comp, :n, :t, :ldt, :q, :ldq, :(view(ifst, 2:2)), :(view(ilst, 2:2))]
    if T<:Real
        push!(signature, Ptr{T})
        push!(args, :(view(work, 2:(length(work) - 1))))
    end
    append!(signature, [Ptr{J}, Clong])
    append!(args, [:(view(info, 2:2)), 1])
    @eval function direct_ordinary_trexc!(
        t::Matrix{$T},
        q::Matrix{$T},
        ::Type{$J};
        first,
        last,
        comp=('V'),
        n=size(t, 2),
        ldt=size(t, 1),
        ldq=size(q, 1),
        pointer=nothing,
    )
        f=if isnothing(pointer)
            Libdl.dlsym(Libdl.dlopen(Colmat.backend_library()), $(QuoteNode(symbol)))
        else
            pointer
        end
        ifst=$J[-777, first, -777]
        ilst=$J[-777, last, -777]
        info=fill($J(-777), 3)
        work=fill($T(777), max(1, n)+2)
        ccall(f, Cvoid, $(Expr(:tuple, signature...)), $(args...))
        @test ifst[[1, 3]]==ilst[[1, 3]]==info[[1, 3]]==[$J(-777), $J(-777)]
        @test work[[1, end]]==[$T(777), $T(777)]
        (;
            first=ifst[2],
            last=ilst[2],
            info=info[2],
            diagnostic=ordinary_schur_diagnostic(),
        )
    end
    if T<:Real
        aux=Symbol(prefix*"laexc_"*(J===Int64 ? "64_" : ""))
        @eval function direct_ordinary_laexc!(
            t::Matrix{$T},
            q::Matrix{$T},
            ::Type{$J};
            first,
            n1,
            n2,
            want=true,
            n=size(t, 2),
            ldt=size(t, 1),
            ldq=size(q, 1),
            pointer=nothing,
        )
            f=if isnothing(pointer)
                Libdl.dlsym(Libdl.dlopen(Colmat.backend_library()), $(QuoteNode(aux)))
            else
                pointer
            end
            info=fill($J(-777), 3)
            work=fill($T(777), max(1, n)+2)
            ccall(
                f,
                Cvoid,
                (
                    Ref{$J},
                    Ref{$J},
                    Ptr{$T},
                    Ref{$J},
                    Ptr{$T},
                    Ref{$J},
                    Ref{$J},
                    Ref{$J},
                    Ref{$J},
                    Ptr{$T},
                    Ptr{$J},
                ),
                want,
                n,
                t,
                ldt,
                q,
                ldq,
                first,
                n1,
                n2,
                view(work, 2:(length(work) - 1)),
                view(info, 2:2),
            )
            @test info[[1, 3]]==[$J(-777), $J(-777)]&&work[[1, end]]==[$T(777), $T(777)]
            (; info=info[2], diagnostic=ordinary_schur_diagnostic())
        end
    end
end
@testset "TREXC direct ABIs block indices and accumulated similarity" begin
    Random.seed!(312)
    lib=Libdl.dlopen(
        only(
            filter(
                l->l.interface==:ilp64&&occursin(
                    "openblas", lowercase(basename(l.libname))
                ),
                BLAS.get_config().loaded_libs,
            ),
        ).libname,
    )
    for (T, prefix) in
        ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
        J in (Int32, Int64),
        n in (0, 1, 2, 5, 8),
        form in (:pairs, :triangular, :repeated, :zero),
        comp in ('N', 'V')

        tol=T in (Float32, ComplexF32) ? 3e-3 : 5e-10
        initial=ordinary_schur_fixture(T, n, form)
        basis=n==0 ? zeros(T, 0, 0) : Matrix(qr(randn(T, n, n)).Q)
        a=basis*initial*basis'
        for first in (n==0 ? (-7,) : (1:n)), last in (n==0 ? (9,) : (1:n))
            t=vcat(initial, fill(T(777), 3, n))
            tr=copy(t)
            old=copy(t)
            q=comp=='V' ? vcat(basis, fill(T(777), 3, n)) : fill(T(777), 1, 1)
            qref=copy(q)
            oldq=copy(q)
            result=direct_ordinary_trexc!(t, q, J; first, last, comp)
            oracle=direct_ordinary_trexc!(
                tr,
                qref,
                Int64;
                first,
                last,
                comp,
                pointer=Libdl.dlsym(lib, Symbol(prefix*"trexc_64_")),
            )
            @test result.info==result.diagnostic==oracle.info==0
            @test result.first==oracle.first&&result.last==oracle.last
            @test ordinary_schur_values(t)≈ordinary_schur_values(tr) atol=tol rtol=tol
            @test t[(n + 1):end, :]==old[(n + 1):end, :]
            if comp=='V'
                @test q[(n + 1):end, :]==qref[(n + 1):end, :]==oldq[(n + 1):end, :]
                @test q[1:n, :]*t[1:n, :]*q[1:n, :]'≈a atol=tol rtol=tol
                @test qref[1:n, :]*tr[1:n, :]*qref[1:n, :]'≈a atol=tol rtol=tol
                @test q[1:n, :]'*q[1:n, :]≈Matrix{T}(I, n, n) atol=tol rtol=tol
            else
                @test q==qref==oldq
            end
        end
    end
end
@testset "LAEXC adjacent 1x1 and 2x2 blocks and no-op dimensions" begin
    Random.seed!(3121)
    lib=Libdl.dlopen(
        only(
            filter(
                l->l.interface==:ilp64&&occursin(
                    "openblas", lowercase(basename(l.libname))
                ),
                BLAS.get_config().loaded_libs,
            ),
        ).libname,
    )
    for (T, prefix) in ((Float32, "s"), (Float64, "d")),
        J in (Int32, Int64),
        n1 in (1, 2),
        n2 in (1, 2),
        leading in (0, 1, 3),
        trailing in (0, 2),
        want in (false, true)

        n=leading+n1+n2+trailing
        first=leading+1
        tol=T===Float32 ? 3e-3 : 5e-10
        initial=ordinary_schur_fixture(T, n, :triangular)
        for (j, len, value) in ((first, n1, 10), (first+n1, n2, 20))
            for k in j:(j + len - 1)
                initial[k, k]=value
            end
            if len==2
                initial[j, j + 1]=3
                initial[j + 1, j]=-2
            end
        end
        basis=Matrix(qr(randn(T, n, n)).Q)
        a=basis*initial*basis'
        t=vcat(initial, fill(T(777), 3, n))
        tr=copy(t)
        old=copy(t)
        q=want ? vcat(basis, fill(T(777), 3, n)) : fill(T(777), 1, 1)
        qref=copy(q)
        oldq=copy(q)
        result=direct_ordinary_laexc!(t, q, J; first, n1, n2, want)
        oracle=direct_ordinary_laexc!(
            tr,
            qref,
            Int64;
            first,
            n1,
            n2,
            want,
            pointer=Libdl.dlsym(lib, Symbol(prefix*"laexc_64_")),
        )
        @test result.info==result.diagnostic==oracle.info==0
        @test ordinary_schur_values(t)≈ordinary_schur_values(tr) atol=tol rtol=tol
        @test t[(n + 1):end, :]==tr[(n + 1):end, :]==old[(n + 1):end, :]
        if want
            @test q[(n + 1):end, :]==qref[(n + 1):end, :]==oldq[(n + 1):end, :]
            @test q[1:n, :]*t[1:n, :]*q[1:n, :]'≈a atol=tol rtol=tol
            @test qref[1:n, :]*tr[1:n, :]*qref[1:n, :]'≈a atol=tol rtol=tol
            @test q[1:n, :]'*q[1:n, :]≈Matrix{T}(I, n, n) atol=tol rtol=tol
        else
            @test q==qref==oldq
        end
    end
    for T in (Float32, Float64),
        J in (Int32, Int64),
        (n, n1, n2) in ((0, 1, 2), (5, 0, 2), (5, 2, 0))

        t=fill(T(777), 1, 1)
        q=copy(t)
        result=direct_ordinary_laexc!(t, q, J; first=-7, n1, n2, n, ldt=1, ldq=1)
        @test result.info==result.diagnostic==0
        @test t==q==fill(T(777), 1, 1)
    end
end

function ordinary_rejected_window(::Type{T}) where {T}
    return T.(
        if T===Float32
            [
                1.0 0.5 8.881784e-16 536870900.0;
                -3.3881318e-21 1.0 128.0 5.421011e-20;
                0.0 0.0 1.000061 5.684342e-14;
                0.0 0.0 -67108864.0 1.000061
            ]
        else
            [
                1.0 7.1362384635298e+44 -3.7414441915671115e+50 -3.622271631530685e-71;
                -7.24454326306137e-71 1.0 -4.7477838728798994e-66 -5.795634610449096e-70;
                0.0 0.0 1.5 2.535301200456459e+30;
                0.0 0.0 -2.8698592549372254e-42 1.5
            ]
        end,
    )
end
@testset "Finite Schur swap rejection preserves matrices and indices" begin
    lib=Libdl.dlopen(
        only(
            filter(
                l->l.interface==:ilp64&&occursin(
                    "openblas", lowercase(basename(l.libname))
                ),
                BLAS.get_config().loaded_libs,
            ),
        ).libname,
    )
    for (T, prefix) in ((Float32, "s"), (Float64, "d")),
        J in (Int32, Int64),
        want in (false, true)

        initial=ordinary_rejected_window(T)
        n=4
        t=vcat(initial, fill(T(777), 3, n))
        tr=copy(t)
        old=copy(t)
        q=want ? vcat(Matrix{T}(I, n, n), fill(T(777), 3, n)) : fill(T(777), 1, 1)
        qref=copy(q)
        oldq=copy(q)
        result=direct_ordinary_laexc!(t, q, J; first=1, n1=2, n2=2, want)
        oracle=direct_ordinary_laexc!(
            tr,
            qref,
            Int64;
            first=1,
            n1=2,
            n2=2,
            want,
            pointer=Libdl.dlsym(lib, Symbol(prefix*"laexc_64_")),
        )
        @test result.info==oracle.info==1&&result.diagnostic==0
        @test isequal(t, old)&&isequal(tr, old)&&isequal(q, oldq)&&isequal(qref, oldq)
        for first in (1, 2), last in (3, 4)
            result=direct_ordinary_trexc!(t, q, J; first, last, comp=want ? 'V' : 'N')
            oracle=direct_ordinary_trexc!(
                tr,
                qref,
                Int64;
                first,
                last,
                comp=want ? 'V' : 'N',
                pointer=Libdl.dlsym(lib, Symbol(prefix*"trexc_64_")),
            )
            @test result.info==oracle.info==1&&result.diagnostic==0
            @test result.first==oracle.first==result.last==oracle.last==1
            @test isequal(t, old)&&isequal(tr, old)&&isequal(q, oldq)&&isequal(qref, oldq)
        end
    end
end
function ordinary_partial_rejection(::Type{T}) where {T}
    window=ordinary_rejected_window(T)
    t=zeros(T, 5, 5)
    index=[1, 2, 4, 5]
    t[index, index]=window
    t[3, 3]=-4
    return t
end
@testset "TREXC refusal retains the last successful adjacent swap" begin
    lib=Libdl.dlopen(
        only(
            filter(
                l->l.interface==:ilp64&&occursin(
                    "openblas", lowercase(basename(l.libname))
                ),
                BLAS.get_config().loaded_libs,
            ),
        ).libname,
    )
    for (T, prefix) in ((Float32, "s"), (Float64, "d")),
        J in (Int32, Int64), want in (false, true),
        original in (false, true)

        initial=ordinary_partial_rejection(T)
        n=5
        t=vcat(initial, fill(T(777), 3, n))
        expected=copy(t)
        q=want ? vcat(Matrix{T}(I, n, n), fill(T(777), 3, n)) : fill(T(777), 1, 1)
        expected_q=copy(q)
        pointer=original ? Libdl.dlsym(lib, Symbol(prefix*"trexc_64_")) : nothing
        IType=original ? Int64 : J
        first_move=direct_ordinary_trexc!(
            expected, expected_q, IType; first=1, last=3, comp=want ? 'V' : 'N', pointer
        )
        @test first_move.info==first_move.diagnostic==0&&first_move.last==2
        result=direct_ordinary_trexc!(
            t, q, IType; first=1, last=4, comp=want ? 'V' : 'N', pointer
        )
        @test result.info==1&&result.diagnostic==0
        @test result.first==1&&result.last==2
        @test isequal(t, expected)&&isequal(q, expected_q)
        @test !isequal(t[1:n, :], initial)
    end
end
