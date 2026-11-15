using Test, LinearAlgebra, Random, Libdl, Colmat

function packed_tridiagonal_diagnostic()
    return ccall(
        Libdl.dlsym(Libdl.dlopen(Colmat.backend_library()), :colmat_blas_last_error),
        Cint,
        (),
    )
end
packed_tridiagonal_diagnostic()
function packed_tridiagonal_store(a, u)
    n=size(a, 2)
    p=fill(eltype(a)(777), n*(n+1)÷2+2)
    k=2
    for j in 1:n, i in (u=='L' ? (j:n) : (1:j))
        p[k]=a[i, j]
        k+=1
    end
    return p
end
for (T, prefix) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    J in (Int32, Int64)

    R=real(T)
    suffix=J===Int64 ? "64_" : ""
    trd=Symbol(prefix*(T<:Real ? "sptrd_" : "hptrd_")*suffix)
    gtr=Symbol(prefix*(T<:Real ? "opgtr_" : "upgtr_")*suffix)
    mtr=Symbol(prefix*(T<:Real ? "opmtr_" : "upmtr_")*suffix)
    @eval function direct_packed_trd!(p::Vector{$T}, ::Type{$J}; n, u, pointer=nothing)
        f=if isnothing(pointer)
            Libdl.dlsym(Libdl.dlopen(Colmat.backend_library()), $(QuoteNode(trd)))
        else
            pointer
        end
        d=fill($R(777), n+2)
        e=fill($R(777), max(0, n-1)+2)
        tau=fill($T(777), max(0, n-1)+2)
        info=fill($J(-777), 3)
        ccall(
            f,
            Cvoid,
            (Ref{UInt8}, Ref{$J}, Ptr{$T}, Ptr{$R}, Ptr{$R}, Ptr{$T}, Ptr{$J}, Clong),
            u,
            n,
            view(p, 2:(length(p) - 1)),
            view(d, 2:(length(d) - 1)),
            view(e, 2:(length(e) - 1)),
            view(tau, 2:(length(tau) - 1)),
            view(info, 2:2),
            1,
        )
        @test info[[1, 3]]==[$J(-777), $J(-777)]
        @test d[[1, end]]==[$R(777), $R(777)] && e[[1, end]]==[$R(777), $R(777)]
        @test tau[[1, end]]==[$T(777), $T(777)] && p[[1, end]]==[$T(777), $T(777)]
        (;
            values=d[2:(end - 1)],
            offdiag=e[2:(end - 1)],
            tau=tau[2:(end - 1)],
            info=info[2],
            diagnostic=packed_tridiagonal_diagnostic(),
        )
    end
    @eval function direct_packed_gtr!(
        p::Vector{$T}, tau::Vector{$T}, q::Matrix{$T}, ::Type{$J}; n, u, pointer=nothing
    )
        f=if isnothing(pointer)
            Libdl.dlsym(Libdl.dlopen(Colmat.backend_library()), $(QuoteNode(gtr)))
        else
            pointer
        end
        work=fill($T(777), max(0, n-1)+2)
        info=fill($J(-777), 3)
        ccall(
            f,
            Cvoid,
            (
                Ref{UInt8},
                Ref{$J},
                Ptr{$T},
                Ptr{$T},
                Ptr{$T},
                Ref{$J},
                Ptr{$T},
                Ptr{$J},
                Clong,
            ),
            u,
            n,
            view(p, 2:(length(p) - 1)),
            tau,
            q,
            size(q, 1),
            view(work, 2:(length(work) - 1)),
            view(info, 2:2),
            1,
        )
        @test info[[1, 3]]==[$J(-777), $J(-777)] && work[[1, end]]==[$T(777), $T(777)]
        (; info=info[2], diagnostic=packed_tridiagonal_diagnostic())
    end
    @eval function direct_packed_mtr!(
        p::Vector{$T},
        tau::Vector{$T},
        c::Matrix{$T},
        ::Type{$J};
        m,
        n,
        side,
        u,
        tr,
        pointer=nothing,
    )
        f=if isnothing(pointer)
            Libdl.dlsym(Libdl.dlopen(Colmat.backend_library()), $(QuoteNode(mtr)))
        else
            pointer
        end
        work=fill($T(777), (side=='L' ? n : m)+2)
        info=fill($J(-777), 3)
        ccall(
            f,
            Cvoid,
            (
                Ref{UInt8},
                Ref{UInt8},
                Ref{UInt8},
                Ref{$J},
                Ref{$J},
                Ptr{$T},
                Ptr{$T},
                Ptr{$T},
                Ref{$J},
                Ptr{$T},
                Ptr{$J},
                Clong,
                Clong,
                Clong,
            ),
            side,
            u,
            tr,
            m,
            n,
            view(p, 2:(length(p) - 1)),
            tau,
            c,
            size(c, 1),
            view(work, 2:(length(work) - 1)),
            view(info, 2:2),
            1,
            1,
            1,
        )
        @test info[[1, 3]]==[$J(-777), $J(-777)] && work[[1, end]]==[$T(777), $T(777)]
        (; info=info[2], diagnostic=packed_tridiagonal_diagnostic())
    end
end

@testset "Packed tridiagonal native and reference factors interoperate" begin
    Random.seed!(309)
    external=only(
        filter(
            l->l.interface==:ilp64&&occursin("openblas", lowercase(basename(l.libname))),
            BLAS.get_config().loaded_libs,
        ),
    )
    lib=Libdl.dlopen(external.libname)
    for (T, prefix) in
        ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
        J in (Int32, Int64), u in ('U', 'L'), n in (0, 1, 2, 5, 9),
        form in (:dense, :zero, :rankone, :repeated)

        tol=T in (Float32, ComplexF32) ? 2e-3 : 2e-10
        raw=randn(T, n, n)/max(1, n)
        a=if form===:dense
            raw+raw'
        elseif form===:zero
            zeros(T, n, n)
        elseif form===:rankone
            v=randn(T, n)
            v*v'
        else
            Matrix(Diagonal(T.(mod.(1:n, 2))))
        end
        p=packed_tridiagonal_store(a, u)
        if T<:Complex
            k=2
            for j in 1:n, i in (u=='L' ? (j:n) : (1:j))
                i==j&&(p[k]=T(real(p[k]), NaN))
                k+=1
            end
        end
        pr=copy(p)
        native=direct_packed_trd!(p, J; n, u)
        reference=direct_packed_trd!(
            pr,
            Int64;
            n,
            u,
            pointer=Libdl.dlsym(lib, Symbol(prefix*(T<:Real ? "sptrd_64_" : "hptrd_64_"))),
        )
        @test native.info==native.diagnostic==reference.info==0
        for (factors, result) in ((p, native), (pr, reference))
            old=copy(factors)
            oldtau=copy(result.tau)
            q=fill(T(777), n+3, n)
            qr=copy(q)
            formed=direct_packed_gtr!(factors, result.tau, q, J; n, u)
            oracle=direct_packed_gtr!(
                factors,
                result.tau,
                qr,
                Int64;
                n,
                u,
                pointer=Libdl.dlsym(
                    lib, Symbol(prefix*(T<:Real ? "opgtr_64_" : "upgtr_64_"))
                ),
            )
            @test formed.info==formed.diagnostic==oracle.info==0
            @test isequal(factors, old)&&isequal(result.tau, oldtau)
            @test q≈qr atol=tol rtol=tol
            basis=q[1:n, :]
            tridiagonal=Matrix(SymTridiagonal(result.values, result.offdiag))
            @test basis*tridiagonal*basis'≈a atol=tol rtol=tol
            @test basis'*basis≈Matrix{T}(I, n, n) atol=tol rtol=tol
            @test all(==(T(777)), q[(n + 1):end, :])
            for side in ('L', 'R'), tr in ('N', T<:Real ? 'T' : 'C')
                m, cols=side=='L' ? (n, 3) : (3, n)
                c=fill(T(777), m+3, cols)
                c[1:m, :].=randn(T, m, cols)
                before=copy(c)
                cr=copy(c)
                applied=direct_packed_mtr!(
                    factors, result.tau, c, J; m, n=cols, side, u, tr
                )
                oracle=direct_packed_mtr!(
                    factors,
                    result.tau,
                    cr,
                    Int64;
                    m,
                    n=cols,
                    side,
                    u,
                    tr,
                    pointer=Libdl.dlsym(
                        lib, Symbol(prefix*(T<:Real ? "opmtr_64_" : "upmtr_64_"))
                    ),
                )
                @test applied.info==applied.diagnostic==oracle.info==0
                @test c≈cr atol=tol rtol=tol
                op=tr=='N' ? basis : basis'
                expected=side=='L' ? op*before[1:m, :] : before[1:m, :]*op
                @test c[1:m, :]≈expected atol=tol rtol=tol
                @test c[(m + 1):end, :]==before[(m + 1):end, :]
                @test isequal(factors, old)&&isequal(result.tau, oldtau)
            end
        end
    end
end
