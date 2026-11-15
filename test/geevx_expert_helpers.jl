using Test, LinearAlgebra, Random, Libdl, Colmat
function geevx_reference()
    return only(
        filter(
            l->l.interface==:ilp64 && occursin("openblas", lowercase(basename(l.libname))),
            BLAS.get_config().loaded_libs,
        ),
    ).libname
end
function geevx_diagnostic()
    return ccall(
        Libdl.dlsym(Libdl.dlopen(Colmat.backend_library()), :colmat_blas_last_error),
        Cint,
        (),
    )
end
geevx_diagnostic()
function geevx_minimum(T, n, left, right, sense)
    n==0 && return 1
    base=T<:Complex || (left=='N' && right=='N') ? 2n : 3n
    return max(1, base, sense in ('V', 'B') ? n*n+(T<:Complex ? 2n : 6n) : 1)
end
for (T, prefix) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    J in (Int32, Int64)

    R=real(T)
    symbol=Symbol(prefix*"geevx_"*(J===Int64 ? "64_" : ""))
    signature=Any[
        Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{J}, Ptr{T}, Ref{J}, Ptr{T}
    ]
    args=Any[:bal, :left, :right, :sense, :n, :a, :lda, :(view(w, 2:(length(w) - 1)))]
    if T<:Real
        push!(signature, Ptr{R})
        push!(args, :(view(wi, 2:(length(wi) - 1))))
    end
    append!(
        signature,
        [
            Ptr{T},
            Ref{J},
            Ptr{T},
            Ref{J},
            Ptr{J},
            Ptr{J},
            Ptr{R},
            Ptr{R},
            Ptr{R},
            Ptr{R},
            Ptr{T},
            Ref{J},
        ],
    )
    append!(
        args,
        [
            :vl,
            :ldvl,
            :vr,
            :ldvr,
            :(view(ilo, 2:2)),
            :(view(ihi, 2:2)),
            :(view(scale, 2:(length(scale) - 1))),
            :(view(abnrm, 2:2)),
            :(view(rconde, 2:(length(rconde) - 1))),
            :(view(rcondv, 2:(length(rcondv) - 1))),
            :(view(work, 2:(length(work) - 1))),
            :lwork,
        ],
    )
    push!(signature, T<:Real ? Ptr{J} : Ptr{R})
    push!(
        args,
        if T<:Real
            :(view(iwork, 2:(length(iwork) - 1)))
        else
            :(view(rwork, 2:(length(rwork) - 1)))
        end,
    )
    append!(signature, [Ptr{J}, Clong, Clong, Clong, Clong])
    append!(args, [:(view(info, 2:2)), 1, 1, 1, 1])
    @eval function direct_geevx!(
        a::Matrix{$T},
        ::Type{$J};
        n=size(a, 2),
        bal=('N'),
        left=('V'),
        right=('V'),
        sense=('N'),
        lwork=geevx_minimum($T, n, left, right, sense),
        lda=size(a, 1),
        ldvl=max(1, n+2),
        ldvr=max(1, n+2),
        library=Colmat.backend_library(),
        pointer=nothing,
    )
        actual=size(a, 2)
        w=fill($T(777), actual+2)
        wi=fill($R(777), actual+2)
        vl=fill($T(777), max(1, actual+2), actual)
        vr=copy(vl)
        ilo=fill($J(-777), 3)
        ihi=copy(ilo)
        info=copy(ilo)
        scale=fill($R(777), actual+2)
        abnrm=fill($R(777), 3)
        rconde=copy(scale)
        rcondv=copy(scale)
        work=fill($T(777), max(1, lwork)+2)
        iwork=fill($J(777), max(1, 2actual-2)+2)
        rwork=fill($R(777), max(1, 2actual)+2)
        old=deepcopy((
            a, w, wi, vl, vr, ilo, ihi, scale, abnrm, rconde, rcondv, iwork, rwork
        ))
        f=if isnothing(pointer)
            Libdl.dlsym(Libdl.dlopen(library), $(QuoteNode(symbol)))
        else
            pointer
        end
        ccall(f, Cvoid, $(Expr(:tuple, signature...)), $(args...))
        @test ilo[[1, 3]]==ihi[[1, 3]]==info[[1, 3]]==[$J(-777), $J(-777)]
        @test work[[1, end]]==w[[1, end]]==[$T(777), $T(777)]
        @test wi[[1, end]]==scale[[1, end]]==rconde[[1, end]]==rcondv[[1, end]]==abnrm[[
            1, end
        ]]==[$R(777), $R(777)]
        @test iwork[[1, end]]==[$J(777), $J(777)] && rwork[[1, end]]==[$R(777), $R(777)]
        @test all(==($T(777)), vl[(actual + 1):end, :]) &&
            all(==($T(777)), vr[(actual + 1):end, :])
        # Complex TREVC3 queried by Netlib GEEVX may write the RWORK scratch.
        # Matrix, eigenvalue/vector and condition outputs remain unchanged.
        if lwork == -1 && info[2]==0
            @test isequal(
                (a, w, wi, vl, vr, ilo, ihi, scale, abnrm, rconde, rcondv, iwork), old[1:12]
            )
        end
        (;
            info=info[2],
            diagnostic=geevx_diagnostic(),
            workspace=real(work[2]),
            ilo=ilo[2],
            ihi=ihi[2],
            values=w[2:(end - 1)],
            imaginary=wi[2:(end - 1)],
            left=vl[1:actual, :],
            right=vr[1:actual, :],
            scale=scale[2:(end - 1)],
            abnrm=abnrm[2],
            rconde=rconde[2:(end - 1)],
            rcondv=rcondv[2:(end - 1)],
            iwork=iwork[2:(end - 1)],
            rwork=rwork[2:(end - 1)],
        )
    end
end
function geevx_unpack(x, wi)
    eltype(x)<:Complex && return x
    z=complex.(x)
    j=1
    while j<=length(wi)
        if wi[j]>0
            z[:, j]=x[:, j]+im*x[:, j + 1]
            z[:, j + 1]=conj.(z[:, j])
            j+=2
        else
            j+=1
        end
    end
    return z
end
function geevx_fixture(T, n)
    a=triu(randn(T, n, n)/max(1, n))
    for j in 1:n
        a[j, j]=T<:Real ? T(j+1) : T(j+1, 0.2j)
    end
    if T<:Real && n>=2
        a[1, 1]=a[2, 2]=2
        a[1, 2]=0.75
        a[2, 1]=-0.5
    end
    return a
end
