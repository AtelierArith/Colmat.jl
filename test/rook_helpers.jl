using Test, LinearAlgebra, Random, Libdl, Colmat
rook_sentinel(::Type{J}) where {J<:Integer} = J===Int64 ? J(0x1357913579) : J(0x13579)
rook_sentinel(::Type{T}) where {T} = T(777)
function rook_storage(T, n)
    storage=fill(rook_sentinel(T), n+2)
    return storage, view(storage, 2:(n + 1))
end
function rook_guards(storage)
    @test storage[1]==storage[end]==rook_sentinel(eltype(storage))
end
function rook_diagnostic()
    return ccall(
        Libdl.dlsym(Libdl.dlopen(Colmat.backend_library()), :colmat_blas_last_error),
        Cint,
        (),
    )
end
function rook_reference()
    return only(
        filter(
            l->l.interface==:ilp64 && occursin("openblas", lowercase(basename(l.libname))),
            BLAS.get_config().loaded_libs,
        ),
    ).libname
end
function rook_stage(a, uplo)
    n=size(a, 1)
    b=fill(eltype(a)(777), n+2, n)
    for j in 1:n, i in 1:n
        if (uplo=='L' && i>=j)||(uplo=='U' && i<=j)
            b[i, j]=a[i, j]
            if i==j && eltype(a)<:Complex
                b[i, j]=complex(real(b[i, j]), NaN)
            end
        end
    end
    return b
end
function rook_matrix_guards(a, old, uplo)
    n=size(a, 2)
    for j in 1:n, i in 1:size(a, 1)
        if i>n || (uplo=='L' && i<j) || (uplo=='U' && i>j)
            @test isequal(a[i, j], old[i, j])
        end
    end
end
for (T, p, k) in (
        (Float32, "s", "sy"),
        (Float64, "d", "sy"),
        (ComplexF32, "c", "he"),
        (ComplexF64, "z", "he"),
    ),
    J in (Int32, Int64)

    suffix=J==Int32 ? "_" : "_64_"
    trf=Symbol(p*k*"trf_rook"*suffix)
    tf2=Symbol(p*k*"tf2_rook"*suffix)
    trs=Symbol(p*k*"trs_rook"*suffix)
    tri=Symbol(p*k*"tri_rook"*suffix)
    sv=Symbol(p*k*"sv_rook"*suffix)
    @eval function rook_factor!(
        a::Matrix{$T},
        ::Type{$J};
        uplo=('L'),
        n=size(a, 2),
        lda=max(1, stride(a, 2)),
        unblocked=false,
        lwork=1,
        library=Colmat.backend_library(),
        pointer=nothing,
    )
        pg, piv=rook_storage($J, size(a, 2))
        wg, work=rook_storage($T, max(1, lwork))
        ig, info=rook_storage($J, 1)
        f=if isnothing(pointer)
            Libdl.dlsym(
                Libdl.dlopen(library), unblocked ? $(QuoteNode(tf2)) : $(QuoteNode(trf))
            )
        else
            pointer
        end
        if unblocked
            ccall(
                f,
                Cvoid,
                (Ref{UInt8}, Ref{$J}, Ptr{$T}, Ref{$J}, Ptr{$J}, Ptr{$J}, Clong),
                uplo,
                n,
                a,
                lda,
                piv,
                info,
                1,
            )
        else
            ccall(
                f,
                Cvoid,
                (
                    Ref{UInt8},
                    Ref{$J},
                    Ptr{$T},
                    Ref{$J},
                    Ptr{$J},
                    Ptr{$T},
                    Ref{$J},
                    Ptr{$J},
                    Clong,
                ),
                uplo,
                n,
                a,
                lda,
                piv,
                work,
                lwork,
                info,
                1,
            )
        end
        rook_guards(pg)
        rook_guards(wg)
        rook_guards(ig)
        (; piv=copy(piv), info=info[1], work=copy(work), diagnostic=rook_diagnostic())
    end
    @eval function rook_solve!(
        a::Matrix{$T},
        piv::Vector{$J},
        b::Matrix{$T};
        uplo=('L'),
        n=size(a, 2),
        lda=max(1, stride(a, 2)),
        nrhs=size(b, 2),
        ldb=max(1, stride(b, 2)),
        library=Colmat.backend_library(),
        pointer=nothing,
    )
        ig, info=rook_storage($J, 1)
        old=copy(piv)
        f=if isnothing(pointer)
            Libdl.dlsym(Libdl.dlopen(library), $(QuoteNode(trs)))
        else
            pointer
        end
        ccall(
            f,
            Cvoid,
            (
                Ref{UInt8},
                Ref{$J},
                Ref{$J},
                Ptr{$T},
                Ref{$J},
                Ptr{$J},
                Ptr{$T},
                Ref{$J},
                Ptr{$J},
                Clong,
            ),
            uplo,
            n,
            nrhs,
            a,
            lda,
            piv,
            b,
            ldb,
            info,
            1,
        )
        rook_guards(ig)
        @test piv==old
        (; info=info[1], diagnostic=rook_diagnostic())
    end
    @eval function rook_inverse!(
        a::Matrix{$T},
        piv::Vector{$J};
        uplo=('L'),
        n=size(a, 2),
        lda=max(1, stride(a, 2)),
        library=Colmat.backend_library(),
        pointer=nothing,
    )
        wg, work=rook_storage($T, size(a, 2))
        ig, info=rook_storage($J, 1)
        old=copy(piv)
        f=if isnothing(pointer)
            Libdl.dlsym(Libdl.dlopen(library), $(QuoteNode(tri)))
        else
            pointer
        end
        ccall(
            f,
            Cvoid,
            (Ref{UInt8}, Ref{$J}, Ptr{$T}, Ref{$J}, Ptr{$J}, Ptr{$T}, Ptr{$J}, Clong),
            uplo,
            n,
            a,
            lda,
            piv,
            work,
            info,
            1,
        )
        rook_guards(wg)
        rook_guards(ig)
        @test piv==old
        (; info=info[1], diagnostic=rook_diagnostic())
    end
    @eval function rook_driver!(
        a::Matrix{$T},
        b::Matrix{$T},
        ::Type{$J};
        uplo=('L'),
        n=size(a, 2),
        lda=max(1, stride(a, 2)),
        nrhs=size(b, 2),
        ldb=max(1, stride(b, 2)),
        lwork=1,
        library=Colmat.backend_library(),
        pointer=nothing,
    )
        pg, piv=rook_storage($J, size(a, 2))
        wg, work=rook_storage($T, max(1, lwork))
        ig, info=rook_storage($J, 1)
        f=if isnothing(pointer)
            Libdl.dlsym(Libdl.dlopen(library), $(QuoteNode(sv)))
        else
            pointer
        end
        ccall(
            f,
            Cvoid,
            (
                Ref{UInt8},
                Ref{$J},
                Ref{$J},
                Ptr{$T},
                Ref{$J},
                Ptr{$J},
                Ptr{$T},
                Ref{$J},
                Ptr{$T},
                Ref{$J},
                Ptr{$J},
                Clong,
            ),
            uplo,
            n,
            nrhs,
            a,
            lda,
            piv,
            b,
            ldb,
            work,
            lwork,
            info,
            1,
        )
        rook_guards(pg)
        rook_guards(wg)
        rook_guards(ig)
        (; piv=copy(piv), info=info[1], work=copy(work), diagnostic=rook_diagnostic())
    end
end

function rook_pointer(T, J, operation, route; classic=false)
    prefix=if T==Float32
        "s"
    elseif T==Float64
        "d"
    elseif T==ComplexF32
        "c"
    else
        "z"
    end
    kind=T<:Real ? "sy" : "he"
    stem=prefix*kind*operation*(classic ? "" : "_rook")
    suffix=J==Int32 ? "_" : "_64_"
    native=Libdl.dlsym(Libdl.dlopen(Colmat.backend_library()), stem*suffix)
    @test BLAS.lbt_get_forward(stem*"_", J==Int32 ? :lp64 : :ilp64)==native
    return if route==:dll
        native
    else
        Libdl.dlsym(Libdl.dlopen(BLAS.libblastrampoline), stem*suffix)
    end
end

# Independent elementary-matrix product: multiply each step's permutations and
# unit triangular factor, then the explicit block diagonal D and its adjoint.
function rook_reconstruct(a, piv, uplo; classic=false)
    T=eltype(a)
    n=length(piv)
    f=Matrix{T}(I, n, n)
    d=zeros(T, n, n)
    k=uplo=='L' ? 1 : n
    while 1<=k<=n
        pair=piv[k]<0
        other=k+(uplo=='L' ? 1 : -1)
        columns=pair ? (k, other) : (k,)
        for column in columns
            if !(classic && pair && column==k)
                pivot=abs(Int(piv[column]))
                f[:, [column, pivot]]=f[:, [pivot, column]]
            end
            d[column, column]=real(a[column, column])
        end
        if pair
            d[other, k]=a[other, k]
            d[k, other]=conj(a[other, k])
        end
        step=Matrix{T}(I, n, n)
        rows=uplo=='L' ? ((k + length(columns)):n) : (1:(k - length(columns)))
        for column in columns, i in rows
            step[i, column]=a[i, column]
        end
        f=f*step
        k+=uplo=='L' ? length(columns) : -length(columns)
    end
    return f*d*f'
end
