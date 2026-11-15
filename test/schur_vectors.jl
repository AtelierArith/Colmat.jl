using Test, LinearAlgebra, Random, Libdl, Colmat
function schur_vectors_diagnostic()
    return ccall(
        Libdl.dlsym(Libdl.dlopen(Colmat.backend_library()), :colmat_blas_last_error),
        Cint,
        (),
    )
end
schur_vectors_diagnostic()
for (T, prefix) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    J in (Int32, Int64),
    blocked in (false, true)

    R=real(T)
    symbol=Symbol(prefix*(blocked ? "trevc3_" : "trevc_")*(J===Int64 ? "64_" : ""))
    signature=Any[
        Ref{UInt8},
        Ref{UInt8},
        Ptr{J},
        Ref{J},
        Ptr{T},
        Ref{J},
        Ptr{T},
        Ref{J},
        Ptr{T},
        Ref{J},
        Ref{J},
        Ptr{J},
        Ptr{T},
    ]
    args=Any[
        :side,
        :how,
        :(view(selected, 2:(length(selected) - 1))),
        :n,
        :t,
        :ldt,
        :vl,
        :ldvl,
        :vr,
        :ldvr,
        :mm,
        :(view(m, 2:2)),
        :(view(work, 2:(length(work) - 1))),
    ]
    if blocked
        push!(signature, Ref{J})
        push!(args, :(query===:work ? -1 : lwork))
    end
    if T<:Complex
        push!(signature, Ptr{R})
        push!(args, :(view(rwork, 2:(length(rwork) - 1))))
        if blocked
            push!(signature, Ref{J})
            push!(args, :(query===:real ? -1 : lrwork))
        end
    end
    append!(signature, [Ptr{J}, Clong, Clong])
    append!(args, [:(view(info, 2:2)), 1, 1])
    @eval function direct_schur_vectors!(
        t::Matrix{$T},
        vl::Matrix{$T},
        vr::Matrix{$T},
        selected::Vector{$J},
        ::Type{$J},
        ::Val{$blocked};
        n=size(t, 2),
        side=('B'),
        how=('A'),
        mm=n,
        ldt=size(t, 1),
        ldvl=size(vl, 1),
        ldvr=size(vr, 1),
        lwork=max(1, ($T<:Real ? 3 : 2)*n),
        lrwork=max(1, n),
        query=:none,
        pointer=nothing,
    )
        f=if isnothing(pointer)
            Libdl.dlsym(Libdl.dlopen(Colmat.backend_library()), $(QuoteNode(symbol)))
        else
            pointer
        end
        work=fill($T(777), (query===:none ? max(1, lwork) : 1)+2)
        rwork=fill($R(777), (query===:none ? max(1, lrwork) : 1)+2)
        m=fill($J(-777), 3)
        info=fill($J(-777), 3)
        ccall(f, Cvoid, $(Expr(:tuple, signature...)), $(args...))
        @test m[[1, 3]]==[$J(-777), $J(-777)]&&info[[1, 3]]==[$J(-777), $J(-777)]
        @test work[[1, end]]==[$T(777), $T(777)]&&rwork[[1, end]]==[$R(777), $R(777)]
        @test selected[[1, end]]==[$J(-777), $J(-777)]
        (;
            count=m[2],
            info=info[2],
            diagnostic=schur_vectors_diagnostic(),
            work=work[2:(end - 1)],
            rwork=rwork[2:(end - 1)],
        )
    end
end
function schur_vectors_fixture(::Type{T}, n, form) where {T}
    raw=randn(T, n, n)/max(1, n)
    t=triu(raw)
    for j in 1:n
        t[j, j]=form===:repeated ? T(2) : T(j+1)+(T<:Complex ? T(0, 0.3j) : zero(T))
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
function schur_vectors_layout(t)
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
function check_schur_vectors(result, vl, vr, t, layout, columns, side, how, ql, qr, tol)
    @test result.info==0
    @test result.count==length(columns)
    n=size(t, 2)
    for (left, output, q) in ((true, vl, ql), (false, vr, qr))
        requested=left ? side!='R' : side!='L'
        if !requested
            @test all(==(eltype(t)(777)), output)
            continue
        end
        matrix=how=='B' ? q*t*q' : t
        column=1
        for (j, len) in layout
            j in columns||continue
            x=if len==2
                complex.(output[1:n, column], output[1:n, column + 1])
            else
                complex.(output[1:n, column])
            end
            lambda=if len==2
                complex(t[j, j], sqrt(abs(t[j, j + 1]))*sqrt(abs(t[j + 1, j])))
            else
                complex(t[j, j])
            end
            lhs=left ? matrix'*x : matrix*x
            rhs=(left ? conj(lambda) : lambda)*x
            scale=max(norm(matrix)*norm(x), abs(lambda)*norm(x), floatmin(real(eltype(t))))
            @test norm(lhs-rhs)<=tol*scale
            @test maximum(abs.(real.(x))+abs.(imag.(x)); init=0)≈1 atol=tol rtol=tol
            column+=len
        end
        @test all(==(eltype(t)(777)), output[(n + 1):end, :])
        @test all(==(eltype(t)(777)), output[:, (length(columns) + 1):end])
    end
end
@testset "Schur eigenvectors selected left right and back-transform contracts" begin
    Random.seed!(311)
    external=only(
        filter(
            l->l.interface==:ilp64&&occursin("openblas", lowercase(basename(l.libname))),
            BLAS.get_config().loaded_libs,
        ),
    )
    reference=Libdl.dlopen(external.libname)
    for (T, prefix) in
        ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
        J in (Int32, Int64),
        blocked in (false, true),
        n in (0, 1, 2, 5, 8),
        side in ('L', 'R', 'B'),
        how in ('A', 'B', 'S'),
        form in (:triangular, :pairs, :repeated, :zero)

        R=real(T)
        tol=R===Float32 ? 4e-3 : 5e-10
        t=schur_vectors_fixture(T, n, form)
        layout=schur_vectors_layout(t)
        ql=n==0 ? zeros(T, 0, 0) : Matrix(qr(randn(T, n, n)).Q)
        qr_matrix=n==0 ? zeros(T, 0, 0) : Matrix(qr(randn(T, n, n)).Q)
        for pattern in (how=='S' ? (:none, :all, :alternate, :second) : (:none,))
            flags=J[
                pattern===:all||pattern===:alternate&&isodd(j)||pattern===:second&&iseven(
                    j
                ) for j in 1:n
            ]
            chosen=if how=='S'
                reduce(
                vcat,
                [
                    collect(j:(j + len - 1)) for
                    (j, len) in layout if any(!iszero, flags[j:(j + len - 1)])
                ];
                init=Int[],
            )
            else
                collect(1:n)
            end
            selected=vcat(J(-777), flags, J(-777))
            sr=Int64.(selected)
            vl=fill(T(777), side=='R' ? 1 : n+3, n+2)
            vr=fill(T(777), side=='L' ? 1 : n+3, n+2)
            if how=='B'
                side!='R'&&(vl[1:n, 1:n].=ql)
                side!='L'&&(vr[1:n, 1:n].=qr_matrix)
            end
            vlr=copy(vl)
            vrr=copy(vr)
            stored=fill(T(NaN), n+3, n)
            for j in 1:n, i in 1:min(n, j + (T <: Real ? 1 : 0))
                stored[i, j]=t[i, j]
            end
            old=copy(stored)
            tr=copy(stored)
            result=direct_schur_vectors!(
                stored, vl, vr, selected, J, Val(blocked); side, how, mm=n+2
            )
            oracle=direct_schur_vectors!(
                tr,
                vlr,
                vrr,
                sr,
                Int64,
                Val(blocked);
                side,
                how,
                mm=n+2,
                pointer=Libdl.dlsym(
                    reference, Symbol(prefix*(blocked ? "trevc3_64_" : "trevc_64_"))
                ),
            )
            @test result.info==result.diagnostic==oracle.info==0
            @test isequal(stored, old)&&isequal(tr, old)
            @test selected==sr
            check_schur_vectors(
                result, vl, vr, t, layout, chosen, side, how, ql, qr_matrix, tol
            )
            check_schur_vectors(
                oracle, vlr, vrr, t, layout, chosen, side, how, ql, qr_matrix, tol
            )
        end
    end
end
