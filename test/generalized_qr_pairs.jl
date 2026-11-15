using Test, LinearAlgebra, Random, Libdl, Colmat
for (T, prefix) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    J in (Int32, Int64),
    rq in (false, true)

    sym=Symbol(prefix*(rq ? "ggrqf" : "ggqrf")*"_"*(J===Int64 ? "64_" : ""))
    @eval function direct_generalized_pair!(
        ::Val{$rq},
        a::Matrix{$T},
        b::Matrix{$T},
        firstdim,
        seconddim,
        thirddim,
        ::Type{$J};
        library=Colmat.backend_library(),
        lwork=max(1, firstdim, seconddim, thirddim),
    )
        ar, ac, br, bc=if $rq
            (firstdim, thirddim, seconddim, thirddim)
        else
            (firstdim, seconddim, firstdim, thirddim)
        end
        ta=fill($T(777), min(ar, ac)+2)
        tb=fill($T(777), min(br, bc)+2)
        w=fill($T(777), max(1, lwork)+2)
        info=fill($J(-777), 3)
        ccall(
            Libdl.dlsym(Libdl.dlopen(library), $(QuoteNode(sym))),
            Cvoid,
            (
                Ref{$J},
                Ref{$J},
                Ref{$J},
                Ptr{$T},
                Ref{$J},
                Ptr{$T},
                Ptr{$T},
                Ref{$J},
                Ptr{$T},
                Ptr{$T},
                Ref{$J},
                Ptr{$J},
            ),
            firstdim,
            seconddim,
            thirddim,
            a,
            size(a, 1),
            view(ta, 2:(length(ta) - 1)),
            b,
            size(b, 1),
            view(tb, 2:(length(tb) - 1)),
            view(w, 2:(length(w) - 1)),
            lwork,
            view(info, 2:2),
        )
        @test info[[1, 3]]==[$J(-777), $J(-777)]
        @test ta[[1, end]]==tb[[1, end]]==w[[1, end]]==[$T(777), $T(777)]
        (; info=info[2], workspace=real(w[2]), ta=ta[2:(end - 1)], tb=tb[2:(end - 1)])
    end
end
for (T, prefix) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    rq in (false, true)

    name=(T<:Real ? "org" : "ung")*(rq ? "rq" : "qr")
    sym=Symbol(prefix*name*"_64_")
    @eval function generalized_pair_q(
        ::Val{$rq}, a::Matrix{$T}, tau::Vector{$T}, rows, cols
    )
        order=$rq ? cols : rows
        k=min(rows, cols)
        q=fill($T(NaN), max(1, order), order)
        if $rq
            for i in 1:k, j in 1:order
                q[order - k + i, j]=a[rows - k + i, j]
            end
        else
            q[1:rows, 1:k]=a[1:rows, 1:k]
        end
        work=zeros($T, max(1, order))
        info=Ref{Int64}(-777)
        ccall(
            Libdl.dlsym(Libdl.dlopen(generalized_qr_reference()), $(QuoteNode(sym))),
            Cvoid,
            (
                Ref{Int64},
                Ref{Int64},
                Ref{Int64},
                Ptr{$T},
                Ref{Int64},
                Ptr{$T},
                Ptr{$T},
                Ref{Int64},
                Ref{Int64},
            ),
            order,
            order,
            k,
            q,
            max(1, order),
            tau,
            work,
            length(work),
            info,
        )
        @test info[]==0
        q[1:order, :]
    end
end
function generalized_pair_triangular(a, rows, cols, rq)
    r=zeros(eltype(a), rows, cols)
    for j in 1:cols, i in 1:rows
        if rq ? i<=j+rows-cols : i<=j
            r[i, j]=a[i, j]
        end
    end
    return r
end
@testset "GGQRF/GGRQF both ABIs, external Q reconstruction" begin
    rng=MersenneTwister(3142)
    ref=generalized_qr_reference()
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        rq in (false, true),
        firstdim in (0, 1, 3, 6),
        seconddim in (0, 1, 2, 5),
        thirddim in (0, 1, 4, 7),
        mode in (:dense, :zero, :rankone)

        ar, ac, br, bc=if rq
            (firstdim, thirddim, seconddim, thirddim)
        else
            (firstdim, seconddim, firstdim, thirddim)
        end
        ao=if mode==:zero
            zeros(T, ar, ac)
        elseif mode==:rankone
            randn(rng, T, ar)*randn(rng, T, 1, ac)
        else
            randn(rng, T, ar, ac)
        end
        bo=if mode==:zero
            zeros(T, br, bc)
        elseif mode==:rankone
            randn(rng, T, br)*randn(rng, T, 1, bc)
        else
            randn(rng, T, br, bc)
        end
        tol=1000eps(real(T))*max(1, firstdim, seconddim, thirddim)
        for (library, K) in ((Colmat.backend_library(), J), (ref, Int64))
            a=vcat(ao, fill(T(777), 2, ac))
            b=vcat(bo, fill(T(777), 3, bc))
            saved=deepcopy((a, b))
            query=direct_generalized_pair!(
                Val(rq), a, b, firstdim, seconddim, thirddim, K; library, lwork=-1
            )
            @test query.info==0 && query.workspace>=max(1, firstdim, seconddim, thirddim)
            @test isequal((a, b), saved)
            fact=direct_generalized_pair!(
                Val(rq), a, b, firstdim, seconddim, thirddim, K; library
            )
            @test fact.info==0
            @test all(==(T(777)), a[(ar + 1):end, :]) && all(==(T(777)), b[(br + 1):end, :])
            qa=generalized_pair_q(Val(rq), a, fact.ta, ar, ac)
            qb=generalized_pair_q(Val(!rq), b, fact.tb, br, bc)
            ra=generalized_pair_triangular(a, ar, ac, rq)
            rb=generalized_pair_triangular(b, br, bc, !rq)
            @test qa'*qa≈Matrix{T}(I, size(qa, 1), size(qa, 1)) atol=tol rtol=tol
            @test qb'*qb≈Matrix{T}(I, size(qb, 1), size(qb, 1)) atol=tol rtol=tol
            @test (rq ? ra*qa : qa*ra)≈ao atol=tol*max(1, norm(ao)) rtol=tol
            @test (rq ? qb*rb*qa : qa*rb*qb)≈bo atol=tol*max(1, norm(bo)) rtol=tol
        end
    end
end
