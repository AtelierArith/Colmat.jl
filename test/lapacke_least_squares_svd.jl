using Test, Colmat, LinearAlgebra, Libdl, Random
isdefined(@__MODULE__, :lapacke288_gelsy_call) || include("lapacke_least_squares_rank.jl")
for (T, prefix) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    J in (Int32, Int64),
    family in (:gelss, :gelsd)

    R=real(T)
    @eval function lapacke288_svd_ls_call(
        ::Type{$T},
        ::Type{$J},
        ::Val{$(QuoteNode(family))},
        high,
        layout,
        m,
        n,
        r,
        a,
        lda,
        b,
        ldb,
        s,
        rcond,
        rank,
        work,
        lwork,
        rwork,
        iwork;
        route=:dll,
    )
        name="LAPACKE_"*$prefix*$(String(family))*(high ? "" : "_work")
        symbol=name*($J===Int64 ? "64_" : "")
        native=Libdl.dlopen(Colmat.backend_library())
        @test BLAS.lbt_get_forward(name, $J===Int64 ? :ilp64 : :lp64)==Libdl.dlsym(
            native, symbol
        )
        pointer=Libdl.dlsym(
            route==:dll ? native : Libdl.dlopen(BLAS.libblastrampoline), symbol
        )
        lapacke288_selected_entry(name, $J, route) do
            if high
                ccall(
                    pointer,
                    $J,
                    (Cint, $J, $J, $J, Ptr{$T}, $J, Ptr{$T}, $J, Ptr{$R}, $R, Ptr{$J}),
                    layout,
                    m,
                    n,
                    r,
                    a,
                    lda,
                    b,
                    ldb,
                    s,
                    rcond,
                    rank,
                )
            elseif $(family==:gelsd)
                if $T<:Complex
                    ccall(
                        pointer,
                        $J,
                        (
                            Cint,
                            $J,
                            $J,
                            $J,
                            Ptr{$T},
                            $J,
                            Ptr{$T},
                            $J,
                            Ptr{$R},
                            $R,
                            Ptr{$J},
                            Ptr{$T},
                            $J,
                            Ptr{$R},
                            Ptr{$J},
                        ),
                        layout,
                        m,
                        n,
                        r,
                        a,
                        lda,
                        b,
                        ldb,
                        s,
                        rcond,
                        rank,
                        work,
                        lwork,
                        rwork,
                        iwork,
                    )
                else
                    ccall(
                        pointer,
                        $J,
                        (
                            Cint,
                            $J,
                            $J,
                            $J,
                            Ptr{$T},
                            $J,
                            Ptr{$T},
                            $J,
                            Ptr{$R},
                            $R,
                            Ptr{$J},
                            Ptr{$T},
                            $J,
                            Ptr{$J},
                        ),
                        layout,
                        m,
                        n,
                        r,
                        a,
                        lda,
                        b,
                        ldb,
                        s,
                        rcond,
                        rank,
                        work,
                        lwork,
                        iwork,
                    )
                end
            elseif $T<:Complex
                ccall(
                    pointer,
                    $J,
                    (
                        Cint,
                        $J,
                        $J,
                        $J,
                        Ptr{$T},
                        $J,
                        Ptr{$T},
                        $J,
                        Ptr{$R},
                        $R,
                        Ptr{$J},
                        Ptr{$T},
                        $J,
                        Ptr{$R},
                    ),
                    layout,
                    m,
                    n,
                    r,
                    a,
                    lda,
                    b,
                    ldb,
                    s,
                    rcond,
                    rank,
                    work,
                    lwork,
                    rwork,
                )
            else
                ccall(
                    pointer,
                    $J,
                    (
                        Cint,
                        $J,
                        $J,
                        $J,
                        Ptr{$T},
                        $J,
                        Ptr{$T},
                        $J,
                        Ptr{$R},
                        $R,
                        Ptr{$J},
                        Ptr{$T},
                        $J,
                    ),
                    layout,
                    m,
                    n,
                    r,
                    a,
                    lda,
                    b,
                    ldb,
                    s,
                    rcond,
                    rank,
                    work,
                    lwork,
                )
            end
        end
    end
end
@testset "GELSS/GELSD C singular values, rank and minimum-norm solutions" begin
    rng=MersenneTwister(288422)
    reference=only(
        filter(
            l->l.interface==:ilp64 && occursin("openblas", lowercase(l.libname)),
            BLAS.get_config().loaded_libs,
        ),
    ).libname
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        family in (:gelss, :gelsd),
        (m, n, r) in
        ((0, 0, 2), (0, 3, 2), (3, 0, 2), (3, 5, 0), (5, 3, 2), (3, 5, 2), (3, 3, 1)),
        mode in (:full, :zero, :one),
        exponent in (-1, 0, 1),
        layout in (101, 102),
        route in (:dll, :lbt_trampoline),
        high in (false, true)

        R=real(T)
        Wide=promote_type(T, Float64)
        k=min(m, n)
        rows=max(m, n)
        rcond=R(1e-3)
        scale=R(2)^((R===Float32 ? 40 : 400)*exponent)
        unscaled=if mode==:zero
            zeros(T, m, n)
        elseif mode==:one
            randn(rng, T, m)*randn(rng, T, n)'
        else
            randn(rng, T, m, n)
        end
        rhs0=randn(rng, T, m, r)
        a0=unscaled*scale
        rhs=rhs0*scale
        b0=fill(high ? zero(T) : T(NaN), rows, r)
        b0[1:m, :]=rhs
        ar=fill(T(777), max(1, m)+2, n)
        ar[1:m, :]=a0
        br=fill(T(777), max(1, rows)+3, r)
        br[1:rows, :]=b0
        solver=family==:gelss ? direct_gelss! : direct_gelsd!
        # The executed NRHS=0 OpenBLAS GELSD defect has a separate MWE.
        # Keep these legal cases, using independent GELSS for S and RANK.
        reference_solver=family==:gelsd && r==0 ? direct_gelss! : solver
        expected=reference_solver(ar, br, m, n, r, Int64; rcond, library=reference)
        own_a=fill(T(777), max(1, m)+2, n)
        own_a[1:m, :]=a0
        own_b=fill(T(777), max(1, rows)+3, r)
        own_b[1:rows, :]=b0
        own=solver(own_a, own_b, m, n, r, J; rcond)
        lda=(layout==101 ? n : m)+4
        ldb=(layout==101 ? r : rows)+5
        aa=lapacke288_ls_store(a0, layout, lda)
        bb=lapacke288_ls_store(b0, layout, ldb)
        saved_a=copy(aa)
        saved_b=copy(bb)
        singular=fill(R(782), k+2)
        rank=J[777, 778, 777]
        work=fill(T(779), 3)
        rwork=fill(R(780), 3)
        iwork=fill(J(781), 3)
        av=view(aa, 2:(length(aa) - 1))
        bv=view(bb, 2:(length(bb) - 1))
        sv=view(singular, 2:(k + 1))
        rv=view(rank, 2:2)
        if !high
            saved=deepcopy((aa, bb, singular, rank))
            @test lapacke288_svd_ls_call(
                T,
                J,
                Val(family),
                false,
                layout,
                m,
                n,
                r,
                av,
                lda,
                bv,
                ldb,
                sv,
                rcond,
                rv,
                view(work, 2:2),
                -1,
                view(rwork, 2:2),
                view(iwork, 2:2);
                route,
            )==0
            @test isequal((aa, bb, singular, rank), saved)
            query=work[2]
            real_query=rwork[2]
            integer_query=iwork[2]
            @test real(query)>=1
            if family==:gelsd
                @test integer_query>=1
                if T<:Complex
                    @test real_query>=1
                end
            end
            work[2]=T(779)
            @test lapacke288_svd_ls_call(
                T,
                J,
                Val(family),
                false,
                layout,
                m,
                n,
                r,
                av,
                lda,
                bv,
                ldb,
                sv,
                rcond,
                rv,
                view(work, 2:2),
                0,
                view(rwork, 2:2),
                view(iwork, 2:2);
                route,
            )==-13
            @test work[2]==query
            @test ccall(
                Libdl.dlsym(
                    Libdl.dlopen(Colmat.backend_library()), :colmat_blas_last_error
                ),
                Cint,
                (),
            )==12
            @test isequal((aa, bb, singular, rank), saved)
            if family==:gelsd
                @test iwork[2]==integer_query
                if T<:Complex
                    @test rwork[2]==real_query
                end
            end
            @test work[[1, end]]==T[779, 779] &&
                rwork[[1, end]]==R[780, 780] &&
                iwork[[1, end]]==J[781, 781]
            nw=ceil(Int, real(query))
            nr=T<:Complex ? (family==:gelsd ? ceil(Int, real_query) : max(1, 5k)) : 1
            ni=family==:gelsd ? Int(integer_query) : 1
            work=fill(T(779), nw+2)
            rwork=fill(R(780), nr+2)
            iwork=fill(J(781), ni+2)
        end
        info=lapacke288_svd_ls_call(
            T,
            J,
            Val(family),
            high,
            layout,
            m,
            n,
            r,
            av,
            lda,
            bv,
            ldb,
            sv,
            rcond,
            rv,
            view(work, 2:(length(work) - 1)),
            length(work)-2,
            view(rwork, 2:(length(rwork) - 1)),
            view(iwork, 2:(length(iwork) - 1));
            route,
        )
        @test info==own.info==expected.info==0
        @test rank[2]==own.rank==expected.rank
        a=lapacke288_ls_load(aa, layout, m, n, lda)
        b=lapacke288_ls_load(bb, layout, rows, r, ldb)
        tol=300eps(R)
        @test a≈own_a[1:m, :] rtol=tol atol=tol
        @test singular[2:(k + 1)]/scale≈expected.s/scale rtol=tol atol=tol
        if k==0
            @test all(iszero, b)
        elseif r>0
            wide_a=Matrix{Wide}(a0)
            wide_b=zeros(Wide, rows, r)
            wide_b[1:m, :]=rhs
            wide=solver(wide_a, wide_b, m, n, r, Int64; rcond, library=reference)
            @test wide.info==0 && wide.rank==expected.rank
            condition=if expected.rank>0
                (expected.s[1]/scale)/(expected.s[expected.rank]/scale)
            else
                1.0
            end
            forward=max(tol, 20eps(R)*condition)
            @test b[1:n, :]≈br[1:n, :] rtol=forward atol=tol
            @test b[1:n, :]≈wide_b[1:n, :] rtol=forward atol=tol
            @test br[1:n, :]≈wide_b[1:n, :] rtol=forward atol=tol
            if mode!=:zero
                x=b[1:n, :]
                # RCOND truncation permits a discarded singular subspace.
                # Bound its full-gradient contribution separately from rounding.
                discarded=expected.rank<k ? expected.s[expected.rank + 1]/scale : zero(R)
                rounding=200eps(R)*max(
                    norm(unscaled)*(norm(unscaled)*norm(x)+norm(rhs0)), floatmin(R)
                )
                @test norm(unscaled'*(unscaled*x-rhs0))<=discarded*norm(rhs0)+rounding
            end
        end
        for (x, old, nrows, ncols, ld) in
            ((aa, saved_a, m, n, lda), (bb, saved_b, rows, r, ldb))
            touched=Set(
                1+(layout==101 ? (i-1)*ld+j : (j-1)*ld+i) for j in 1:ncols for i in 1:nrows
            )
            @test all(isequal(x[i], old[i]) for i in eachindex(x) if !(i in touched))
        end
        @test rank[[1, end]]==J[777, 777] && singular[[1, end]]==R[782, 782]
        @test work[[1, end]]==T[779, 779] &&
            rwork[[1, end]]==R[780, 780] &&
            iwork[[1, end]]==J[781, 781]
    end
end
