using Test, Colmat, LinearAlgebra, Libdl, Random
isdefined(@__MODULE__, :lapacke288_ls_store) || include("lapacke_least_squares.jl")
for (T, prefix) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    J in (Int32, Int64)

    R=real(T)
    @eval function lapacke288_gelsy_call(
        ::Type{$T},
        ::Type{$J},
        high,
        layout,
        m,
        n,
        r,
        a,
        lda,
        b,
        ldb,
        piv,
        rcond,
        rank,
        work,
        lwork,
        rwork;
        route=:dll,
    )
        name="LAPACKE_"*$prefix*"gelsy"*(high ? "" : "_work")
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
                    (Cint, $J, $J, $J, Ptr{$T}, $J, Ptr{$T}, $J, Ptr{$J}, $R, Ptr{$J}),
                    layout,
                    m,
                    n,
                    r,
                    a,
                    lda,
                    b,
                    ldb,
                    piv,
                    rcond,
                    rank,
                )
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
                        Ptr{$J},
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
                    piv,
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
                        Ptr{$J},
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
                    piv,
                    rcond,
                    rank,
                    work,
                    lwork,
                )
            end
        end
    end
end
@testset "GELSY C minimum-norm solutions, pivots and rank contracts" begin
    rng=MersenneTwister(288421)
    reference=only(
        filter(
            l->l.interface==:ilp64 && occursin("openblas", lowercase(l.libname)),
            BLAS.get_config().loaded_libs,
        ),
    ).libname
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        (m, n, r) in
        ((0, 0, 2), (0, 3, 2), (3, 0, 2), (3, 5, 0), (5, 3, 2), (3, 5, 2), (3, 3, 1)),
        mode in (:full, :zero, :one),
        fixed in (:none, :some, :all),
        layout in (101, 102),
        route in (:dll, :lbt_trampoline),
        high in (false, true)

        R=real(T)
        Wide=promote_type(T, Float64)
        rcond=R(1e-3)
        rows=max(m, n)
        a0=if mode==:zero
            zeros(T, m, n)
        elseif mode==:one
            randn(rng, T, m)*randn(rng, T, n)'
        else
            randn(rng, T, m, n)
        end
        rhs=randn(rng, T, m, r)
        b0=fill(high ? zero(T) : T(NaN), rows, r)
        b0[1:m, :]=rhs
        flags=zeros(J, n)
        if fixed==:all
            flags.=J(-3)
        elseif fixed==:some && n>0
            flags[min(2, n)]=J(-7)
        end
        ar=fill(T(777), max(1, m)+2, n)
        ar[1:m, :]=a0
        br=fill(T(777), max(1, rows)+3, r)
        br[1:rows, :]=b0
        expected=direct_gelsy!(ar, br, m, n, r, Int64.(flags); rcond, library=reference)
        own_a=fill(T(777), max(1, m)+2, n)
        own_a[1:m, :]=a0
        own_b=fill(T(777), max(1, rows)+3, r)
        own_b[1:rows, :]=b0
        own=direct_gelsy!(own_a, own_b, m, n, r, copy(flags); rcond)
        lda=(layout==101 ? n : m)+4
        ldb=(layout==101 ? r : rows)+5
        aa=lapacke288_ls_store(a0, layout, lda)
        bb=lapacke288_ls_store(b0, layout, ldb)
        saved_a=copy(aa)
        saved_b=copy(bb)
        pivot=vcat(J(777), flags, J(777))
        saved_pivot=copy(pivot)
        rank=J[777, 778, 777]
        work=fill(T(779), 256)
        rwork=fill(R(780), max(1, 2n)+2)
        av=view(aa, 2:(length(aa) - 1))
        bv=view(bb, 2:(length(bb) - 1))
        pv=view(pivot, 2:(n + 1))
        rv=view(rank, 2:2)
        wv=view(work, 2:255)
        rw=view(rwork, 2:(length(rwork) - 1))
        if !high
            @test lapacke288_gelsy_call(
                T,
                J,
                false,
                layout,
                m,
                n,
                r,
                av,
                lda,
                bv,
                ldb,
                pv,
                rcond,
                rv,
                wv,
                -1,
                rw;
                route,
            )==0
            query=work[2]
            @test isequal(aa, saved_a) &&
                isequal(bb, saved_b) &&
                pivot==saved_pivot &&
                rank==J[777, 778, 777]
            work[2]=T(779)
            @test lapacke288_gelsy_call(
                T,
                J,
                false,
                layout,
                m,
                n,
                r,
                av,
                lda,
                bv,
                ldb,
                pv,
                rcond,
                rv,
                wv,
                0,
                rw;
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
            @test isequal(aa, saved_a) &&
                isequal(bb, saved_b) &&
                pivot==saved_pivot &&
                rank==J[777, 778, 777]
        end
        info=lapacke288_gelsy_call(
            T, J, high, layout, m, n, r, av, lda, bv, ldb, pv, rcond, rv, wv, 254, rw; route
        )
        @test info==own.info==expected.info==0
        @test rank[2]==own.rank==expected.rank
        a=lapacke288_ls_load(aa, layout, m, n, lda)
        b=lapacke288_ls_load(bb, layout, rows, r, ldb)
        @test a≈own_a[1:m, :] rtol=300eps(R) atol=300eps(R)
        @test pivot[2:(n + 1)]==own.piv
        if min(m, n)==0 || r==0
            @test isequal(aa, saved_a) && isequal(bb, saved_b) && pivot==saved_pivot
        else
            @test sort(pivot[2:(n + 1)])==collect(1:n)
            fixed_columns=findall(!iszero, flags)
            @test pivot[2:(1 + length(fixed_columns))]==fixed_columns
            wide_a=Matrix{Wide}(a0)
            wide_b=zeros(Wide, rows, r)
            wide_b[1:m, :]=rhs
            wide=direct_gelsy!(
                wide_a, wide_b, m, n, r, Int64.(flags); rcond, library=reference
            )
            @test wide.info==0 && wide.rank==expected.rank
            singular=svdvals(Matrix{Wide}(a0))
            condition=expected.rank>0 ? singular[1]/singular[expected.rank] : 1.0
            tol=300eps(R)
            forward=max(tol, 20eps(R)*condition)
            @test b[1:n, :]≈br[1:n, :] rtol=forward atol=tol
            @test b[1:n, :]≈wide_b[1:n, :] rtol=forward atol=tol
            @test br[1:n, :]≈wide_b[1:n, :] rtol=forward atol=tol
            if mode!=:zero
                x=b[1:n, :]
                @test norm(a0'*(a0*x-rhs))<=200eps(R)*max(
                    norm(a0)*(norm(a0)*norm(x)+norm(rhs)), floatmin(R)
                )
            end
        end
        for (x, old, nrows, ncols, ld) in
            ((aa, saved_a, m, n, lda), (bb, saved_b, rows, r, ldb))
            touched=Set(
                1+(layout==101 ? (i-1)*ld+j : (j-1)*ld+i) for j in 1:ncols for i in 1:nrows
            )
            @test all(isequal(x[i], old[i]) for i in eachindex(x) if !(i in touched))
        end
        @test rank[[1, end]]==J[777, 777] && pivot[[1, end]]==J[777, 777]
        @test work[[1, end]]==T[779, 779] && rwork[[1, end]]==R[780, 780]
    end
end
