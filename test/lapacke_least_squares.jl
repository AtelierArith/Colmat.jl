using Test, Colmat, LinearAlgebra, Libdl, Random
isdefined(@__MODULE__, :direct_gels!) || include("least_squares.jl")
isdefined(@__MODULE__, :lapacke288_selected_entry) || include("lapacke_calls.jl")
for (T, prefix) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    J in (Int32, Int64)

    @eval function lapacke288_gels_call(
        ::Type{$T},
        ::Type{$J},
        high,
        layout,
        trans,
        m,
        n,
        nrhs,
        a,
        lda,
        b,
        ldb,
        work,
        lwork;
        route=:dll,
    )
        name="LAPACKE_"*$prefix*"gels"*(high ? "" : "_work")
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
                    (Cint, Int8, $J, $J, $J, Ptr{$T}, $J, Ptr{$T}, $J),
                    layout,
                    trans,
                    m,
                    n,
                    nrhs,
                    a,
                    lda,
                    b,
                    ldb,
                )
            else
                ccall(
                    pointer,
                    $J,
                    (Cint, Int8, $J, $J, $J, Ptr{$T}, $J, Ptr{$T}, $J, Ptr{$T}, $J),
                    layout,
                    trans,
                    m,
                    n,
                    nrhs,
                    a,
                    lda,
                    b,
                    ldb,
                    work,
                    lwork,
                )
            end
        end
    end
end
function lapacke288_ls_store(x, layout, ld)
    rows, cols=size(x)
    out=fill(eltype(x)(777), max(1, (layout==101 ? rows : cols)*ld)+2)
    for j in 1:cols, i in 1:rows
        out[1 + (layout == 101 ? (i - 1) * ld + j : (j - 1) * ld + i)]=x[i, j]
    end
    return out
end
function lapacke288_ls_load(x, layout, rows, cols, ld)
    return [
        x[1 + (layout == 101 ? (i - 1) * ld + j : (j - 1) * ld + i)] for
        i in 1:rows, j in 1:cols
    ]
end
@testset "GELS C solutions, residuals and independent overwrite contracts" begin
    rng=MersenneTwister(288420)
    reference=only(
        filter(
            l->l.interface==:ilp64 && occursin("openblas", lowercase(l.libname)),
            BLAS.get_config().loaded_libs,
        ),
    ).libname
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        (m, n, r) in (
            (0, 0, 2),
            (0, 3, 2),
            (3, 0, 2),
            (3, 5, 0),
            (5, 3, 2),
            (3, 5, 2),
            (3, 3, 1),
            (1, 1, 1),
        ),
        trans in ('N', T<:Complex ? 'C' : 'T'),
        layout in (101, 102),
        route in (:dll, :lbt_trampoline),
        high in (false, true),
        zero_case in (false, true)

        a0=zero_case ? zeros(T, m, n) : randn(rng, T, m, n)
        inputs=trans=='N' ? m : n
        solutions=trans=='N' ? n : m
        rows=max(m, n)
        rhs=randn(rng, T, inputs, r)
        b0=fill(high ? zero(T) : T(NaN), rows, r)
        b0[1:inputs, :]=rhs
        ar=fill(T(777), max(1, m)+2, n)
        ar[1:m, :]=a0
        br=fill(T(777), max(1, rows)+3, r)
        br[1:inputs, :]=rhs
        _, _, expected_info=direct_gels!(trans, ar, br, m, n, r, Int64; library=reference)
        lda=(layout==101 ? n : m)+4
        ldb=(layout==101 ? r : rows)+5
        aa=lapacke288_ls_store(a0, layout, lda)
        bb=lapacke288_ls_store(b0, layout, ldb)
        saved_a=copy(aa)
        saved_b=copy(bb)
        work=fill(T(779), max(1, min(m, n)+max(min(m, n), r))+2)
        av=view(aa, 2:(length(aa) - 1))
        bv=view(bb, 2:(length(bb) - 1))
        wv=view(work, 2:(length(work) - 1))
        if !high
            @test lapacke288_gels_call(
                T, J, false, layout, trans, m, n, r, av, lda, bv, ldb, wv, -1; route
            )==0
            query=work[2]
            @test isequal(aa, saved_a) && isequal(bb, saved_b)
            work[2]=T(779)
            @test lapacke288_gels_call(
                T, J, false, layout, trans, m, n, r, av, lda, bv, ldb, wv, 0; route
            )==-11
            @test work[2]==query
            @test ccall(
                Libdl.dlsym(
                    Libdl.dlopen(Colmat.backend_library()), :colmat_blas_last_error
                ),
                Cint,
                (),
            )==10
            @test isequal(aa, saved_a) && isequal(bb, saved_b)
        end
        info=lapacke288_gels_call(
            T, J, high, layout, trans, m, n, r, av, lda, bv, ldb, wv, length(work)-2; route
        )
        @test info==expected_info
        a=lapacke288_ls_load(aa, layout, m, n, lda)
        b=lapacke288_ls_load(bb, layout, rows, r, ldb)
        tol=300eps(real(T))
        @test a≈ar[1:m, :] rtol=tol atol=tol
        if info==0
            @test b[1:solutions, :]≈br[1:solutions, :] rtol=tol atol=tol
            if inputs>solutions && r>0
                @test norm(b[(solutions + 1):inputs, :])≈norm(br[(solutions + 1):inputs, :]) rtol=tol atol=tol
            end
            if m>0 && n>0 && r>0 && !zero_case
                op=trans=='N' ? a0 : a0'
                @test norm(op' * (op*b[1:solutions, :]-rhs))<=100tol*max(
                    norm(op)*norm(rhs), one(real(T))
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
        @test work[[1, end]]==T[779, 779]
    end
end
