using Test, Colmat, LinearAlgebra, Libdl
isdefined(@__MODULE__, :lapacke288_selected_entry) || include("lapacke_calls.jl")

for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
    @eval function lapacke_matrix_aux_lacpy_call(
        ::Type{$T}, ::Type{$J}, p, layout, uplo, m, n, a, lda, b, ldb, name, route
    )
        lapacke288_selected_entry(name, $J, route) do
            ccall(
                p,
                $J,
                (Cint, Cchar, $J, $J, Ptr{$T}, $J, Ptr{$T}, $J),
                layout,
                uplo,
                m,
                n,
                a,
                lda,
                b,
                ldb,
            )
        end
    end
    @eval function lapacke_matrix_aux_laset_call(
        ::Type{$T}, ::Type{$J}, p, layout, uplo, m, n, alpha::$T, beta::$T, a, lda, name, route
    )
        lapacke288_selected_entry(name, $J, route) do
            ccall(
                p,
                $J,
                (Cint, Cchar, $J, $J, $T, $T, Ptr{$T}, $J),
                layout,
                uplo,
                m,
                n,
                alpha,
                beta,
                a,
                lda,
            )
        end
    end
    @eval function lapacke_matrix_aux_laswp_call(
        ::Type{$T}, ::Type{$J}, p, layout, n, a, lda, k1, k2, ipiv::Vector{$J}, incx, name, route
    )
        lapacke288_selected_entry(name, $J, route) do
            ccall(
                p,
                $J,
                (Cint, $J, Ptr{$T}, $J, $J, $J, Ptr{$J}, $J),
                layout,
                n,
                a,
                lda,
                k1,
                k2,
                ipiv,
                incx,
            )
        end
    end
end

value(::Type{T}, i, j) where {T<:Real} = T((2 + 3i - j) / 7)
value(::Type{T}, i, j) where {T<:Complex} =
    T(complex((2 + 3i - j) / 7, (i - 2j) / 11))
complex_value(::Type{T}, re, im) where {T<:Real} = T(re)
complex_value(::Type{T}, re, im) where {T<:Complex} = T(complex(re, im))

@testset "LAPACKE matrix copy, fill and row-swap storage" begin
    native = Libdl.dlopen(Colmat.backend_library())
    trampoline = Libdl.dlopen(BLAS.libblastrampoline)
    for (T, prefix) in
        ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
        J in (Int32, Int64)

        suffix = J === Int64 ? "64_" : ""
        interface = J === Int64 ? :ilp64 : :lp64
        for family in ("lacpy", "laset", "laswp"), high in (false, true)
            name = "LAPACKE_" * prefix * family * (high ? "" : "_work")
            @test BLAS.lbt_get_forward(name, interface) ==
                Libdl.dlsym(native, name * suffix)
            @test Libdl.dlsym(native, name * suffix) !=
                Libdl.dlsym(trampoline, name * suffix)
        end
        for route in (:dll, :lbt_trampoline),
            layout in (101, 102),
            uplo in ('A', 'U', 'L'),
            m in (0, 1, 3, 5),
            n in (0, 1, 4)

            ld = layout == 101 ? n + 2 : max(1, m) + 2
            at(i, j) = layout == 101 ? i * ld + j : i + j * ld
            selected(i, j) = uplo == 'U' ? i <= j : uplo == 'L' ? i >= j : true
            a = fill(T(NaN), layout == 101 ? max(1, m) * ld + 1 : ld * max(1, n))
            b = fill(T(777), layout == 101 ? max(1, m) * ld + 1 : ld * max(1, n))
            for j in 0:n-1, i in 0:m-1
                a[at(i, j) + 1] = value(T, i, j)
            end
            saved_a, saved_b = copy(a), copy(b)
            for high in (false, true)
                name = "LAPACKE_" * prefix * "lacpy" * (high ? "" : "_work")
                p = Libdl.dlsym(route === :dll ? native : trampoline, name * suffix)
                copyto!(b, saved_b)
                @test lapacke_matrix_aux_lacpy_call(
                    T, J, p, layout, uplo, m, n, a, ld, b, ld, name, route
                ) == 0
                @test isequal(a, saved_a)
                for j in 0:n-1, i in 0:m-1
                    q = at(i, j) + 1
                    if selected(i, j)
                        @test isequal(b[q], a[q])
                    else
                        @test isequal(b[q], saved_b[q])
                    end
                end
                for q in eachindex(b)
                    index = q - 1
                    row = layout == 101 ? index ÷ ld : index % ld
                    col = layout == 101 ? index % ld : index ÷ ld
                    (row < m && col < n && selected(row, col)) || @test isequal(b[q], saved_b[q])
                end
            end
        end

        # LACPY and LASET reject ignored leading dimensions, not selected
        # padding entries. The row-major destination stride is checked before
        # any numerical work, as in the reference wrappers.
        for high in (false, true)
            lacpy = "LAPACKE_" * prefix * "lacpy" * (high ? "" : "_work")
            p = Libdl.dlsym(native, lacpy * suffix)
            a = fill(T(1), 9)
            b = fill(T(2), 9)
            @test lapacke_matrix_aux_lacpy_call(
                T, J, p, 301, 'A', 1, 1, a, 1, b, 1, lacpy, :dll
            ) == -1
            @test lapacke_matrix_aux_lacpy_call(
                T, J, p, 101, 'A', 2, 3, a, 2, b, 3, lacpy, :dll
            ) == -6
            @test lapacke_matrix_aux_lacpy_call(
                T, J, p, 101, 'A', 2, 3, a, 3, b, 2, lacpy, :dll
            ) == -8
            @test lapacke_matrix_aux_lacpy_call(
                T, J, p, 101, '?', 2, 3, a, 3, b, 3, lacpy, :dll
            ) == 0
        end
        # High-level LACPY checks the logical rectangle for NaNs before any
        # leading-dimension or UPLO work.
        lacpy = "LAPACKE_" * prefix * "lacpy"
        p = Libdl.dlsym(native, lacpy * suffix)
        a = fill(T(1), 12)
        b = fill(T(2), 12)
        a[1] = T(NaN)
        @test lapacke_matrix_aux_lacpy_call(
            T, J, p, 102, 'A', 2, 2, a, 3, b, 3, lacpy, :dll
        ) == -5
        a[1] = T(1)
        # NaN outside the logical rectangle is not inspected.
        a[12] = T(NaN)
        @test lapacke_matrix_aux_lacpy_call(
            T, J, p, 102, 'A', 2, 2, a, 3, b, 3, lacpy, :dll
        ) == 0

        for route in (:dll, :lbt_trampoline),
            layout in (101, 102),
            uplo in ('A', 'U', 'L'),
            m in (0, 1, 4),
            n in (0, 1, 3)

            ld = layout == 101 ? n + 2 : max(1, m) + 2
            at(i, j) = layout == 101 ? i * ld + j : i + j * ld
            selected(i, j) = uplo == 'U' ? i <= j : uplo == 'L' ? i >= j : true
            a = fill(T(NaN), layout == 101 ? max(1, m) * ld + 1 : ld * max(1, n))
            saved = copy(a)
            alpha = complex_value(T, -0.0, 1.5)
            beta = complex_value(T, 2.0, 2.0)
            for high in (false, true)
                name = "LAPACKE_" * prefix * "laset" * (high ? "" : "_work")
                p = Libdl.dlsym(route === :dll ? native : trampoline, name * suffix)
                copyto!(a, saved)
                @test lapacke_matrix_aux_laset_call(
                    T, J, p, layout, uplo, m, n, alpha, beta, a, ld, name, route
                ) == 0
                for j in 0:n-1, i in 0:m-1
                    q = at(i, j) + 1
                    if selected(i, j)
                        @test isequal(a[q], i == j ? beta : alpha)
                    else
                        @test isequal(a[q], saved[q])
                    end
                end
            end
        end
        laset = "LAPACKE_" * prefix * "laset"
        p = Libdl.dlsym(native, laset * suffix)
        a = fill(T(1), 9)
        @test lapacke_matrix_aux_laset_call(
            T, J, p, 102, 'A', 2, 2, T(NaN), T(0), a, 2, laset, :dll
        ) == -5
        @test lapacke_matrix_aux_laset_call(
            T, J, p, 102, 'A', 2, 2, T(0), T(NaN), a, 2, laset, :dll
        ) == -6
        @test lapacke_matrix_aux_laset_call(
            T, J, p, 101, 'A', 2, 3, T(0), T(1), a, 2, laset, :dll
        ) == -8

        for route in (:dll, :lbt_trampoline),
            layout in (101, 102),
            n in (1, 2, 5),
            m in (3, 5),
            (k1, k2) in ((1, m), (2, m), (1, 1), (m, m), (3, 2)),
            incx in (-2, -1, 1, 2)

            ld = layout == 101 ? n + 2 : m + 2
            at(i, j) = layout == 101 ? i * ld + j : i + j * ld
            rows = layout == 101 ? m : ld
            a = fill(T(999), layout == 101 ? m * ld : ld * n)
            for j in 0:n-1, i in 0:rows-1
                a[at(i, j) + 1] = value(T, i, j)
            end
            saved = copy(a)
            # A legal interchange: swap row i with its neighbour. IPIV uses
            # Fortran's 1-based indexing at stride abs(INCX) starting at K1.
            ix0 = incx > 0 ? k1 : k1 + (k1 - k2) * incx
            slot(i) = incx > 0 ? ix0 + (i - k1) * incx : ix0 + (k2 - i) * incx
            ipiv = fill(J(1), k1 > k2 ? max(1, k2) : maximum(slot(i) for i in k1:k2))
            for i in k1:k2
                ipiv[slot(i)] = J(i == k2 ? k1 : i + 1)
            end
            expected = copy(a)
            order = incx > 0 ? collect(k1:k2) : reverse(collect(k1:k2))
            for i in order
                pivot = Int(ipiv[slot(i)])
                pivot == i && continue
                for j in 0:n-1
                    q, r = at(i - 1, j) + 1, at(pivot - 1, j) + 1
                    expected[q], expected[r] = expected[r], expected[q]
                end
            end
            for high in (false, true)
                name = "LAPACKE_" * prefix * "laswp" * (high ? "" : "_work")
                p = Libdl.dlsym(route === :dll ? native : trampoline, name * suffix)
                copyto!(a, saved)
                info = lapacke_matrix_aux_laswp_call(
                    T, J, p, layout, n, a, ld, k1, k2, ipiv, incx, name, route
                )
                @test info == 0
                for j in 0:n-1, i in 0:rows-1
                    @test isequal(a[at(i, j) + 1], expected[at(i, j) + 1])
                end
                if layout == 101
                    # Row-major storage keeps entries after column N fixed.
                    for i in 0:m-1, j in n:ld-1
                        @test isequal(a[at(i, j) + 1], saved[at(i, j) + 1])
                    end
                end
            end
        end
        laswp = "LAPACKE_" * prefix * "laswp"
        p = Libdl.dlsym(native, laswp * suffix)
        a = fill(T(1), 9)
        piv = J[1, 2]
        @test lapacke_matrix_aux_laswp_call(
            T, J, p, 101, 3, a, 2, 1, 2, piv, 1, laswp, :dll
        ) == -4
        @test lapacke_matrix_aux_laswp_call(
            T, J, p, 301, 3, a, 2, 1, 2, piv, 1, laswp, :dll
        ) == -1
    end
end
