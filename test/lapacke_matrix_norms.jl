using Test, Colmat, LinearAlgebra, Libdl
isdefined(@__MODULE__, :lapacke288_selected_entry) || include("lapacke_calls.jl")

for (T, R) in ((Float32, Float32), (Float64, Float64), (ComplexF32, Float32), (ComplexF64, Float64)),
    J in (Int32, Int64)
    @eval function lapacke_norm_call_lansy(
        ::Type{$T}, ::Type{$J}, pointer, layout, norm, uplo, n, a, lda, work,
        name, route,
    )
        lapacke288_selected_entry(name, $J, route) do
            ccall(
                pointer,
                $R,
                (Cint, Cchar, Cchar, $J, Ptr{$T}, $J, Ptr{$R}),
                layout, norm, uplo, n, a, lda, work,
            )
        end
    end
    @eval function lapacke_norm_call_lantr(
        ::Type{$T}, ::Type{$J}, pointer, layout, norm, uplo, diag, m, n, a,
        lda, work, name, route,
    )
        lapacke288_selected_entry(name, $J, route) do
            ccall(
                pointer,
                $R,
                (Cint, Cchar, Cchar, Cchar, $J, $J, Ptr{$T}, $J, Ptr{$R}),
                layout, norm, uplo, diag, m, n, a, lda, work,
            )
        end
    end
    @eval function lapacke_norm_call_dense(
        ::Type{$T}, ::Type{$J}, pointer, layout, norm, m, n, a, lda, work, name, route,
    )
        lapacke288_selected_entry(name, $J, route) do
            ccall(
                pointer,
                $R,
                (Cint, Cchar, $J, $J, Ptr{$T}, $J, Ptr{$R}),
                layout, norm, m, n, a, lda, work,
            )
        end
    end
end

scalar(::Type{T}, re, im) where {T<:Real} = T(re)
scalar(::Type{T}, re, im) where {T<:Complex} = T(complex(re, im))

function dense_expected(A, norm)
    norm = uppercase(norm)
    if norm == 'M'
        return maximum(abs.(A); init=zero(real(eltype(A))))
    elseif norm == '1' || norm == 'O'
        return maximum(sum(abs.(A); dims=1); init=zero(real(eltype(A))))
    elseif norm == 'I'
        return maximum(sum(abs.(A); dims=2); init=zero(real(eltype(A))))
    else
        return sqrt(sum(abs2.(A)))
    end
end

function triangle_expected(A, uplo, diag, norm)
    m, n = size(A)
    selected = falses(m, n)
    for j in 1:n, i in 1:m
        inside = uplo == 'U' ? i <= j : i >= j
        selected[i, j] = inside
    end
    values = zeros(real(eltype(A)), m, n)
    for j in 1:n, i in 1:m
        if selected[i, j]
            values[i, j] = (diag == 'U' && i == j) ? one(real(eltype(A))) : abs(A[i, j])
        end
    end
    norm = uppercase(norm)
    if norm == 'M'
        return maximum(values)
    elseif norm == '1' || norm == 'O'
        return maximum(vec(sum(values; dims=1)))
    elseif norm == 'I'
        return maximum(vec(sum(values; dims=2)))
    else
        return sqrt(sum(abs2, values))
    end
end

@testset "LAPACKE matrix norms and row-major transposes" begin
    native = Libdl.dlopen(Colmat.backend_library())
    trampoline = Libdl.dlopen(BLAS.libblastrampoline)
    for (T, R, prefix) in
        ((Float32, Float32, "s"), (Float64, Float64, "d"),
         (ComplexF32, Float32, "c"), (ComplexF64, Float64, "z")),
        J in (Int32, Int64), layout in (101, 102),
        family in ("lange", "lansy", "lantr"), high in (false, true),
        route in (:dll, :lbt_trampoline), norm in ('M', '1', 'O', 'I', 'F', 'E')

        suffix = J === Int64 ? "64_" : ""
        interface = J === Int64 ? :ilp64 : :lp64
        name = "LAPACKE_" * prefix * family * (high ? "" : "_work")
        @test BLAS.lbt_get_forward(name, interface) ==
            Libdl.dlsym(native, name * suffix)
        pointer = Libdl.dlsym(route === :dll ? native : trampoline, name * suffix)
        m, n = family == "lansy" ? (4, 4) : (3, 4)
        uplo = family == "lange" ? 'A' : (isodd(m + n) ? 'U' : 'L')
        diag = family == "lantr" ? (isodd(m) ? 'N' : 'U') : 'N'
        ld = layout == 101 ? n + 2 : max(1, m) + 2
        R0 = real(T)
        A = fill(T(NaN), layout == 101 ? max(1, m) * ld : ld * max(1, n))
        at(i, j) = layout == 101 ? (i - 1) * ld + j : i + (j - 1) * ld
        logical = zeros(T, m, n)
        for j in 1:n, i in 1:m
            selected = family == "lange" ? true :
                family == "lansy" ? (uplo == 'U' ? i <= j : i >= j) :
                (uplo == 'U' ? i <= j : i >= j)
            if selected
                value = if family == "lantr" && diag == 'U' && i == j
                    T(999)
                else
                    scalar(T, (2 + i + 3j) / 7, (i - 2j) / 9)
                end
                logical[i, j] = value
                A[at(i, j)] = value
            elseif family == "lansy"
                mirror = uplo == 'U' ? (j, i) : (j, i)
                A[at(i, j)] = logical[mirror[1], mirror[2]] == 0 ? T(NaN) : T(NaN)
            end
        end
        if family == "lansy"
            # Build the full symmetric logical matrix from the selected triangle.
            for j in 1:n, i in 1:m
                if (uplo == 'U' && i > j) || (uplo == 'L' && i < j)
                    logical[i, j] = logical[j, i]
                end
            end
        end
        before = copy(A)
        work = zeros(R0, 16)
        expected = family == "lange" ? dense_expected(logical, norm) :
            family == "lansy" ? dense_expected(logical, norm) :
            triangle_expected(logical, uplo, diag, norm)
        result = if family == "lange"
            lapacke_norm_call_dense(
                T, J, pointer, layout, norm, m, n, A, ld, work,
                name, route,
            )
        elseif family == "lansy"
            lapacke_norm_call_lansy(
                T, J, pointer, layout, norm, uplo, n, A, ld, work,
                name, route,
            )
        else
            lapacke_norm_call_lantr(
                T, J, pointer, layout, norm, uplo, diag, m, n, A, ld, work,
                name, route,
            )
        end
        @test isapprox(result, expected; rtol=50 * eps(R0), atol=50 * eps(R0))
        for q in eachindex(A)
            index = q - 1
            row = layout == 101 ? index ÷ ld : index % ld
            col = layout == 101 ? index % ld : index ÷ ld
            selected = family != "lange" && ((uplo == 'U' && row < col) || (uplo == 'L' && row > col))
            if row >= m || col >= n || !selected
                @test isequal(A[q], before[q])
            end
        end
    end
end

@testset "LAPACKE matrix norm diagnostics" begin
    native = Libdl.dlopen(Colmat.backend_library())
    for (T, R, prefix) in
        ((Float64, Float64, "d"), (ComplexF64, Float64, "z")), J in (Int32, Int64)
        suffix = J === Int64 ? "64_" : ""
        A = zeros(T, 16)
        work = zeros(R, 16)
        for (family, high, arg, expected) in (
            ("lange", true, :nan, -5.0),
            ("lansy", true, :nan, -5.0),
            ("lantr", true, :nan, -7.0),
            ("lange", false, :ld, -6.0),
            ("lansy", false, :ld, -6.0),
            ("lantr", false, :ld, -8.0),
        )
            name = "LAPACKE_" * prefix * family * (high ? "" : "_work")
            pointer = Libdl.dlsym(native, name * suffix)
            data = copy(A)
            arg == :nan && (data[1] = T(NaN))
            ld = arg == :ld ? 1 : 4
            result = if family == "lange"
                lapacke_norm_call_dense(
                    T, J, pointer, 101, 'M', 3, 3, data, ld, work, name, :dll
                )
            elseif family == "lansy"
                lapacke_norm_call_lansy(
                    T, J, pointer, 101, 'M', 'U', 3, data, ld, work, name, :dll
                )
            else
                lapacke_norm_call_lantr(
                    T, J, pointer, 101, 'M', 'U', 'N', 3, 3, data, ld, work,
                    name, :dll,
                )
            end
            @test result == expected
        end
        for family in ("lange", "lansy", "lantr")
            name = "LAPACKE_" * prefix * family
            pointer = Libdl.dlsym(native, name * suffix)
            result = if family == "lange"
                lapacke_norm_call_dense(
                    T, J, pointer, 301, 'M', 3, 3, A, 3, work, name, :dll
                )
            elseif family == "lansy"
                lapacke_norm_call_lansy(
                    T, J, pointer, 301, 'M', 'U', 3, A, 3, work, name, :dll
                )
            else
                lapacke_norm_call_lantr(
                    T, J, pointer, 301, 'M', 'U', 'N', 3, 3, A, 3, work,
                    name, :dll,
                )
            end
            @test result == -1.0
        end
    end
end
