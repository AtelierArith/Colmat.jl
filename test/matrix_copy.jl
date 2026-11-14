using Test, LinearAlgebra, Libdl, Colmat

for (T, p) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    J in (Int32, Int64)

    suffix = J === Int64 ? "64_" : ""
    @eval function direct_matrix_copy!(
        family,
        a::Matrix{$T},
        b::Matrix{$T},
        ::Type{$J};
        uplo,
        m,
        n,
        alpha=($T(-0.0)),
        beta=($T(NaN)),
    )
        f = Libdl.dlsym(
            Libdl.dlopen(Colmat.backend_library()), Symbol($p * family * "_" * $suffix)
        )
        if family == "lacpy"
            ccall(
                f,
                Cvoid,
                (Ref{UInt8}, Ref{$J}, Ref{$J}, Ptr{$T}, Ref{$J}, Ptr{$T}, Ref{$J}),
                uplo,
                m,
                n,
                a,
                size(a, 1),
                b,
                size(b, 1),
            )
        else
            ccall(
                f,
                Cvoid,
                (Ref{UInt8}, Ref{$J}, Ref{$J}, Ref{$T}, Ref{$T}, Ptr{$T}, Ref{$J}),
                uplo,
                m,
                n,
                alpha,
                beta,
                b,
                size(b, 1),
            )
        end
    end
end

@testset "Native LACPY/LASET selected storage and direct ABIs" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64), m in 0:5, n in 0:5, u in ('U', 'L', 'A', '?'),
        family in ("lacpy", "laset")

        guard = T(777)
        a = fill(T(NaN), m+2, n+1)
        b = fill(guard, m+3, n+1)
        expected = copy(b)
        alpha = T <: Complex ? T(-0.0, -0.0) : T(-0.0)
        beta = T <: Complex ? T(NaN, 2) : T(NaN)
        for j in 1:n, i in 1:m
            selected = if u == 'U'
                i <= j
            elseif u == 'L'
                i >= j
            else
                true
            end
            if selected
                value = if i == j
                    T(NaN)
                elseif isodd(i+j)
                    alpha
                else
                    T(i+3j)
                end
                a[i, j] = value
                expected[i, j] = if family == "lacpy"
                    value
                elseif i == j
                    beta
                else
                    alpha
                end
            end
        end
        before = copy(a)
        direct_matrix_copy!(family, a, b, J; uplo=u, m, n, alpha, beta)
        @test isequal(a, before)
        @test isequal(b, expected)
    end
end
