using Test, LinearAlgebra, Libdl, Random, Colmat

# All calls below have concrete scalar and integer types. Each test executes both
# the DLL export and the current LBT forward, independently of Julia's BlasInt.
for J in (Int32, Int64), T in (Float32, Float64, ComplexF32, ComplexF64)
    @eval begin
        function factor_abi_lu(f, ::Type{$J}, a::AbstractMatrix{$T}, m, n, piv, info)
            ccall(
                f,
                Cvoid,
                (Ref{$J}, Ref{$J}, Ptr{$T}, Ref{$J}, Ptr{$J}, Ptr{$J}),
                m,
                n,
                a,
                stride(a, 2),
                piv,
                info,
            )
        end
        function factor_abi_getrs(
            f, ::Type{$J}, a::AbstractMatrix{$T}, n, b, piv, info, trans
        )
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
                trans,
                n,
                size(b, 2),
                a,
                stride(a, 2),
                piv,
                b,
                stride(b, 2),
                info,
                1,
            )
        end
        function factor_abi_getri(
            f, ::Type{$J}, a::AbstractMatrix{$T}, n, piv, work, lw, info
        )
            ccall(
                f,
                Cvoid,
                (Ref{$J}, Ptr{$T}, Ref{$J}, Ptr{$J}, Ptr{$T}, Ref{$J}, Ptr{$J}),
                n,
                a,
                stride(a, 2),
                piv,
                work,
                lw,
                info,
            )
        end
        function factor_abi_gesv(f, ::Type{$J}, a::AbstractMatrix{$T}, n, b, piv, info)
            ccall(
                f,
                Cvoid,
                (Ref{$J}, Ref{$J}, Ptr{$T}, Ref{$J}, Ptr{$J}, Ptr{$T}, Ref{$J}, Ptr{$J}),
                n,
                size(b, 2),
                a,
                stride(a, 2),
                piv,
                b,
                stride(b, 2),
                info,
            )
        end
        function factor_abi_chol(f, ::Type{$J}, a::AbstractMatrix{$T}, n, info, uplo)
            ccall(
                f,
                Cvoid,
                (Ref{UInt8}, Ref{$J}, Ptr{$T}, Ref{$J}, Ptr{$J}, Clong),
                uplo,
                n,
                a,
                stride(a, 2),
                info,
                1,
            )
        end
        function factor_abi_positive(f, ::Type{$J}, a::AbstractMatrix{$T}, n, b, info, uplo)
            ccall(
                f,
                Cvoid,
                (
                    Ref{UInt8},
                    Ref{$J},
                    Ref{$J},
                    Ptr{$T},
                    Ref{$J},
                    Ptr{$T},
                    Ref{$J},
                    Ptr{$J},
                    Clong,
                ),
                uplo,
                n,
                size(b, 2),
                a,
                stride(a, 2),
                b,
                stride(b, 2),
                info,
                1,
            )
        end
    end
end

function factor_abi_matrix(a)
    m, n = size(a)
    storage = fill(eltype(a)(-934), m + 5, n + 2)
    data = @view storage[2:(m + 1), 2:(n + 1)]
    data .= a
    return (; storage, data, m, n)
end
function factor_abi_guards(x)
    @test all(
        x.storage[i, j] == eltype(x.storage)(-934) for
        j in axes(x.storage, 2), i in axes(x.storage, 1) if
        !(2 <= i <= x.m + 1 && 2 <= j <= x.n + 1)
    )
end
function factor_abi_sentinel(::Type{T}) where {T}
    return T(T <: Integer && sizeof(T)>4 ? 0x1357913579 : 0x13579)
end
function factor_abi_integer(::Type{J}, n=1) where {J}
    storage=fill(factor_abi_sentinel(J), n+2)
    return storage, @view(storage[2:(n + 1)])
end
function factor_abi_integer_guards(storage)
    @test first(storage)==last(storage)==factor_abi_sentinel(eltype(storage))
end

function factor_abi_diagnostic()
    return ccall(
        Libdl.dlsym(Libdl.dlopen(Colmat.backend_library()), :colmat_blas_last_error),
        Cint,
        (),
    )
end

@testset "Direct LU and Cholesky integer ABI contracts" begin
    Random.seed!(317)
    handle = Libdl.dlopen(Colmat.backend_library())
    for (T, prefix) in
        ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
        J in (Int32, Int64),
        route in (:dll, :lbt)

        @testset "$T/$J/$route" begin
            entry(name) =
                if route == :dll
                    Libdl.dlsym(handle, prefix * name * (J == Int32 ? "_" : "_64_"))
                else
                    Libdl.dlsym(
                        Libdl.dlopen(BLAS.libblastrampoline),
                        prefix * name * (J == Int32 ? "_" : "_64_"),
                    )
                end
            ig, info = factor_abi_integer(J)
            pg, piv = factor_abi_integer(J, 4)
            original = randn(T, 4, 4) + 5I
            rhs = randn(T, 4, 2)
            for family in ("getrf", "getrf2", "getf2")
                a = factor_abi_matrix(original)
                factor_abi_lu(entry(family), J, a.data, 4, 4, piv, info)
                @test info[1] == 0
                p = collect(1:4)
                for k in 1:4
                    @test k <= piv[k] <= 4
                    p[k], p[piv[k]] = p[piv[k]], p[k]
                end
                @test Matrix(UnitLowerTriangular(a.data)) *
                      Matrix(UpperTriangular(a.data)) ≈ original[p, :]
                for trans in ('N', 'T', 'C')
                    b = factor_abi_matrix(rhs)
                    factor_abi_getrs(entry("getrs"), J, a.data, 4, b.data, piv, info, trans)
                    @test info[1] == 0
                    op = if trans == 'N'
                        original
                    elseif trans == 'T'
                        transpose(original)
                    else
                        original'
                    end
                    @test op * b.data ≈ rhs
                    factor_abi_guards(b)
                end
                wg, work = factor_abi_integer(T, 4)
                saved = copy(a.data)
                factor_abi_getri(entry("getri"), J, a.data, 4, piv, work, -1, info)
                @test info[1] == 0 && real(work[1]) >= 4
                @test a.data == saved
                factor_abi_getri(entry("getri"), J, a.data, 4, piv, work, 3, info)
                @test info[1] == -6
                @test factor_abi_diagnostic() == -info[1]
                @test a.data == saved
                factor_abi_getri(entry("getri"), J, a.data, 4, piv, work, 4, info)
                @test info[1] == 0
                @test original * a.data ≈ Matrix{T}(I, 4, 4)
                factor_abi_integer_guards(wg)
                factor_abi_guards(a)
                factor_abi_integer_guards(pg)
                # Invalid dimensions and zero-size calls must not touch payloads.
                saved = copy(a.storage)
                factor_abi_lu(entry(family), J, a.data, -1, 4, piv, info)
                @test info[1] == -1
                @test factor_abi_diagnostic() == -info[1]
                factor_abi_lu(entry(family), J, a.data, 0, 4, piv, info)
                @test info[1] == 0 && a.storage == saved
                if J == Int64
                    factor_abi_lu(entry(family), J, a.data, Int64(1) << 32, 0, piv, info)
                    @test info[1] == -4 && a.storage == saved
                    @test factor_abi_diagnostic() == -info[1]
                end
            end
            a, b = factor_abi_matrix(original), factor_abi_matrix(rhs)
            factor_abi_gesv(entry("gesv"), J, a.data, 4, b.data, piv, info)
            @test info[1] == 0
            @test original * b.data ≈ rhs
            factor_abi_guards(a)
            factor_abi_guards(b)
            for uplo in ('L', 'U'), family in ("potrf", "potrf2", "potf2")
                positive = original * original' + I
                a = factor_abi_matrix(positive)
                factor_abi_chol(entry(family), J, a.data, 4, info, uplo)
                @test info[1] == 0
                f = if uplo == 'L'
                    Matrix(LowerTriangular(a.data))
                else
                    Matrix(UpperTriangular(a.data))
                end
                @test (uplo == 'L' ? f * f' : f' * f) ≈ positive
                b = factor_abi_matrix(rhs)
                factor_abi_positive(entry("potrs"), J, a.data, 4, b.data, info, uplo)
                @test info[1] == 0
                @test positive * b.data ≈ rhs
                factor_abi_chol(entry("potri"), J, a.data, 4, info, uplo)
                @test info[1] == 0
                @test positive * Matrix(Hermitian(a.data, Symbol(uplo))) ≈
                    Matrix{T}(I, 4, 4)
                factor_abi_guards(a)
                factor_abi_guards(b)
                saved = copy(a.storage)
                factor_abi_chol(entry(family), J, a.data, 4, info, '?')
                @test info[1] == -1 && a.storage == saved
                @test factor_abi_diagnostic() == -info[1]
                factor_abi_chol(entry(family), J, a.data, 0, info, uplo)
                @test info[1] == 0 && a.storage == saved
                bad = factor_abi_matrix(T[1 2; 2 1])
                factor_abi_chol(entry(family), J, bad.data, 2, info, uplo)
                @test info[1] == 2
                factor_abi_guards(bad)
                a, b = factor_abi_matrix(positive), factor_abi_matrix(rhs)
                factor_abi_positive(entry("posv"), J, a.data, 4, b.data, info, uplo)
                @test info[1] == 0
                @test positive * b.data ≈ rhs
                factor_abi_guards(a)
                factor_abi_guards(b)
            end
            factor_abi_integer_guards(ig)
            factor_abi_integer_guards(pg)
        end
    end
end
