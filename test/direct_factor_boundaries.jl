using Test, LinearAlgebra, Libdl, Random, Colmat
isdefined(@__MODULE__, :factor_abi_lu) || include("direct_factor_abi.jl")

@testset "Direct factor ABI rectangular and empty contracts" begin
    Random.seed!(31712)
    dll = Libdl.dlopen(Colmat.backend_library())
    trampoline = Libdl.dlopen(BLAS.libblastrampoline)
    for (T, prefix) in
        ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
        J in (Int32, Int64),
        route in (:dll, :lbt)

        @testset "$T/$J/$route" begin
            entry(name) = Libdl.dlsym(
                route==:dll ? dll : trampoline, prefix*name*(J==Int32 ? "_" : "_64_")
            )
            ig, info = factor_abi_integer(J)
            for family in ("getrf", "getrf2", "getf2"),
                (m, n) in ((6, 3), (3, 6), (0, 0), (0, 3), (3, 0))

                original = randn(T, m, n)
                a = factor_abi_matrix(original)
                pg, piv = factor_abi_integer(J, max(1, min(m, n)))
                saved = copy(a.storage)
                factor_abi_lu(entry(family), J, a.data, m, n, piv, info)
                @test info[1]==0
                k=min(m, n)
                p=collect(1:m)
                for j in 1:k
                    @test j<=piv[j]<=m
                    p[j], p[piv[j]]=p[piv[j]], p[j]
                end
                if k>0
                    lower=tril(a.data[:, 1:k], -1)+Matrix{T}(I, m, k)
                    upper=triu(a.data[1:k, :])
                    @test lower*upper≈original[p, :]
                else
                    @test a.storage==saved
                end
                factor_abi_guards(a)
                factor_abi_integer_guards(pg)
            end
            # Every solve/inverse/driver entry also receives an empty problem,
            # not merely a factorization whose empty branch delegates elsewhere.
            a = factor_abi_matrix(zeros(T, 0, 0))
            b = factor_abi_matrix(zeros(T, 0, 2))
            pg, piv = factor_abi_integer(J)
            wg, work = factor_abi_integer(T)
            for family in ("getrs", "getri", "gesv")
                call(n; trans=('N')) =
                    if family=="getrs"
                        factor_abi_getrs(
                            entry(family), J, a.data, n, b.data, piv, info, trans
                        )
                    elseif family=="getri"
                        factor_abi_getri(entry(family), J, a.data, n, piv, work, 1, info)
                    else
                        factor_abi_gesv(entry(family), J, a.data, n, b.data, piv, info)
                    end
                call(0)
                @test info[1]==0
                call(-1)
                @test info[1]==-(family=="getrs" ? 2 : 1)
                @test factor_abi_diagnostic()==-info[1]
                if family=="getrs"
                    call(0; trans=('?'))
                    @test info[1]==-1
                    @test factor_abi_diagnostic()==1
                end
                factor_abi_guards(a)
                factor_abi_guards(b)
                factor_abi_integer_guards(pg)
                factor_abi_integer_guards(wg)
            end
            for family in ("potrf", "potrf2", "potf2", "potri", "potrs", "posv"),
                uplo in ('L', 'U')

                call(n; u=uplo) = if family in ("potrs", "posv")
                    factor_abi_positive(entry(family), J, a.data, n, b.data, info, u)
                else
                    factor_abi_chol(entry(family), J, a.data, n, info, u)
                end
                call(0)
                @test info[1]==0
                call(-1)
                @test info[1]==-2
                @test factor_abi_diagnostic()==2
                call(0; u=('?'))
                @test info[1]==-1
                @test factor_abi_diagnostic()==1
                factor_abi_guards(a)
                factor_abi_guards(b)
            end
            for family in ("getrf", "getrf2", "getf2", "gesv")
                a = factor_abi_matrix(T[1 2; 2 4])
                b = factor_abi_matrix(ones(T, 2, 2))
                pg, piv = factor_abi_integer(J, 2)
                saved_b=copy(b.storage)
                if family=="gesv"
                    factor_abi_gesv(entry(family), J, a.data, 2, b.data, piv, info)
                    @test b.storage==saved_b
                else
                    factor_abi_lu(entry(family), J, a.data, 2, 2, piv, info)
                end
                @test info[1]==2
                @test factor_abi_diagnostic()==0
                factor_abi_guards(a)
                factor_abi_guards(b)
                factor_abi_integer_guards(pg)
            end
            for family in ("potrf", "potrf2", "potf2"), uplo in ('L', 'U')
                source=randn(T, 4, 4)
                positive=source*source'+I
                a=factor_abi_matrix(positive)
                ignored=[(i, j) for j in 1:4 for i in 1:4 if uplo=='L' ? i<j : i>j]
                for (i, j) in ignored
                    a.data[i, j]=T(NaN)
                end
                factor_abi_chol(entry(family), J, a.data, 4, info, uplo)
                @test info[1]==0
                f=if uplo=='L'
                    Matrix(LowerTriangular(a.data))
                else
                    Matrix(UpperTriangular(a.data))
                end
                @test (uplo=='L' ? f*f' : f'*f)≈positive
                @test all(isnan(a.data[i, j]) for (i, j) in ignored)
                factor_abi_guards(a)
            end
            factor_abi_integer_guards(ig)
        end
    end
end
