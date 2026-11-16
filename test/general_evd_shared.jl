using Test, LinearAlgebra, Libdl, Colmat
isdefined(@__MODULE__, :direct_geevx!) || include("geevx_expert_helpers.jl")

@testset "Shared general EVD Schur outputs and extreme units" begin
    for (T, prefix) in
        ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
        J in (Int32, Int64),
        route in (:dll, :lbt)

        R=real(T)
        n=6
        core=T[
            i==j ? i/3 : (i÷2==j÷2 ? (i<j ? -2 : 2) : (3i+7j)%11/17-0.3) for
            i in 1:n, j in 1:n
        ]
        T<:Complex && (core .+= T.(im .* [(5i+3j)%7/13 for i in 1:n, j in 1:n]))
        # Isolation exercises permutation metadata; unequal units exercise scaling.
        core[1, 2:end].=0
        core[2:end, 1].=0
        units=exp2.(R.(-8:3:7))
        core=core .* reshape(units, :, 1) ./ reshape(units, 1, :)
        f=Libdl.dlsym(
            Libdl.dlopen(route==:dll ? Colmat.backend_library() : BLAS.libblastrampoline),
            prefix*"geevx"*(J==Int32 ? "_" : "_64_"),
        )
        for exponent in (R==Float32 ? (0, 100, -140) : (0, 900, -1000)),
            bal in ('N', 'P', 'S', 'B'), left in ('N', 'V'),
            right in ('N', 'V')

            @testset "$T/$J/$route/$exponent/$bal/$left/$right" begin
                scale=exp2(Float64(exponent))
                original=T.(core .* scale)
                a=vcat(original, fill(T(777), 2, n))
                old=copy(a)
                query=direct_geevx!(a, J; bal, left, right, lwork=-1, pointer=f)
                @test query.info==query.diagnostic==0 && isequal(a, old)
                @test query.workspace>=geevx_minimum(T, n, left, right, 'N')
                result=direct_geevx!(a, J; bal, left, right, pointer=f)
                @test result.info==result.diagnostic==0
                @test all(==(T(777)), a[(n + 1):end, :])
                normalized=ComplexF64.(original) ./ scale
                schur=ComplexF64.(a[1:n, :]) ./ scale
                values=T<:Real ? complex.(result.values, result.imaginary) : result.values
                values=ComplexF64.(values) ./ scale
                quantum=Float64(nextfloat(zero(R)))
                tol=2000n*eps(R)+n*quantum/scale
                @test all(isfinite, schur)
                @test norm(tril(schur, T<:Real ? -2 : -1))==0
                # Orthogonal/unitary similarity preserves the balanced Frobenius norm.
                transform=Matrix{T}(I, n, n)
                LAPACK.gebak!(
                    bal, 'R', Int(result.ilo), Int(result.ihi), result.scale, transform
                )
                transform=ComplexF64.(transform)
                balanced=transform\(normalized*transform)
                @test norm(schur)≈norm(balanced) atol=tol*max(1, norm(balanced)) rtol=tol
                @test tr(schur)≈tr(balanced) atol=tol*max(1, norm(balanced)) rtol=tol
                @test result.abnrm/scale≈opnorm(balanced, 1) atol=tol*max(
                    1, opnorm(balanced, 1)
                ) rtol=tol
                j=1
                while j<=n
                    if T<:Real && j<n && !iszero(schur[j + 1, j])
                        @test real(values[j])≈real(schur[j, j]) atol=tol rtol=tol
                        @test real(schur[j, j])≈real(schur[j + 1, j + 1]) atol=tol rtol=tol
                        @test imag(values[j])>0 && values[j + 1]==conj(values[j])
                        @test imag(values[j])≈sqrt(
                            abs(real(schur[j, j + 1])*real(schur[j + 1, j]))
                        ) atol=tol rtol=tol
                        j+=2
                    else
                        @test values[j]≈schur[j, j] atol=tol rtol=tol
                        j+=1
                    end
                end
                for (job, output, op, w) in (
                    (left, result.left, normalized', conj.(values)),
                    (right, result.right, normalized, values),
                )
                    if job=='V'
                        vectors=ComplexF64.(geevx_unpack(output, result.imaginary))
                        @test all(isfinite, vectors)
                        @test all(
                            isapprox(norm(v), 1; atol=100eps(R), rtol=100eps(R)) for
                            v in eachcol(vectors)
                        )
                        @test norm(op*vectors-vectors*Diagonal(w))<=tol*max(
                            1, norm(normalized)*norm(vectors)
                        )
                    else
                        @test all(==(T(777)), output)
                    end
                end
            end
        end
        for invalid in (R(NaN), R(Inf))
            a=Matrix{T}(I, 2, 2)
            a[1, 1]=T(invalid)
            result=direct_geevx!(a, J; pointer=f)
            @test result.info==1 && result.diagnostic==0
        end
    end
end
