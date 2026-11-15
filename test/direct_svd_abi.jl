using Test, LinearAlgebra, Libdl, Random, Colmat
isdefined(@__MODULE__, :factor_abi_matrix) || include("direct_factor_abi.jl")

for J in (Int32, Int64),
    T in (Float32, Float64, ComplexF32, ComplexF64),
    family in (:gesvd, :gesdd)

    R = real(T)
    signature = Any[Ref{UInt8}]
    args = Any[:job]
    if family == :gesvd
        push!(signature, Ref{UInt8})
        push!(args, :right)
    end
    append!(
        signature,
        [
            Ref{J},
            Ref{J},
            Ptr{T},
            Ref{J},
            Ptr{R},
            Ptr{T},
            Ref{J},
            Ptr{T},
            Ref{J},
            Ptr{T},
            Ref{J},
        ],
    )
    append!(
        args,
        [
            :m,
            :n,
            :a,
            :(stride(a, 2)),
            :s,
            :u,
            :(stride(u, 2)),
            :vt,
            :(stride(vt, 2)),
            :work,
            :lw,
        ],
    )
    if T <: Complex
        push!(signature, Ptr{R})
        push!(args, :rwork)
    end
    if family == :gesdd
        push!(signature, Ptr{J})
        push!(args, :iwork)
    end
    append!(signature, [Ptr{J}, Clong])
    append!(args, [:info, 1])
    if family == :gesvd
        push!(signature, Clong)
        push!(args, 1)
    end
    @eval function direct_svd_abi(
        f,
        ::Val{$(QuoteNode(family))},
        ::Type{$J},
        a::AbstractMatrix{$T},
        m,
        n,
        s,
        u,
        vt,
        work,
        lw,
        rwork,
        iwork,
        info,
        job,
        right,
    )
        ccall(f, Cvoid, ($(signature...),), $(args...))
    end
end

# LAPACK's documented minimum work; reference GESVD/GESDD receives the same
# minimum so adapter-specific query recommendations cannot define the oracle.
function direct_svd_minimum(T, family, job, m, n)
    k, big = min(m, n), max(m, n)
    family == :gesvd && return max(1, T <: Complex ? 2k + big : max(3k + big, 5k))
    if T <: Complex
        return max(
            1,
            if job == 'N'
                2k + big
            elseif job == 'O'
                2k*k + 2k + big
            elseif job == 'S'
                k*k + 3k
            else
                k*k + 2k + big
            end,
        )
    end
    return max(
        1,
        if job == 'N'
            3k + max(big, 7k)
        elseif job == 'O'
            3k + max(big, 5k*k + 4k)
        elseif job == 'S'
            4k*k + 7k
        else
            4k*k + 6k + big
        end,
    )
end

function direct_svd_fixture(
    T, J, original, family, job; right=(family==:gesvd && job=='O' ? 'S' : job)
)
    m, n = size(original)
    r = min(m, n)
    a = factor_abi_matrix(original)
    if family==:gesvd
        ucols=if job=='A'
            m
        elseif job=='S'
            r
        else
            0
        end
        vtrows=if right=='A'
            n
        elseif right=='S'
            r
        else
            0
        end
    else
        ucols=if job=='A' || (job=='O' && m<n)
            m
        elseif job=='S'
            r
        else
            0
        end
        vtrows=if job=='A' || (job=='O' && m>=n)
            n
        elseif job=='S'
            r
        else
            0
        end
    end
    u=factor_abi_matrix(zeros(T, ucols>0 ? max(1, m) : 1, max(1, ucols)))
    vt=factor_abi_matrix(zeros(T, max(1, vtrows), vtrows>0 ? max(1, n) : 1))
    sg, s = factor_abi_integer(real(T), max(1, r))
    wg, work = factor_abi_integer(T, direct_svd_minimum(T, family, job, m, n))
    # LAPACK 3.12.1 CGESDD RWORK contract (these nonempty fixtures have
    # aspect ratio 4/3). CGESVD and values-only CGESDD require 5*min(M,N).
    real_work=if T<:Complex
        family==:gesvd || job=='N' ? 5r : max(5r*r+5r, 2max(m, n)*r+2r*r+r)
    else
        1
    end
    rg, rwork=factor_abi_integer(real(T), max(1, real_work))
    pg, iwork = factor_abi_integer(J, max(1, 8r))
    ig, info = factor_abi_integer(J)
    return (; a, u, vt, sg, s, wg, work, rg, rwork, pg, iwork, ig, info)
end
function direct_svd_execute(f, family, J, x, m, n, job, right, lw)
    return direct_svd_abi(
        f,
        Val(family),
        J,
        x.a.data,
        m,
        n,
        x.s,
        x.u.data,
        x.vt.data,
        x.work,
        lw,
        x.rwork,
        x.iwork,
        x.info,
        job,
        right,
    )
end
function direct_svd_guards(x)
    for matrix in (x.a, x.u, x.vt)
        factor_abi_guards(matrix)
    end
    for array in (x.sg, x.wg, x.rg, x.pg, x.ig)
        factor_abi_integer_guards(array)
    end
end

@testset "Direct SVD integer ABI and workspace contracts" begin
    Random.seed!(31714)
    library = Libdl.dlopen(Colmat.backend_library())
    original_library = only(
        filter(
            l -> l.interface == :ilp64 && occursin("openblas", lowercase(l.libname)),
            BLAS.get_config().loaded_libs,
        ),
    ).libname
    reference_library = Libdl.dlopen(original_library)
    for (T, prefix) in
        ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
        J in (Int32, Int64),
        route in (:dll, :lbt),
        family in (:gesvd, :gesdd),
        (m, n) in ((4, 3), (3, 4), (0, 0), (0, 3), (3, 0)),
        job in ('N', 'S', 'A', 'O')

        @testset "$T/$J/$route/$family/$m,$n/$job" begin
            symbol = prefix * string(family)
            f = if route == :dll
                Libdl.dlsym(library, symbol * (J == Int32 ? "_" : "_64_"))
            else
                Libdl.dlsym(
                    Libdl.dlopen(BLAS.libblastrampoline),
                    symbol * (J == Int32 ? "_" : "_64_"),
                )
            end
            reference = Libdl.dlsym(reference_library, symbol * "_64_")
            original = randn(T, m, n)
            x = direct_svd_fixture(T, J, original, family, job)
            # GESVD cannot overwrite both sides. Exercise left overwrite here;
            # the right-overwrite companion below uses the same concrete ABI.
            right = family == :gesvd && job == 'O' ? 'S' : job
            minimum = length(x.work)
            direct_svd_execute(f, family, J, x, m, n, job, right, -1)
            @test x.info[1] == 0 && real(x.work[1]) >= minimum
            @test x.a.data == original
            @test all(iszero, x.u.data) &&
                all(iszero, x.vt.data) &&
                all(==(factor_abi_sentinel(real(T))), x.s)
            direct_svd_guards(x)
            direct_svd_execute(f, family, J, x, m, n, job, right, minimum-1)
            @test x.info[1] == -(family == :gesvd ? 13 : 12)
            @test factor_abi_diagnostic() == -x.info[1]
            @test x.a.data == original
            direct_svd_guards(x)
            direct_svd_execute(f, family, J, x, m, n, job, right, minimum)
            @test x.info[1] == 0
            y = direct_svd_fixture(T, Int64, original, family, job)
            direct_svd_execute(
                reference, family, Int64, y, m, n, job, right, length(y.work)
            )
            @test y.info[1] == 0
            direct_svd_guards(y)
            r = min(m, n)
            @test x.s[1:r] ≈ y.s[1:r]
            @test issorted(x.s[1:r]; rev=true) && all(>=(0), x.s[1:r])
            if job != 'N' && r > 0
                left = if job == 'O' && (family == :gesvd || m >= n)
                    x.a.data[:, 1:r]
                else
                    x.u.data[1:m, 1:r]
                end
                right_vectors = if job == 'O' && family == :gesdd && m < n
                    x.a.data[1:r, :]
                else
                    x.vt.data[1:r, 1:n]
                end
                @test left * Diagonal(x.s[1:r]) * right_vectors ≈ original
                @test left' * left ≈ Matrix{T}(I, r, r)
                @test right_vectors * right_vectors' ≈ Matrix{T}(I, r, r)
            end
            if job=='A' && r>0
                @test x.u.data'*x.u.data ≈ Matrix{T}(I, m, m)
                @test x.vt.data*x.vt.data' ≈ Matrix{T}(I, n, n)
            end
            direct_svd_guards(x)
            if family == :gesvd && job == 'O' && r > 0
                x = direct_svd_fixture(T, J, original, family, 'S'; right=('O'))
                direct_svd_execute(f, family, J, x, m, n, 'S', 'O', length(x.work))
                @test x.info[1] == 0
                @test x.u.data[1:m, 1:r] * Diagonal(x.s[1:r]) * x.a.data[1:r, :] ≈ original
                direct_svd_guards(x)
            end
        end
    end
end
