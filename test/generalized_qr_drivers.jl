using Test, LinearAlgebra, Random, Libdl, Colmat
function generalized_qr_reference()
    return only(
        filter(
            l ->
                l.interface == :ilp64 &&
                occursin("openblas", lowercase(basename(l.libname))),
            BLAS.get_config().loaded_libs,
        ),
    ).libname
end
for (T, prefix) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    J in (Int32, Int64)

    for (name, lse) in (("gglse", true), ("ggglm", false))
        sym=Symbol(prefix*name*"_"*(J===Int64 ? "64_" : ""))
        @eval function direct_generalized_driver!(
            ::Val{$lse},
            a::Matrix{$T},
            b::Matrix{$T},
            rhs::Vector{$T},
            second::Vector{$T},
            last::Vector{$T},
            firstdim,
            seconddim,
            p,
            ::Type{$J};
            lwork=firstdim+seconddim+p,
            library=Colmat.backend_library(),
            lda=size(a, 1),
            ldb=size(b, 1),
        )
            guard=$T(777)
            w=fill(guard, max(1, lwork)+2)
            info=fill($J(-777), 3)
            f=Libdl.dlsym(Libdl.dlopen(library), $(QuoteNode(sym)))
            ccall(
                f,
                Cvoid,
                (
                    Ref{$J},
                    Ref{$J},
                    Ref{$J},
                    Ptr{$T},
                    Ref{$J},
                    Ptr{$T},
                    Ref{$J},
                    Ptr{$T},
                    Ptr{$T},
                    Ptr{$T},
                    Ptr{$T},
                    Ref{$J},
                    Ptr{$J},
                ),
                firstdim,
                seconddim,
                p,
                a,
                lda,
                b,
                ldb,
                view(rhs, 2:(length(rhs) - 1)),
                view(second, 2:(length(second) - 1)),
                view(last, 2:(length(last) - 1)),
                view(w, 2:(length(w) - 1)),
                lwork,
                view(info, 2:2),
            )
            @test info[[1, 3]]==[$J(-777), $J(-777)]
            @test w[[1, end]]==[guard, guard]
            @test rhs[[1, end]]==second[[1, end]]==last[[1, end]]==[guard, guard]
            (; info=info[2], workspace=real(w[2]))
        end
    end
end
@testset "GGLSE native constraints and independent drivers, both integer ABIs" begin
    rng=MersenneTwister(314)
    ref=generalized_qr_reference()
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64), m in 0:5, n in 0:5,
        p in 0:n

        n<=m+p || continue
        R=real(T)
        tol=1000eps(R)*max(1, m, n, p)
        ao=randn(rng, T, m, n)
        bo=randn(rng, T, p, n)
        c0=randn(rng, T, m)
        d0=randn(rng, T, p)
        native_x=nothing
        for (library, K) in ((Colmat.backend_library(), J), (ref, Int64))
            a=vcat(ao, fill(T(777), 2, n))
            b=vcat(bo, fill(T(777), 3, n))
            c=vcat(T(777), c0, T(777))
            d=vcat(T(777), d0, T(777))
            x=fill(T(777), n+2)
            saved=deepcopy((a, b, c, d, x))
            query=direct_generalized_driver!(
                Val(true), a, b, c, d, x, m, n, p, K; library, lwork=-1
            )
            @test query.info==0 && query.workspace>=(n==0 ? 1 : m+n+p)
            @test isequal((a, b, c, d, x), saved)
            result=direct_generalized_driver!(
                Val(true), a, b, c, d, x, m, n, p, K; library, lwork=n==0 ? 1 : m+n+p
            )
            @test result.info==0
            @test all(==(T(777)), a[(m + 1):end, :]) && all(==(T(777)), b[(p + 1):end, :])
            sol=x[2:(end - 1)]
            @test norm(bo*sol-d0)<=tol*max(1, norm(bo)*norm(sol), norm(d0))
            z=p==0 ? Matrix{T}(I, n, n) : nullspace(bo)
            gradient=z'*(ao'*(ao*sol-c0))
            @test norm(gradient)<=tol*max(1, norm(ao)^2*norm(sol), norm(ao)*norm(c0))
            @test norm(c[(n - p + 2):(m + 1)])≈norm(ao*sol-c0) atol=tol*max(1, norm(c0)) rtol=tol
            if isnothing(native_x)
                native_x=sol
            else
                @test native_x≈sol atol=tol*max(1, norm(sol)) rtol=tol
            end
        end
    end
end
@testset "GGGLM native minimum norm and independent drivers, both integer ABIs" begin
    rng=MersenneTwister(3141)
    ref=generalized_qr_reference()
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64), n in 0:5, m in 0:n,
        p in (n - m):7

        R=real(T)
        tol=1000eps(R)*max(1, n, m, p)
        ao=randn(rng, T, n, m)
        bo=randn(rng, T, n, p)
        d0=randn(rng, T, n)
        native=nothing
        for (library, K) in ((Colmat.backend_library(), J), (ref, Int64))
            a=vcat(ao, fill(T(777), 2, m))
            b=vcat(bo, fill(T(777), 3, p))
            d=vcat(T(777), d0, T(777))
            x=fill(T(777), m+2)
            y=fill(T(777), p+2)
            saved=deepcopy((a, b, d, x, y))
            query=direct_generalized_driver!(
                Val(false), a, b, d, x, y, n, m, p, K; library, lwork=-1
            )
            @test query.info==0 && query.workspace>=(n==0 ? 1 : n+m+p)
            @test isequal((a, b, d, x, y), saved)
            result=direct_generalized_driver!(
                Val(false), a, b, d, x, y, n, m, p, K; library, lwork=n==0 ? 1 : n+m+p
            )
            @test result.info==0
            @test all(==(T(777)), a[(n + 1):end, :]) && all(==(T(777)), b[(n + 1):end, :])
            sx=x[2:(end - 1)]
            sy=y[2:(end - 1)]
            @test norm(ao*sx+bo*sy-d0)<=tol*max(
                1, norm(ao)*norm(sx), norm(bo)*norm(sy), norm(d0)
            )
            # Feasible perturbations come from null([A B]); y is orthogonal to their y components.
            z=nullspace(hcat(ao, bo))
            @test norm(z[(m + 1):end, :]'*sy)<=tol*max(1, norm(sy))
            if isnothing(native)
                native=(sx, sy)
            else
                @test native[1]≈sx atol=tol*max(1, norm(sx)) rtol=tol
                @test native[2]≈sy atol=tol*max(1, norm(sy)) rtol=tol
            end
        end
    end
end
