using Test, LinearAlgebra, Random, Libdl, Colmat
@testset "All generalized QR and constrained drivers use native LBT providers" begin
    Colmat.load_backend()
    rng=MersenneTwister(3143)
    for (T, prefix) in
        ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
        J in (Int32, Int64),
        family in ("ggqrf", "ggrqf", "gglse", "ggglm")

        interface=J===Int64 ? :ilp64 : :lp64
        symbol=prefix*family*"_"
        provider=BLAS.lbt_find_backing_library(symbol, interface)
        @test provider!==nothing &&
            realpath(provider.libname)==realpath(Colmat.backend_library())
        pointer=BLAS.lbt_get_forward(symbol, interface)
        tol=3000eps(real(T))
        if family in ("ggqrf", "ggrqf")
            rq=family=="ggrqf"
            ar, ac, br, bc=rq ? (3, 4, 2, 4) : (3, 2, 3, 4)
            ao=randn(rng, T, ar, ac)
            bo=randn(rng, T, br, bc)
            a=vcat(ao, fill(T(777), 2, ac))
            b=vcat(bo, fill(T(777), 3, bc))
            result=direct_generalized_pair!(Val(rq), a, b, 3, 2, 4, J; pointer)
            @test result.info==result.diagnostic==0
            qa=generalized_pair_q(Val(rq), a, result.ta, ar, ac)
            qb=generalized_pair_q(Val(!rq), b, result.tb, br, bc)
            ra=generalized_pair_triangular(a, ar, ac, rq)
            rb=generalized_pair_triangular(b, br, bc, !rq)
            @test (rq ? ra*qa : qa*ra)≈ao atol=tol rtol=tol
            @test (rq ? qb*rb*qa : qa*rb*qb)≈bo atol=tol rtol=tol
        else
            lse=family=="gglse"
            ao=randn(rng, T, 3, 2)
            bo=randn(rng, T, lse ? 1 : 3, 2)
            a=vcat(ao, fill(T(777), 2, 2))
            b=vcat(bo, fill(T(777), 3, 2))
            rhs0=randn(rng, T, 3)
            rhs=vcat(T(777), rhs0, T(777))
            second0=lse ? randn(rng, T, 1) : fill(T(777), 2)
            second=vcat(T(777), second0, T(777))
            last=fill(T(777), 4)
            result=direct_generalized_driver!(
                Val(lse), a, b, rhs, second, last, 3, 2, lse ? 1 : 2, J; pointer
            )
            @test result.info==result.diagnostic==0
            if lse
                x=last[2:(end - 1)]
                @test bo*x≈second0 atol=tol rtol=tol
                @test norm(nullspace(bo)'*(ao'*(ao*x-rhs0)))<=tol*max(1, norm(rhs0))
            else
                x=second[2:(end - 1)]
                y=last[2:(end - 1)]
                @test ao*x+bo*y≈rhs0 atol=tol rtol=tol
                @test norm(nullspace(hcat(ao, bo))[3:end, :]'*y)<=tol*max(1, norm(y))
            end
        end
    end
end
