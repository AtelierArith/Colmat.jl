using Test, LinearAlgebra, Libdl, Colmat
@testset "Constrained drivers common near-overflow and subnormal scales" begin
    reference=generalized_qr_reference()
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        lse in (false, true)

        R=real(T)
        phase=T<:Complex ? T(1, 0.2) : one(T)
        ao=(
            if lse
                T[3 0.2 -0.3; 0.4 2 0.1; 0.2 -0.1 1; 0.3 0.4 -0.2]
            else
                T[3 0.2; 0.4 2; 0.2 -0.1]
            end
        )*phase
        bo=(lse ? T[0.2 -0.4 2] : T[0.5 -0.3; 0.2 0.1; 2 0.4])*conj(phase)
        rhs0=lse ? T[1, 2, -1, 0.4] : T[1, 2, -1]
        second0=lse ? T[0.6] : fill(T(777), 2)
        dims=lse ? (4, 3, 1) : (3, 2, 2)
        expected=nothing
        for scale in (one(R), floatmin(R)/32, floatmax(R)/4096)
            # Allow the information actually lost while representing subnormal inputs.
            tol=2000eps(R)+128nextfloat(zero(R))/scale
            for (library, K) in ((Colmat.backend_library(), J), (reference, Int64))
                # OpenBLAS 0.3.30 returned NaNs for subnormal complex pencils.
                # A common nonzero scale leaves these problems unchanged, so
                # use its unit-scale solution as the independent oracle there.
                # The native call always receives the original extreme scale.
                backend_scale=if library==reference && T<:Complex && scale<floatmin(R)
                    one(R)
                else
                    scale
                end
                a=vcat(ao*backend_scale, fill(T(777), 2, size(ao, 2)))
                b=vcat(bo*backend_scale, fill(T(777), 3, size(bo, 2)))
                rhs=vcat(T(777), rhs0*backend_scale, T(777))
                second=vcat(T(777), lse ? second0*backend_scale : second0, T(777))
                last=fill(T(777), (lse ? 3 : 2)+2)
                result=direct_generalized_driver!(
                    Val(lse), a, b, rhs, second, last, dims..., K; library
                )
                @test result.info==result.diagnostic==0
                solution=if lse
                    last[2:(end - 1)]
                else
                    vcat(second[2:(end - 1)], last[2:(end - 1)])
                end
                all(isfinite, solution) ||
                    @info "nonfinite constrained solve" T J lse scale library
                @test all(isfinite, solution)
                if isnothing(expected)
                    expected=copy(solution)
                else
                    @test solution≈expected atol=tol*max(1, norm(expected)) rtol=tol
                end
                if lse
                    @test bo*solution≈second0 atol=tol rtol=tol
                    @test norm(nullspace(bo)'*(ao'*(ao*solution-rhs0)))<=tol*max(
                        1, norm(rhs0)
                    )
                else
                    x=solution[1:2]
                    y=solution[3:4]
                    @test ao*x+bo*y≈rhs0 atol=tol rtol=tol
                    @test norm(nullspace(hcat(ao, bo))[3:end, :]'*y)<=tol*max(1, norm(y))
                end
            end
        end
    end
end
