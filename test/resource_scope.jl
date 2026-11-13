using Colmat
using LinearAlgebra
using SparseArrays
using Test

@testset "Scoped resources" begin
    A = FaerMatrix([4.0 1; 1 3])
    captured = Ref{Any}()
    F = with_resource(FaerScratch, svd, A) do scratch
        captured[] = scratch
        @test isopen(scratch)
        svd(A; scratch)
    end
    @test !isopen(captured[])
    @test Matrix(A) * (F \ [1.0, 2.0]) ≈ [1.0, 2.0]

    failure = ErrorException("block failed")
    caught = try
        with_resource(FaerTeam, 2; pin=false) do team
            captured[] = team
            @test isopen(team)
            throw(failure)
        end
    catch err
        err
    end
    @test caught === failure
    @test !isopen(captured[])

    events = Symbol[]
    @test with_resource(() -> IOBuffer()) do outer
        captured[] = outer
        with_resource(() -> IOBuffer()) do inner
            @test isopen(outer) && isopen(inner)
            push!(events, :inner)
        end
        @test isopen(outer)
        push!(events, :outer)
        42
    end == 42
    @test events == [:inner, :outer]
    @test !isopen(captured[])
    called = Ref(false)
    @test_throws ErrorException with_resource(() -> error("construction failed")) do _
        called[] = true
    end
    @test !called[]

    # A retained sparse factor must survive closing its input, plan and workspace.
    A0 = ComplexF64[0.1 4; 5 0.2im]
    saved = with_resource(FaerSparseMatrix, sparse(A0)) do a
        with_resource(
            FaerSparseSymbolic, a; kind=:lu, algorithm=:supernodal, ordering=[2, 1]
        ) do plan
            FaerTeam(2) do team
                with_resource(FaerScratch, factorize, plan, a; team) do scratch
                    factor = factorize(plan, a; scratch, team)
                    with_resource(FaerSparseMatrix, sparse(2A0); format=:csr) do b
                        reserve!(scratch, lu!, factor, b; team)
                        @test lu!(factor, b; scratch, team) === factor
                        @test Matrix(factor.L) * Matrix(factor.U) ≈
                            (2A0)[Vector(factor.p), Vector(factor.q)]
                    end
                    factor
                end
            end
        end
    end
    @test saved \ ones(ComplexF64, 2) ≈ (2A0) \ ones(ComplexF64, 2)
    close(saved)
end
