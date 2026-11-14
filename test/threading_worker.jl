# Launched in a fresh process with four Julia OS threads by runtests.jl.
using Colmat, LinearAlgebra, Test
BLAS.set_num_threads(2)
@test Colmat.get_num_threads() == 2
@testset "Concurrent native teams" begin
    jobs = map(1:8) do seed
        Threads.@spawn begin
            A = reshape(Float64.(1:1200), 40, 30) .+ seed
            B = reshape(Float64.(1:600), 30, 20) ./ seed
            ref = [sum(A[i, k] * B[k, j] for k in 1:30) for i in 1:40, j in 1:20]
            all(isapprox(A * B, ref) for _ in 1:4)
        end
    end
    @test all(fetch, jobs)
    BLAS.set_num_threads(1)
    @test Colmat.get_num_threads() == 1
end
