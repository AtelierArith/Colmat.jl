using Test, Colmat, LinearAlgebra

@testset "LAPACKE generalized QR ABI from an independent C caller" begin
    if Sys.islinux() || Sys.isapple()
        mktempdir() do directory
            executable = joinpath(directory, "lapacke-generalized-qr-abi-worker")
            source = joinpath(@__DIR__, "lapacke_generalized_qr_abi_worker.c")
            compiler = get(ENV, "CC", "cc")
            libraries = Sys.islinux() ? ["-ldl", "-lm"] : String[]
            run(
                `$compiler -std=c11 -Wall -Wextra -Werror $source -o $executable $libraries`
            )
            output = IOBuffer()
            library = Colmat.backend_library()
            process = run(
                pipeline(ignorestatus(`$executable $library`); stdout=output, stderr=output)
            )
            @test success(process)
            @test occursin(
                "LAPACKE generalized QR C ABI: 32 names verified", String(take!(output))
            )
        end
    else
        @test_skip "independent C ABI caller is implemented for Linux/macOS"
    end
end
