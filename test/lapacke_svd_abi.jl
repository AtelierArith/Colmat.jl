using Test, Colmat

@testset "LAPACKE SVD ABI from an independent C caller" begin
    if Sys.islinux() || Sys.isapple()
        mktempdir() do directory
            executable = joinpath(directory, "lapacke-svd-abi-worker")
            source = joinpath(@__DIR__, "lapacke_svd_abi_worker.c")
            compiler = get(ENV, "CC", "cc")
            libraries = Sys.islinux() ? ["-ldl", "-lm"] : String[]
            run(
                `$compiler -std=c11 -Wall -Wextra -Werror $source -o $executable $libraries`
            )
            library = Colmat.backend_library()
            output = IOBuffer()
            process = run(
                pipeline(ignorestatus(`$executable $library`); stdout=output, stderr=output)
            )
            @test success(process)
            @test occursin("LAPACKE SVD C ABI:", String(take!(output)))
        end
    else
        @test_skip "independent C ABI caller is implemented for Linux/macOS"
    end
end
