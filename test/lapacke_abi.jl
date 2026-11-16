using Test, Colmat

@testset "LAPACKE ABI from an independent C caller" begin
    if Sys.islinux() || Sys.isapple()
        mktempdir() do directory
            executable = joinpath(directory, "lapacke-abi-worker")
            source = joinpath(@__DIR__, "lapacke_abi_worker.c")
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
            @test occursin("LAPACKE C ABI:", String(take!(output)))
            for (value, expected) in
                (("0", "0"), ("1", "1"), (" -0002rest", "1"), ("garbage", "0"))
                command = addenv(
                    `$executable $library $expected`, "LAPACKE_NANCHECK" => value
                )
                @test success(run(ignorestatus(command)))
            end
        end
    else
        @test_skip "independent C ABI caller is implemented for Linux/macOS"
    end
end
