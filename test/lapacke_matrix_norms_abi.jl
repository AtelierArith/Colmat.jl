using Test, Colmat

@testset "LAPACKE matrix norm ABI from an independent C caller" begin
    if Sys.islinux() || Sys.isapple()
        mktempdir() do directory
            executable = joinpath(directory, "lapacke-matrix-norms-abi-worker")
            source = joinpath(@__DIR__, "lapacke_matrix_norms_abi_worker.c")
            compiler = get(ENV, "CC", "cc")
            libraries = Sys.islinux() ? ["-ldl", "-lm"] : String[]
            run(
                `$compiler -std=c11 -Wall -Wextra -Werror $source -o $executable $libraries`
            )
            output = IOBuffer()
            process = run(
                pipeline(
                    ignorestatus(`$executable $(Colmat.backend_library())`);
                    stdout=output,
                    stderr=output,
                ),
            )
            @test success(process)
            @test occursin("LAPACKE matrix norm C ABI:", String(take!(output)))
        end
    else
        @test_skip "independent C ABI caller is implemented for Linux/macOS"
    end
end
