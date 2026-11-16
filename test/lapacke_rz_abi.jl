using Test, Colmat, LinearAlgebra

@testset "LAPACKE RZ ABI from an independent C caller" begin
    if Sys.islinux() || Sys.isapple()
        mktempdir() do directory
            executable = joinpath(directory, "lapacke-rz-abi-worker")
            source = joinpath(@__DIR__, "lapacke_rz_abi_worker.c")
            compiler = get(ENV, "CC", "cc")
            libraries = Sys.islinux() ? ["-ldl", "-lm"] : String[]
            run(
                `$compiler -std=c11 -Wall -Wextra -Werror $source -o $executable $libraries`
            )
            library = Colmat.backend_library()
            reference=only(
                filter(
                    l->l.interface==:ilp64 && occursin("openblas", lowercase(l.libname)),
                    BLAS.get_config().loaded_libs,
                ),
            )
            reference_library=reference.libname
            output = IOBuffer()
            process = run(
                pipeline(
                    ignorestatus(`$executable $library $reference_library`);
                    stdout=output,
                    stderr=output,
                ),
            )
            @test success(process)
            @test occursin("LAPACKE RZ C ABI:", String(take!(output)))
        end
    else
        @test_skip "independent C ABI caller is implemented for Linux/macOS"
    end
end
