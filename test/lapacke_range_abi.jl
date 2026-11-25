using Test, Colmat, LinearAlgebra

@testset "LAPACKE range ABI from an independent C caller" begin
    if Sys.islinux() || Sys.isapple()
        mktempdir() do directory
            executable = joinpath(directory, "lapacke-range-abi-worker")
            source = joinpath(@__DIR__, "lapacke_range_abi_worker.c")
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
            group=Colmat._fallback_group("range self-adjoint EVD").group
            output = IOBuffer()
            process = run(
                pipeline(
                    ignorestatus(`$executable $library $reference_library $group`);
                    stdout=output,
                    stderr=output,
                ),
            )
            @test success(process)
            @test occursin("LAPACKE range C ABI:", String(take!(output)))
        end
    else
        @test_skip "independent C ABI caller is implemented for Linux/macOS"
    end
end
