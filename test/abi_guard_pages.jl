using Test, Colmat
@testset "Integer ABI width crash is isolated" begin
    if Sys.islinux() || Sys.isapple()
        mktempdir() do directory
            executable = joinpath(directory, "guard-page-worker")
            source = joinpath(@__DIR__, "abi_guard_page_worker.c")
            compiler = get(ENV, "CC", "cc")
            libraries = Sys.islinux() ? ["-ldl"] : String[]
            run(
                `$compiler -std=c11 -D_DEFAULT_SOURCE -O0 $source -o $executable $libraries`
            )
            library = Colmat.backend_library()
            control_output, crash_output = IOBuffer(), IOBuffer()
            control = run(
                pipeline(
                    ignorestatus(`$executable $library valid`);
                    stdout=control_output,
                    stderr=control_output,
                ),
            )
            @test success(control)
            @test occursin("VALID_WIDTH_CONTROL_PASSED", String(take!(control_output)))
            narrow_output=IOBuffer()
            narrow=run(
                pipeline(
                    ignorestatus(`$executable $library narrow`);
                    stdout=narrow_output,
                    stderr=narrow_output,
                ),
            )
            @test success(narrow)
            @test occursin("NARROW_INFO_WRITE_DETECTED", String(take!(narrow_output)))
            if success(control)
                crash = run(
                    pipeline(
                        ignorestatus(`$executable $library wrong`);
                        stdout=crash_output,
                        stderr=crash_output,
                    ),
                )
                @test occursin("WRONG_WIDTH_CALL_READY", String(take!(crash_output)))
                @test !success(crash)
                @test crash.termsignal in (11, Sys.isapple() ? 10 : 7, 6)
            end
        end
    else
        @test_skip "guard-page witness is implemented for Linux/macOS"
    end
end
