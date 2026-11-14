using Test, Colmat, LinearAlgebra

@testset "Complete LBT symbol inventory" begin
    file = joinpath(@__DIR__, "..", "docs", "src", "assets", "lbt-symbol-inventory.tsv")
    lines = readlines(file)
    header = split(first(lines), '\t')
    rows = [Dict(zip(header, split(line, '\t'; keepempty=true))) for line in lines[2:end]]
    config = BLAS.get_config()
    recorded = [row["symbol"] for row in rows]
    @test length(recorded) == length(unique(recorded))
    @test Set(recorded) == Set(config.exported_symbols)
    @test all(!isempty(row["colmat_api"]) && !isempty(row["contract"]) for row in rows)
    for (interface, column) in ((:lp64, "adapter_lp64"), (:ilp64, "adapter_ilp64"))
        actual = Set{String}()
        for lib in config.loaded_libs
            if lib.interface == interface &&
                realpath(lib.libname) == realpath(Colmat.backend_library())
                union!(actual, BLAS.lbt_forwarded_funcs(config, lib))
            end
        end
        @test actual == Set(row["symbol"] for row in rows if row[column] == "true")
    end
end
