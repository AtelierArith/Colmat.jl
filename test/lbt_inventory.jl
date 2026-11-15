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
    @test all(
        haskey(row, field) for row in rows for field in (
            "capability_status",
            "capability_evidence",
            "evidence_status",
            "execution_evidence_status",
            "execution_evidence",
            "independent_algorithm_status",
        )
    )
    @test all(
        row["capability_status"] in (
            "native",
            "representation",
            "composition",
            "conditional",
            "unsupported",
            "unverified",
        ) for row in rows
    )
    absent=filter(row -> row["capability_status"]=="unsupported", rows)
    @test !isempty(absent)
    @test all(
        row["evidence_status"]=="scoped_review" && !isempty(row["capability_evidence"]) for
        row in absent
    )
    coverage=readlines(joinpath(dirname(file), "lbt-candidate-coverage.tsv"))
    coverage_header=split(first(coverage), '\t')
    coverage_rows=[
        Dict(zip(coverage_header, split(line, '\t'; keepempty=true))) for
        line in coverage[2:end]
    ]
    @test Set(row["symbol"] for row in coverage_rows)==Set(recorded)
    @test length(coverage_rows)==length(unique(row["symbol"] for row in coverage_rows))
    historical=filter(row -> row["historical_candidate"]=="true", coverage_rows)
    @test length(historical)==2910
    @test all(!isempty(row["owner_issues"]) for row in historical)
    added=filter(row -> row["reassessment_addition"]=="true", coverage_rows)
    @test length(added)==40
    @test all("372" in split(row["owner_issues"]) for row in added)
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
