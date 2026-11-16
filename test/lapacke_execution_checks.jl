using Test

@testset "Actual LAPACKE calls cover the implemented frozen scope" begin
    families=(
        "getrf",
        "getf2",
        "getrf2",
        "getrs",
        "gesv",
        "getri",
        "potrf",
        "potrf2",
        "potri",
        "potrs",
        "posv",
        "gesvd",
        "gesdd",
    )
    targets=Set(
        "LAPACKE_"*p*f*w for p in ("s", "d", "c", "z") for f in families for
        w in ("", "_work")
    )
    union!(targets, Set("LAPACKE_"*p*"syr"*w for p in ("c", "z") for w in ("", "_work")))
    union!(
        targets,
        Set(
            "LAPACKE_"*p*f*w for
            (p, f) in (("s", "syev"), ("d", "syev"), ("c", "heev"), ("z", "heev")) for
            w in ("", "_work")
        ),
    )
    @test length(targets)==116
    @test targets==ABIEvidence.expected_lapacke()
    for bits in (32, 64), route in ("dll", "lbt_trampoline")
        observed=Set(
            e[1] for e in ABIEvidence.witnesses if e[2]==bits &&
                e[3]==route &&
                e[1] in targets &&
                e[4]!="test/abi_execution_checks.jl"
        )
        @test observed==targets
    end
    for e in ABIEvidence.witnesses
        e[1] in targets || continue
        e[4]=="test/abi_execution_checks.jl" && continue
        @test ABIEvidence.return_types[e]=="Int"*string(e[2])
        @test startswith(e[6], "(Int32,")
        @test e[7]
        @test isfile(joinpath(ABIEvidence.root, e[4]))
        @test 1<=e[5]<=countlines(joinpath(ABIEvidence.root, e[4]))
    end
end
