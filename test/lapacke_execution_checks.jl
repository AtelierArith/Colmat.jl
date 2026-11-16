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
        "geev",
        "geevx",
        "trtri",
        "trtrs",
        "lauum",
        "geqrf",
        "geqr2",
        "geqrfp",
        "gelqf",
        "gelq2",
        "geqlf",
        "gerqf",
        "tzrzf",
        "gels",
        "gelsy",
        "gelss",
        "gelsd",
        "tpqrt",
        "tpqrt2",
        "tpmqrt",
        "tprfb",
        "geqp3",
        "geqpf",
        "geqrt",
        "geqrt2",
        "geqrt3",
        "gemqrt",
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
    union!(
        targets,
        Set(
            "LAPACKE_"*p*f*w for
            (p, f) in (("s", "syevd"), ("d", "syevd"), ("c", "heevd"), ("z", "heevd")) for
            w in ("", "_work")
        ),
    )
    union!(
        targets,
        Set(
            "LAPACKE_"*p*f*w for
            (p, f) in (("s", "syevr"), ("d", "syevr"), ("c", "heevr"), ("z", "heevr")) for
            w in ("", "_work")
        ),
    )
    union!(
        targets,
        Set(
            "LAPACKE_"*p*f*w for
            (p, stem) in (("s", "org"), ("d", "org"), ("c", "ung"), ("z", "ung")) for
            f in (stem*"qr", stem*"lq", stem*"ql", stem*"rq") for w in ("", "_work")
        ),
    )
    union!(
        targets,
        Set(
            "LAPACKE_"*p*f*w for
            (p, stem) in (("s", "orm"), ("d", "orm"), ("c", "unm"), ("z", "unm")) for
            f in (stem*"qr", stem*"lq", stem*"ql", stem*"rq", stem*"rz") for
            w in ("", "_work")
        ),
    )
    @test length(targets)==420
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
