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
        "lacpy",
        "laset",
        "laswp",
        "tgevc",
        "lange",
        "lansy",
        "lantr",
        "lapmr",
        "lapmt",
        "trevc",
        "trexc",
        "lascl",
        "trsen",
        "geqrf",
        "geqr2",
        "geqrfp",
        "gelqf",
        "gelq2",
        "geqlf",
        "gerqf",
        "tzrzf",
        "ggqrf",
        "ggrqf",
        "gglse",
        "ggglm",
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
    union!(
        targets,
        Set(
            "LAPACKE_"*p*f*w for
            (p, f) in (("s", "orgtr"), ("d", "orgtr"), ("c", "ungtr"), ("z", "ungtr")) for
            w in ("", "_work")
        ),
    )
    union!(
        targets,
        Set(
            "LAPACKE_"*p*f*w for
            (p, f) in (("s", "ormtr"), ("d", "ormtr"), ("c", "unmtr"), ("z", "unmtr")) for
            w in ("", "_work")
        ),
    )
    union!(
        targets,
        Set(
            "LAPACKE_"*p*f*w for
            (p, f) in (("s", "orgbr"), ("d", "orgbr"), ("c", "ungbr"), ("z", "ungbr")) for
            w in ("", "_work")
        ),
    )
    union!(
        targets,
        Set(
            "LAPACKE_"*p*f*w for
            (p, f) in (("s", "ormbr"), ("d", "ormbr"), ("c", "unmbr"), ("z", "unmbr")) for
            w in ("", "_work")
        ),
    )
    union!(
        targets,
        Set(
            "LAPACKE_"*p*f*w for
            (p, f) in (("s", "orghr"), ("d", "orghr"), ("c", "unghr"), ("z", "unghr")) for
            w in ("", "_work")
        ),
    )
    union!(
        targets,
        Set(
            "LAPACKE_"*p*f*w for
            (p, f) in (("s", "ormhr"), ("d", "ormhr"), ("c", "unmhr"), ("z", "unmhr")) for
            w in ("", "_work")
        ),
    )
    union!(
        targets,
        Set("LAPACKE_"*p*f*w for p in ("c", "z") for f in ("lacgv", "lacp2") for
            w in ("", "_work")),
    )
    union!(targets, Set("LAPACKE_"*p*"larfg"*w for p in ("s", "d", "c", "z") for w in ("", "_work")))
    union!(targets, Set("LAPACKE_"*p*"larft"*w for p in ("s", "d", "c", "z") for w in ("", "_work")))
    union!(targets, Set("LAPACKE_"*p*"larfb"*w for p in ("s", "d", "c", "z") for w in ("", "_work")))
    union!(targets, Set("LAPACKE_"*p*"lartg"*q*w for p in ("s", "d") for q in ("p", "s") for w in ("", "_work")))
    union!(targets, Set("LAPACKE_"*p*"gees"*w for p in ("s", "d", "c", "z") for w in ("", "_work")))
    union!(targets, Set("LAPACKE_"*p*"geesx"*w for p in ("s", "d", "c", "z") for w in ("", "_work")))
    union!(targets, Set("LAPACKE_"*p*f*w for p in ("s", "d", "c", "z") for f in ("ggev", "ggev3", "ggevx") for w in ("", "_work")))
    union!(targets, Set("LAPACKE_"*p*f*w for p in ("s", "d", "c", "z") for f in ("gghrd", "gghd3") for w in ("", "_work")))
    union!(targets, Set("LAPACKE_"*p*"hgeqz"*w for p in ("s", "d", "c", "z") for w in ("", "_work")))
    union!(targets, Set("LAPACKE_"*p*f*w for p in ("c", "z") for f in ("upgtr", "upmtr") for w in ("", "_work")))
    union!(targets, Set("LAPACKE_"*p*f*w for p in ("s", "d") for f in ("opgtr", "opmtr") for w in ("", "_work")))
    union!(targets, Set("LAPACKE_"*p*f*w for p in ("s", "d", "c", "z") for f in ("pttrf", "pttrs") for w in ("", "_work")))
    union!(targets, Set("LAPACKE_"*p*f*w for p in ("s", "d", "c", "z") for f in ("pbtrf", "pbtrs") for w in ("", "_work")))
    union!(targets, Set("LAPACKE_"*p*f*w for (p, f) in (("s", "sbtrd"), ("d", "sbtrd"), ("c", "hbtrd"), ("z", "hbtrd")) for w in ("", "_work")))
    union!(targets, Set("LAPACKE_"*p*"pteqr"*w for p in ("s", "d", "c", "z") for w in ("", "_work")))
    union!(targets, Set("LAPACKE_"*p*f*w for p in ("s", "d", "c", "z") for f in ("pptrf", "pptrs") for w in ("", "_work")))
    union!(targets, Set("LAPACKE_"*p*f*w for p in ("s", "d", "c", "z") for f in ("gbtrf", "gbtrs", "gbsv") for w in ("", "_work")))
    union!(targets, Set("LAPACKE_"*p*f*w for p in ("s", "d", "c", "z") for f in ("gbequ", "gbequb") for w in ("", "_work")))
    union!(targets, Set("LAPACKE_"*p*f*w for p in ("s", "d", "c", "z") for f in ("gttrf", "gttrs", "gtsv") for w in ("", "_work")))
    union!(targets, Set("LAPACKE_"*p*f*w for p in ("s", "d") for f in ("sptrf", "sptrs", "spsv") for w in ("", "_work")))
    union!(targets, Set("LAPACKE_"*p*f*w for (p, f) in (("s", "sytrf"), ("d", "sytrf"), ("s", "sytrs"), ("d", "sytrs"), ("s", "sysv"), ("d", "sysv"), ("c", "hetrf"), ("z", "hetrf"), ("c", "hetrs"), ("z", "hetrs"), ("c", "hesv"), ("z", "hesv")) for w in ("", "_work")))
    union!(targets, Set("LAPACKE_"*p*f*w for p in ("s", "d", "c", "z") for f in ("gebal", "gebak") for w in ("", "_work")))
    union!(targets, Set("LAPACKE_"*p*f*w for p in ("s", "d", "c", "z") for f in ("tbtrs", "tptrs", "tptri") for w in ("", "_work")))
    union!(targets, Set("LAPACKE_"*p*"langb"*w for p in ("s", "d", "c", "z") for w in ("", "_work")))
    union!(targets, Set("LAPACKE_"*p*"lanhe"*w for p in ("c", "z") for w in ("", "_work")))
    union!(targets, Set("LAPACKE_"*p*f*w for p in ("s", "d", "c", "z") for f in ("gehrd", "gebrd", "larfx") for w in ("", "_work")))
    union!(targets, Set("LAPACKE_"*p*f*w for p in ("s", "d", "c", "z") for f in ("geequ", "geequb") for w in ("", "_work")))
    union!(targets, Set("LAPACKE_"*p*f*w for p in ("s", "d", "c", "z") for f in ("gges", "gges3") for w in ("", "_work")))
    union!(targets, Set("LAPACKE_"*p*"hseqr"*w for p in ("s", "d", "c", "z") for w in ("", "_work")))
    union!(targets, Set("LAPACKE_"*p*f*w for p in ("s", "d", "c", "z") for f in ("geqr", "gelq", "gemqr", "gemlq") for w in ("", "_work")))
    union!(targets, Set("LAPACKE_"*p*f*w for p in ("s", "d", "c", "z") for f in ("pbequ", "pbstf", "pbsv") for w in ("", "_work")))
    union!(targets, Set("LAPACKE_"*p*f*w for p in ("s", "d", "c", "z") for f in ("pftrf", "pftri", "pftrs") for w in ("", "_work")))
    @test length(targets)==1084
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
    norm_targets=Set(
        "LAPACKE_"*p*f*w for p in ("s", "d", "c", "z") for
        f in ("lange", "lansy", "lantr", "langb") for w in ("", "_work")
    )
    union!(norm_targets, Set("LAPACKE_"*p*"lanhe"*w for p in ("c", "z") for w in ("", "_work")))
    for e in ABIEvidence.witnesses
        e[1] in targets || continue
        e[4]=="test/abi_execution_checks.jl" && continue
        expected_return = if e[1] in norm_targets
            e[1][9] in ('s', 'c') ? "Float32" : "Float64"
        else
            "Int"*string(e[2])
        end
        @test ABIEvidence.return_types[e]==expected_return
        if occursin("lacgv", e[1]) || occursin("larfg", e[1]) || occursin("gttrf", e[1])
            @test startswith(e[6], e[2] == 32 ? "(Int32," : "(Int64,")
        elseif occursin("lartg", e[1])
            @test startswith(e[6], "(Float32,") || startswith(e[6], "(Float64,")
        elseif occursin("pttrf", e[1]) || occursin("gttrf", e[1])
            @test startswith(e[6], "(Int32,") || startswith(e[6], "(Int64,")
        else
            @test startswith(e[6], "(Int32,")
        end
        @test e[7] || occursin("lartg", e[1]) || occursin("pttrf", e[1]) || occursin("gttrf", e[1])
        @test isfile(joinpath(ABIEvidence.root, e[4]))
        @test 1<=e[5]<=countlines(joinpath(ABIEvidence.root, e[4]))
    end
end
