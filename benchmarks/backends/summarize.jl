# Summarize backend benchmark TOML files into Markdown and CSV.
#
#   COLMAT_BENCH_RESULTS_DIR=benchmarks/backends/results/after \
#   julia --startup-file=no --project=benchmarks/backends benchmarks/backends/summarize.jl

using TOML
using Printf
using Statistics

const DIR = get(
    ENV,
    "COLMAT_BENCH_RESULTS_DIR",
    joinpath(@__DIR__, "results", "after"),
)
const BACKENDS = ["openblas", "mkl", "colmat"]

const OPS = [
    "gemm" => "GEMM `mul!`",
    "lu" => "`lu`",
    "cholesky" => "`cholesky`",
    "qr_factor" => "`qr` (factor)",
    "qr" => "`qr` (explicit Q)",
    "qr_pivot" => "`qr(A, ColumnNorm())`",
    "svd_values" => "`svdvals`",
    "svd_thin" => "`svd`",
    "eigh" => "self-adjoint `eigen`",
    "eig" => "general `eigen`",
]

struct Row
    backend::String
    threads::Int
    op::String
    eltype::String
    n::Int
    min_ns::Float64
    median_ns::Float64
    allocs::Int
    bytes::Int
    residual::Any
end

rows = Row[]
files = filter(f -> endswith(f, ".toml"), readdir(DIR; join=true))
for file in files
    data = TOML.parsefile(file)
    backend = data["backend"]
    threads = Int(data["blas_threads"])
    for c in data["cases"]
        push!(
            rows,
            Row(
                backend,
                threads,
                c["op"],
                c["eltype"],
                Int(c["n"]),
                Float64(c["minimum_ns"]),
                Float64(c["median_ns"]),
                Int(c["allocs"]),
                Int(c["bytes"]),
                c["residual"],
            ),
        )
    end
end
isempty(rows) && error("no results found in $(DIR)")

sort!(rows; by=r -> (r.threads, r.eltype, r.op, r.n, r.backend))

lookup = Dict(
    (r.backend, r.threads, r.op, r.eltype, r.n) => r for r in rows
)

csv = open(joinpath(DIR, "summary.csv"), "w")
println(
    csv,
    "backend,threads,op,eltype,n,min_ms,median_ms,median_speedup_vs_openblas,allocs,bytes,residual",
)
for r in rows
    base = get(lookup, ("openblas", r.threads, r.op, r.eltype, r.n), nothing)
    speedup = base === nothing ? NaN : base.median_ns / r.median_ns
    @printf(
        csv,
        "%s,%d,%s,%s,%d,%.6g,%.6g,%.4f,%d,%d,%s\n",
        r.backend,
        r.threads,
        r.op,
        r.eltype,
        r.n,
        r.min_ns / 1e6,
        r.median_ns / 1e6,
        speedup,
        r.allocs,
        r.bytes,
        r.residual,
    )
end
close(csv)

fmt_ms(x) = x < 1.0 ? @sprintf("%.3f", x) : @sprintf("%.2f", x)
thread_list = sort(unique(r.threads for r in rows))

md = open(joinpath(DIR, "summary.md"), "w")
println(md, "# Colmat.jl backend comparison\n")
println(
    md,
    "Median wall time per call in milliseconds (BenchmarkTools, single Julia thread). ",
    "`Colmat/OpenBLAS` is the time ratio, so values below 1.0 mean Colmat is faster. ",
    "Timings include the allocations and input copies performed by the public Julia API.\n",
)
println(
    md,
    "Families follow `tensor4all/colmat-benchmark`: `gemm`, `lu`, `cholesky`, ",
    "`qr_factor`, `qr` (explicit Q), `svd_values`, `svd_thin`, `eigh` and `eig`, ",
    "plus column-pivoted QR.\n",
)
for threads in thread_list
    println(md, "## Threads = $(threads)\n")
    for T in ("Float64", "ComplexF64")
        println(md, "### $(T)\n")
        println(
            md,
            "| Operation | n | OpenBLAS | MKL | Colmat | MKL/OpenBLAS | Colmat/OpenBLAS |",
        )
        println(md, "|---|---:|---:|---:|---:|---:|---:|")
        for (op, label) in OPS
            ns = sort(unique([
                r.n for r in rows if
                r.op == op && r.eltype == T && r.threads == threads
            ]))
            for n in ns
                vals = Dict{String,Float64}()
                for b in BACKENDS
                    r = get(lookup, (b, threads, op, T, n), nothing)
                    r !== nothing && (vals[b] = r.median_ns / 1e6)
                end
                o = get(vals, "openblas", NaN)
                m = get(vals, "mkl", NaN)
                c = get(vals, "colmat", NaN)
                println(
                    md,
                    "| $(label) | $(n) | $(fmt_ms(o)) | $(fmt_ms(m)) | $(fmt_ms(c)) | ",
                    @sprintf("%.2f", m / o),
                    " | ",
                    @sprintf("%.2f", c / o),
                    " |",
                )
            end
        end
        println(md)
    end
end

# Geometric mean ratios per operation across sizes, per thread setting.
println(md, "## Geometric mean time ratio across sizes\n")
println(md, "| Threads | Operation | Type | MKL/OpenBLAS | Colmat/OpenBLAS |")
println(md, "|---:|---|---|---:|---:|")
gm(v) = isempty(v) ? NaN : exp(mean(log.(v)))
for threads in thread_list, (op, label) in OPS, T in ("Float64", "ComplexF64")
    ratios_m = Float64[]
    ratios_c = Float64[]
    for n in sort(unique([
        r.n for r in rows if r.op == op && r.eltype == T && r.threads == threads
    ]))
        o = get(lookup, ("openblas", threads, op, T, n), nothing)
        m = get(lookup, ("mkl", threads, op, T, n), nothing)
        c = get(lookup, ("colmat", threads, op, T, n), nothing)
        o === nothing && continue
        m !== nothing && push!(ratios_m, m.median_ns / o.median_ns)
        c !== nothing && push!(ratios_c, c.median_ns / o.median_ns)
    end
    isempty(ratios_m) && isempty(ratios_c) && continue
    println(
        md,
        "| $(threads) | $(label) | $(T) | ",
        @sprintf("%.2f", gm(ratios_m)),
        " | ",
        @sprintf("%.2f", gm(ratios_c)),
        " |",
    )
end
println(md)
close(md)

println("wrote ", joinpath(DIR, "summary.md"))
println("wrote ", joinpath(DIR, "summary.csv"))
