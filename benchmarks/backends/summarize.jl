# Summarize one directory of run.jl TOMLs into summary.csv and summary.md.
#
#   COLMAT_BENCH_RESULTS_DIR=benchmarks/backends/results/20260101-120000 \
#     julia --startup-file=no --project=benchmarks/backends benchmarks/backends/summarize.jl

using TOML
using Printf
using Statistics

const DIR = get(
    ENV,
    "COLMAT_BENCH_RESULTS_DIR",
    joinpath(@__DIR__, "results"),
)
const BACKENDS = ["openblas", "mkl", "aocl", "colmat", "apple"]
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
    mean_ns::Float64
    samples::Int
    evals::Int
    allocs::Int
    bytes::Int
    residual::Any
end

struct Meta
    backend::String
    threads::Int
    julia_version::String
    cpu::String
    machine::String
    blas_config::String
    samples::Int
    vendor_threads::Int
    git_commit::String
    library_sha256::String
    library::String
end

rows = Row[]
meta = Dict{Tuple{String,Int},Meta}()
files = sort(filter(f -> endswith(f, ".toml"), readdir(DIR; join=true)))
for file in files
    data = TOML.parsefile(file)
    backend = get(data, "backend", "unknown")
    threads = Int(get(data, "threads", get(data, "blas_threads", 1)))
    meta[(backend, threads)] = Meta(
        backend,
        threads,
        string(get(data, "julia_version", "?")),
        string(get(data, "cpu", "?")),
        string(get(data, "machine", "?")),
        string(get(data, "blas_config", "?")),
        Int(get(data, "samples", 0)),
        Int(get(data, "vendor_threads", get(data, "blas_num_threads", threads))),
        string(get(data, "git_commit", "unknown")),
        string(get(data, "library_sha256", "n/a")),
        string(get(data, "library", "")),
    )
    for c in data["cases"]
        push!(
            rows,
            Row(
                backend,
                threads,
                string(c["op"]),
                string(c["eltype"]),
                Int(c["n"]),
                Float64(c["minimum_ns"]),
                Float64(c["median_ns"]),
                Float64(c["mean_ns"]),
                Int(c["samples"]),
                Int(c["evals"]),
                Int(c["allocs"]),
                Int(c["bytes"]),
                c["residual"],
            ),
        )
    end
end
isempty(rows) && error("no TOML cases found in $(DIR)")

sort!(rows; by = r -> (r.threads, r.eltype, r.op, r.n, r.backend))
lookup = Dict((r.backend, r.threads, r.op, r.eltype, r.n) => r for r in rows)
thread_list = sort(unique(r.threads for r in rows))

# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

csv = open(joinpath(DIR, "summary.csv"), "w")
println(
    csv,
    "backend,threads,op,eltype,n,min_ms,median_ms,mean_ms,median_ratio_to_openblas,min_ratio_to_openblas,samples,evals,allocs,bytes,residual",
)
for r in rows
    base = get(lookup, ("openblas", r.threads, r.op, r.eltype, r.n), nothing)
    med_ratio = base === nothing ? NaN : base.median_ns / r.median_ns
    min_ratio = base === nothing ? NaN : base.min_ns / r.min_ns
    @printf(
        csv,
        "%s,%d,%s,%s,%d,%.6g,%.6g,%.6g,%.4f,%.4f,%d,%d,%d,%d,%s\n",
        r.backend,
        r.threads,
        r.op,
        r.eltype,
        r.n,
        r.min_ns / 1e6,
        r.median_ns / 1e6,
        r.mean_ns / 1e6,
        med_ratio,
        min_ratio,
        r.samples,
        r.evals,
        r.allocs,
        r.bytes,
        r.residual,
    )
end
close(csv)

# ---------------------------------------------------------------------------
# Markdown helpers
# ---------------------------------------------------------------------------

fmt_ms(x) = isnan(x) ? "—" : (x < 1.0 ? @sprintf("%.3f", x) : @sprintf("%.2f", x))
ratio(x, y) = (isnan(x) || isnan(y) || y == 0) ? "—" : @sprintf("%.2f", x / y)
fmt_ratio(x) = isnan(x) ? "—" : @sprintf("%.2f", x)
gm(v) = isempty(v) ? NaN : exp(mean(log.(v)))

function medians_table(md, threads, T, op_label, op)
    println(
        md,
        "| Operation | n | OpenBLAS | MKL | AOCL | AppleAccelerate | Colmat | Colmat/OpenBLAS | Colmat/MKL | Colmat/AOCL | Colmat/Apple |",
    )
    println(md, "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
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
        x = get(vals, "aocl", NaN)
        p = get(vals, "apple", NaN)
        c = get(vals, "colmat", NaN)
        println(
            md,
            "| $(op_label) | $(n) | $(fmt_ms(o)) | $(fmt_ms(m)) | $(fmt_ms(x)) | $(fmt_ms(p)) | $(fmt_ms(c)) | ",
            ratio(c, o),
            " | ",
            ratio(c, m),
            " | ",
            ratio(c, x),
            " | ",
            ratio(c, p),
            " |",
        )
    end
    return nothing
end

function aggregate_table(op::String, T::String, threads::Int; field=:median_ns)
    ratios_o = Float64[]
    ratios_m = Float64[]
    ratios_x = Float64[]
    ratios_a = Float64[]
    for n in sort(unique([
        r.n for r in rows if
        r.op == op && r.eltype == T && r.threads == threads
    ]))
        o = get(lookup, ("openblas", threads, op, T, n), nothing)
        m = get(lookup, ("mkl", threads, op, T, n), nothing)
        x = get(lookup, ("aocl", threads, op, T, n), nothing)
        c = get(lookup, ("colmat", threads, op, T, n), nothing)
        a = get(lookup, ("apple", threads, op, T, n), nothing)
        c === nothing && continue
        cbase = field === :median_ns ? c.median_ns : c.min_ns
        if o !== nothing
            ob = field === :median_ns ? o.median_ns : o.min_ns
            ob != 0 && push!(ratios_o, cbase / ob)
        end
        if m !== nothing
            mb = field === :median_ns ? m.median_ns : m.min_ns
            mb != 0 && push!(ratios_m, cbase / mb)
        end
        if x !== nothing
            xb = field === :median_ns ? x.median_ns : x.min_ns
            xb != 0 && push!(ratios_x, cbase / xb)
        end
        if a !== nothing
            ab = field === :median_ns ? a.median_ns : a.min_ns
            ab != 0 && push!(ratios_a, cbase / ab)
        end
    end
    return (gm(ratios_o), gm(ratios_m), gm(ratios_x), gm(ratios_a))
end

# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

md = open(joinpath(DIR, "summary.md"), "w")
println(md, "# Colmat backend comparison\n")
println(md, "Median wall time per call in milliseconds, one process per backend and thread budget. `Colmat/<backend>` is the time ratio, so values below 1.0 mean Colmat is faster. Timings include the allocations and input copies of the public Julia API.\n")
println(md, "Source directory: `$(DIR)`\n")
if !isempty(meta)
    first_meta = first(sort(collect(values(meta)); by = m -> (m.threads, m.backend)))
    println(md, "Julia $(first_meta.julia_version) on $(first_meta.machine), `$(first_meta.cpu)`. Commit `$(first_meta.git_commit)`.\n")
end

println(md, "## Environment\n")
println(md, "| backend | threads | vendor threads | BLAS library | samples | library sha256 |")
println(md, "|---|---:|---:|---|---:|---|")
for (key, m) in sort(collect(meta); by = kv -> (kv[1][2], kv[1][1]))
    sha = m.library_sha256 == "n/a" ? "" : m.library_sha256[1:min(end, 12)]
    println(md, "| $(m.backend) | $(m.threads) | $(m.vendor_threads) | `$(m.blas_config)` | $(m.samples) | `$(sha)` |")
end
println(md)

for threads in thread_list
    println(md, "## Threads = $(threads)\n")
    for T in ("Float64", "ComplexF64")
        println(md, "### $(T)\n")
        for (op, label) in OPS
            medians_table(md, threads, T, label, op)
        end
        println(md)
    end
end

println(md, "## Geometric mean time ratio across sizes (median)\n")
println(md, "The median includes Julia GC/allocation. The minimum-time table after this one separates that from the steady-state kernel/adapter cost.\n")
println(md, "| Threads | Operation | Type | Colmat/OpenBLAS | Colmat/MKL | Colmat/AOCL | Colmat/Apple |")
println(md, "|---:|---|---|---:|---:|---:|---:|")
for threads in thread_list, (op, label) in OPS, T in ("Float64", "ComplexF64")
    o, m, x, a = aggregate_table(op, T, threads; field=:median_ns)
    (isnan(o) && isnan(m) && isnan(x) && isnan(a)) && continue
    println(md, "| $(threads) | $(label) | $(T) | $(fmt_ratio(o)) | $(fmt_ratio(m)) | $(fmt_ratio(x)) | $(fmt_ratio(a)) |")
end
println(md)

println(md, "## Geometric mean time ratio across sizes (minimum)\n")
println(md, "Each cell is the geometric mean of the per-size minimum times, which removes most of the GC/allocation variance that dominates microsecond cells.\n")
println(md, "| Threads | Operation | Type | Colmat/OpenBLAS | Colmat/MKL | Colmat/AOCL | Colmat/Apple |")
println(md, "|---:|---|---|---:|---:|---:|---:|")
for threads in thread_list, (op, label) in OPS, T in ("Float64", "ComplexF64")
    o, m, x, a = aggregate_table(op, T, threads; field=:min_ns)
    (isnan(o) && isnan(m) && isnan(x) && isnan(a)) && continue
    println(md, "| $(threads) | $(label) | $(T) | $(fmt_ratio(o)) | $(fmt_ratio(m)) | $(fmt_ratio(x)) | $(fmt_ratio(a)) |")
end
println(md)

failed = [r for r in rows if !(r.residual isa Real && isfinite(r.residual))]
println(md, "## Checks\n")
println(md, "Every case runs a reconstruction/orthogonality/entry probe outside the timed region. $(isempty(failed) ? "All probes returned a finite residual." : "$(length(failed)) probes failed; see summary.csv.")\n")
println(md, "Times are the best of one BenchmarkTools run: one evaluation per sample for slow calls, several for fast calls (about 20 ms of work per sample).")
close(md)

println("wrote ", joinpath(DIR, "summary.md"))
println("wrote ", joinpath(DIR, "summary.csv"))
