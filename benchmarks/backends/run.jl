# One backend, one thread budget, one process.
#
#   COLMAT_BENCH_BACKEND=colmat \
#   COLMAT_BENCH_THREADS=4 \
#   COLMAT_BENCH_OUTPUT=results/colmat_t4.toml \
#     julia --startup-file=no -t 1 --project=benchmarks/backends benchmarks/backends/run.jl
#
# The suite is self-contained: gemm, lu, cholesky, qr factor, explicit Q,
# column-pivoted QR, svdvals, thin svd, self-adjoint eigen and general eigen,
# f64/c64, fixed seeds. It writes one TOML; summarize.jl turns a directory of
# these files into CSV and Markdown.

using LinearAlgebra
using Random
using Statistics
using TOML
using Dates
using Printf
using BenchmarkTools
import MKL_jll

const BACKEND = get(ENV, "COLMAT_BENCH_BACKEND", "colmat")
const THREADS = parse(Int, get(ENV, "COLMAT_BENCH_THREADS", "1"))
const OUTPUT = get(
    ENV,
    "COLMAT_BENCH_OUTPUT",
    joinpath(@__DIR__, "results", "$(BACKEND)_t$(THREADS).toml"),
)
const SAMPLES = parse(Int, get(ENV, "COLMAT_BENCH_SAMPLES", "5"))
const QUICK = get(ENV, "COLMAT_BENCH_QUICK", "false") == "true"

const SMALL = [32, 64, 96]
const MID = [128, 256, 512]
const GEMM_SIZES = QUICK ? [32, 128] : vcat(SMALL, MID, [768, 1024])
const FACTOR_SIZES = QUICK ? [32, 128] : vcat(SMALL, MID, [1024])
const EIG_SIZES = QUICK ? [32, 128] : vcat(SMALL, MID)

# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

function select_backend(name)
    if name == "openblas"
        # Julia's default backend.
    elseif name == "mkl"
        @eval using MKL
    elseif name == "colmat"
        @eval using Colmat
    elseif name == "apple"
        @eval using AppleAccelerate
    else
        error("unknown backend $(name); expected openblas, mkl, colmat or apple")
    end
    return nothing
end

select_backend(BACKEND)
BLAS.set_num_threads(THREADS)
# MKL's LAPACK threading is not covered by the LBT setter; without this a
# "1 thread" run still uses MKL's default (8) for potrf/gesdd/syevd.
# Accelerate's API is a single/multi toggle, so record what it selected.
vendor_threads = BLAS.get_num_threads()
if BACKEND == "mkl"
    ccall((:MKL_Set_Num_Threads, MKL_jll.libmkl_rt), Cvoid, (Cint,), THREADS)
    vendor_threads = THREADS
elseif BACKEND == "apple"
    AppleAccelerate.set_num_threads(THREADS)
    vendor_threads = Int(AppleAccelerate.get_num_threads())
end

# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

"""
    measure(f; samples, target)

Warm `f`, choose an evaluation count so one sample lasts about `target`
seconds, collect `samples` BenchmarkTools samples, and report per-evaluation
times in nanoseconds. `GC.gc()` runs once before the case so the median is not
carried over from the previous case's allocations.
"""
function measure(f; samples=SAMPLES, target=0.02)
    f()
    t = Inf
    for _ in 1:3
        t = min(t, @elapsed f())
    end
    evals = clamp(round(Int, target / max(t, 1.0e-9)), 1, 64)
    GC.gc()
    trial = run(@benchmarkable($f()); samples=samples, evals=evals, seconds=600.0)
    return (
        minimum_ns = minimum(trial.times),
        median_ns = median(trial.times),
        mean_ns = mean(trial.times),
        samples = length(trial.times),
        evals = evals,
        allocs = trial.allocs,
        bytes = trial.memory,
    )
end

randmat(rng, ::Type{T}, n) where {T} = randn(rng, T, n, n)

# ---------------------------------------------------------------------------
# Correctness probes (outside the timed region)
# ---------------------------------------------------------------------------

function relres(x, y)
    d = norm(x - y)
    s = norm(y)
    return s == 0 ? d : d / s
end

"""Entry check for GEMM that does not use a BLAS backend."""
function gemm_entry_error(C, A, B, rng)
    n = size(A, 1)
    err = 0.0
    for _ in 1:3
        i = rand(rng, 1:n)
        j = rand(rng, 1:n)
        s = zero(eltype(C))
        @inbounds for k in 1:n
            s += A[i, k] * B[k, j]
        end
        err = max(err, abs(s - C[i, j]) / max(abs(s), eps(real(float(one(eltype(C)))))))
    end
    return err
end

# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

const RESULTS = Any[]

function record(; op, T, n, f, probe)
    local residual
    try
        residual = probe()
    catch err
        residual = "probe failed: $(sprint(showerror, err))"
    end
    shown = residual isa AbstractString ? residual : @sprintf("%.3e", residual)
    @printf("  %-12s %-12s n=%-5d residual=%s\n", op, string(T), n, shown)
    timing = measure(f)
    push!(
        RESULTS,
        Dict{String,Any}(
            "op" => op,
            "eltype" => string(T),
            "n" => n,
            "residual" => residual,
            "minimum_ns" => timing.minimum_ns,
            "median_ns" => timing.median_ns,
            "mean_ns" => timing.mean_ns,
            "samples" => timing.samples,
            "evals" => timing.evals,
            "allocs" => timing.allocs,
            "bytes" => timing.bytes,
        ),
    )
    return nothing
end

probe_rng = MersenneTwister(0xD00D)

for (ti, T) in enumerate((Float64, ComplexF64))
    for n in GEMM_SIZES
        rng = MersenneTwister(1_000_003 * ti + n)
        A = randmat(rng, T, n)
        B = randmat(rng, T, n)
        C = Matrix{T}(undef, n, n)
        mul!(C, A, B)
        record(;
            op = "gemm",
            T,
            n,
            f = () -> mul!(C, A, B),
            probe = () -> gemm_entry_error(C, A, B, probe_rng),
        )
    end

    for n in FACTOR_SIZES
        rng = MersenneTwister(2_000_003 * ti + n)
        A = randmat(rng, T, n)
        # Positive definite for the Cholesky row; Hermitian for eigh.
        H = Hermitian(A * A' + n * I)

        record(;
            op = "lu",
            T,
            n,
            f = () -> lu(A),
            probe = () -> begin
                F = lu(A)
                relres(F.L * F.U, A[F.p, :])
            end,
        )
        record(;
            op = "cholesky",
            T,
            n,
            f = () -> cholesky(H),
            probe = () -> begin
                F = cholesky(H)
                relres(F.L * F.L', Matrix(H))
            end,
        )
        record(;
            op = "qr_factor",
            T,
            n,
            f = () -> qr(A),
            probe = () -> begin
                F = qr(A)
                relres(Matrix(F.Q) * F.R, A)
            end,
        )
        record(;
            op = "qr",
            T,
            n,
            f = () -> Matrix(qr(A).Q),
            probe = () -> begin
                Q = Matrix(qr(A).Q)
                relres(Q' * Q, Matrix{T}(I, n, n))
            end,
        )
        record(;
            op = "qr_pivot",
            T,
            n,
            f = () -> qr(A, ColumnNorm()),
            probe = () -> begin
                F = qr(A, ColumnNorm())
                relres(Matrix(F.Q) * F.R, A[:, F.p])
            end,
        )
        record(;
            op = "svd_values",
            T,
            n,
            f = () -> svdvals(A),
            probe = () -> begin
                S = svdvals(A)
                S2 = svd(A).S
                norm(S - S2) / norm(S2)
            end,
        )
        record(;
            op = "svd_thin",
            T,
            n,
            f = () -> svd(A),
            probe = () -> begin
                F = svd(A)
                relres(F.U * Diagonal(F.S) * F.Vt, A)
            end,
        )
        record(;
            op = "eigh",
            T,
            n,
            f = () -> eigen(H),
            probe = () -> begin
                E = eigen(H)
                relres(Matrix(H) * E.vectors, E.vectors * Diagonal(E.values))
            end,
        )
    end

    for n in EIG_SIZES
        rng = MersenneTwister(3_000_003 * ti + n)
        A = randmat(rng, T, n)
        record(;
            op = "eig",
            T,
            n,
            f = () -> eigen(A),
            probe = () -> begin
                E = eigen(A)
                relres(A * E.vectors, E.vectors * Diagonal(E.values))
            end,
        )
    end
end

# ---------------------------------------------------------------------------
# Metadata and output
# ---------------------------------------------------------------------------

const SYMBOLS = [
    "dgemm_", "zgemm_",
    "dgetrf_", "zgetrf_", "dpotrf_", "zpotrf_",
    "dgeqrt_", "zgeqrt_", "dgeqrf_", "zgeqrf_", "dgeqp3_", "zgeqp3_",
    "dsyevr_", "zheevr_", "dgeevx_", "zgeevx_",
    "dgesdd_", "zgesdd_", "dgesvd_", "zgesvd_",
]

function forwarding_target(s)
    lib = BLAS.lbt_find_backing_library(s, :ilp64)
    return lib === nothing ? "none" : basename(lib.libname)
end
forwarding = Dict{String,Any}(s => forwarding_target(s) for s in SYMBOLS)

config = BLAS.get_config()
blas_config = join(
    ["[$(l.interface)] $(basename(l.libname))" for l in config.loaded_libs],
    ", ",
)

result = Dict{String,Any}(
    "backend" => BACKEND,
    "julia_version" => string(VERSION),
    "machine" => Sys.MACHINE,
    "cpu" => Sys.CPU_NAME,
    "threads" => THREADS,
    "blas_num_threads" => BLAS.get_num_threads(),
    "vendor_threads" => vendor_threads,
    "blas_config" => blas_config,
    "forwarding" => forwarding,
    "samples" => SAMPLES,
    "quick" => QUICK,
    "timestamp" => string(Dates.now()),
    "git_commit" => get(ENV, "COLMAT_BENCH_GIT_COMMIT", "unknown"),
    "library_sha256" => BACKEND == "colmat" ? get(ENV, "COLMAT_BENCH_LIB_SHA", "n/a") : "n/a",
    "library" => BACKEND == "colmat" ? Colmat.backend_library() : "",
    "cases" => RESULTS,
)

mkpath(dirname(OUTPUT))
open(OUTPUT, "w") do io
    TOML.print(io, result)
end
println("wrote ", OUTPUT)
