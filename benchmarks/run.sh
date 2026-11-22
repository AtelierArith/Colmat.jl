#!/usr/bin/env bash
# Run the Colmat backend comparison with no external harness.
#
#   ./benchmarks/run.sh                 # OpenBLAS, MKL, Colmat; 1/4/8 threads
#   ./benchmarks/run.sh --quick         # tiny smoke run (~1 minute)
#   ./benchmarks/run.sh --build         # rebuild Colmat first
#   ./benchmarks/run.sh --threads 1 8 --backends openblas colmat
#
# Results: <out>/summary.md, <out>/summary.csv, one TOML per backend/thread,
# and the process logs under <out>/logs.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH="$ROOT/benchmarks/backends"
JULIA="${JULIA:-julia}"
BACKENDS=(${COLMAT_BENCH_BACKENDS:-openblas mkl colmat})
THREADS=(${COLMAT_BENCH_THREADS_LIST:-1 4 8})
SAMPLES="${COLMAT_BENCH_SAMPLES:-5}"
QUICK="${COLMAT_BENCH_QUICK:-false}"
BUILD=0
AFFINITY="${COLMAT_BENCH_AFFINITY:-1}"
OUT=""
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"

usage() {
    sed -n '2,10p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build) BUILD=1; shift ;;
        --quick) QUICK=true; shift ;;
        --no-affinity) AFFINITY=0; shift ;;
        --samples) SAMPLES="${2:?--samples needs a number}"; shift 2 ;;
        --threads)
            shift
            THREADS=()
            while [[ $# -gt 0 && "$1" != --* ]]; do THREADS+=("$1"); shift; done
            ;;
        --backends)
            shift
            BACKENDS=()
            while [[ $# -gt 0 && "$1" != --* ]]; do BACKENDS+=("$1"); shift; done
            ;;
        --out) OUT="${2:?--out needs a directory}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ ${#BACKENDS[@]} -gt 0 ]] || { echo "no backend selected" >&2; exit 2; }
[[ ${#THREADS[@]} -gt 0 ]] || { echo "no thread count selected" >&2; exit 2; }

if [[ -z "$OUT" ]]; then
    OUT="$BENCH/results/runs/$TIMESTAMP"
fi
mkdir -p "$OUT" "$OUT/logs"
if compgen -G "$OUT/*.toml" >/dev/null; then
    echo "ERROR: $OUT already contains TOML results; choose a fresh --out" >&2
    exit 2
fi

echo "== Colmat backend benchmark =="
echo "root:     $ROOT"
echo "output:   $OUT"
echo "backends: ${BACKENDS[*]}"
echo "threads:  ${THREADS[*]}"
echo "samples:  $SAMPLES"
echo "quick:    $QUICK"
echo

# --- environment -----------------------------------------------------------
if [[ "$BUILD" == 1 ]] || ! compgen -G "$ROOT/deps/usr/lib/libcolmat_blas.*" >/dev/null; then
    echo "-- building Colmat"
    "$JULIA" --startup-file=no --project="$ROOT" \
        -e 'using Pkg; Pkg.instantiate(); Pkg.build("Colmat")'
fi
echo "-- instantiating the benchmark environment"
"$JULIA" --startup-file=no --project="$BENCH" \
    -e 'using Pkg; Pkg.instantiate()'

LIB="$(compgen -G "$ROOT/deps/usr/lib/libcolmat_blas.*" | head -1 || true)"
LIB_SHA="n/a"
if [[ -n "$LIB" ]] && command -v sha256sum >/dev/null 2>&1; then
    LIB_SHA="$(sha256sum "$LIB" | awk '{print $1}')"
fi
GIT_COMMIT="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
GIT_DIRTY="$(git -C "$ROOT" status --porcelain --untracked-files=no 2>/dev/null | head -1 || true)"
[[ -n "$GIT_DIRTY" ]] && GIT_COMMIT="$GIT_COMMIT-dirty"

# --- CPU affinity ----------------------------------------------------------
NCPU="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)"
if [[ "$(uname -s)" == "Darwin" ]] || ! command -v taskset >/dev/null 2>&1; then
    AFFINITY=0
fi
CPUS=()
if [[ "$AFFINITY" == 1 ]]; then
    # Build the allowed CPU list, not just a count: a container or scheduler
    # allocation may allow only 8-15, where a hard-coded "0-N" mask is invalid.
    ALLOWED="$(taskset -pc $$ 2>/dev/null | sed 's/.*: //' || true)"
    if [[ -n "$ALLOWED" ]]; then
        IFS=',' read -ra PARTS <<< "$ALLOWED"
        for part in "${PARTS[@]}"; do
            if [[ "$part" == *-* ]]; then
                lo="${part%-*}"
                hi="${part#*-}"
                for ((i=lo; i<=hi; i++)); do CPUS+=("$i"); done
            else
                CPUS+=("$part")
            fi
        done
    fi
fi
if [[ ${#CPUS[@]} -eq 0 ]]; then
    for ((i=0; i<NCPU; i++)); do CPUS+=("$i"); done
fi
NCPU="${#CPUS[@]}"
CPU_START="${COLMAT_BENCH_CPU_START:-0}"
(( CPU_START < 0 )) && CPU_START=0
(( CPU_START >= NCPU )) && CPU_START=0
mask_for() {
    local t="$1"
    local last=$(( CPU_START + t ))
    (( last >= NCPU )) && last=$(( NCPU - 1 ))
    local -a selected=()
    for ((i=CPU_START; i<=last; i++)); do selected+=("${CPUS[$i]}"); done
    (IFS=,; echo "${selected[*]}")
}

run_one() {
    local backend="$1" threads="$2"
    local toml="$OUT/${backend}_t${threads}.toml"
    local log="$OUT/logs/${backend}_t${threads}.log"
    local mask
    mask="$(mask_for "$threads")"
    echo "-- $backend, ${threads}T (mask $mask)"
    local -a envkv=(
        COLMAT_BENCH_BACKEND="$backend"
        COLMAT_BENCH_THREADS="$threads"
        COLMAT_BENCH_SAMPLES="$SAMPLES"
        COLMAT_BENCH_QUICK="$QUICK"
        COLMAT_BENCH_OUTPUT="$toml"
        COLMAT_BENCH_GIT_COMMIT="$GIT_COMMIT"
        COLMAT_BENCH_LIB_SHA="$LIB_SHA"
    )
    if [[ "$AFFINITY" == 1 ]]; then
        if ! taskset -c "$mask" env "${envkv[@]}" \
            "$JULIA" --startup-file=no -t 1 --project="$BENCH" "$BENCH/run.jl" \
            >"$log" 2>&1; then
            echo "   FAILED; see $log" >&2
            return 1
        fi
    else
        if ! env "${envkv[@]}" \
            "$JULIA" --startup-file=no -t 1 --project="$BENCH" "$BENCH/run.jl" \
            >"$log" 2>&1; then
            echo "   FAILED; see $log" >&2
            return 1
        fi
    fi
}

failed=0
for threads in "${THREADS[@]}"; do
    for backend in "${BACKENDS[@]}"; do
        run_one "$backend" "$threads" || failed=1
    done
done

if [[ "$failed" != 0 ]]; then
    echo "one or more backend runs failed; generated files are in $OUT" >&2
    exit 1
fi

echo "-- summarizing"
COLMAT_BENCH_RESULTS_DIR="$OUT" \
    "$JULIA" --startup-file=no --project="$BENCH" "$BENCH/summarize.jl"

echo
echo "done"
echo "  markdown: $OUT/summary.md"
echo "  csv:      $OUT/summary.csv"
echo "  raw:      $OUT/*.toml"
