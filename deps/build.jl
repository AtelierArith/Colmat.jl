using Libdl
using TOML
using RustToolChain: cargo

const source = joinpath(@__DIR__, "colmat-rs")
const upstream = TOML.parsefile(joinpath(@__DIR__, "upstream.toml"))
const revision = upstream["revision"]

if !ispath(source)
    repository = upstream["repository"]
    run(`git clone --no-checkout $repository $source`)
    run(`git -C $source checkout --detach $revision`)
end
actual = strip(read(`git -C $source rev-parse HEAD`, String))
actual == revision ||
    error("Colmat source is at $actual; expected $revision from deps/upstream.toml")
isempty(read(`git -C $source status --porcelain --untracked-files=no`, String)) ||
    error("Colmat source has local changes; restore the pinned source before building")
println("Building Colmat BLAS with Rust colmat revision ", actual)

const manifest = joinpath(@__DIR__, "rust", "Cargo.toml")
const target = joinpath(@__DIR__, "rust", "target")

# Disable user rustc-wrapper settings so a broken wrapper (for example from
# ~/.cargo/config.toml) cannot break the build. Environment variables take
# precedence over cargo config.
withenv("RUSTC_WRAPPER" => "", "RUSTC_WORKSPACE_WRAPPER" => "") do
    run(`$(cargo()) build --release --locked --manifest-path $manifest --target-dir $target`)
end

const destination = joinpath(@__DIR__, "usr", "lib")
mkpath(destination)
const library = Sys.iswindows() ? "colmat_blas.dll" : "libcolmat_blas." * Libdl.dlext
cp(
    joinpath(target, "release", library),
    joinpath(destination, "libcolmat_blas." * Libdl.dlext);
    force=true,
)
println("Installed ", joinpath(destination, "libcolmat_blas." * Libdl.dlext))
