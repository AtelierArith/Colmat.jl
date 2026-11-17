using Libdl
using TOML

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

RustToolChain = Base.require(
    Base.PkgId(
        Base.UUID("e9dc52e2-edb8-4742-9783-5e542d30dbb5"),
        "RustToolChain",
    ),
)
# Use invokelatest because the dynamically-loaded module's methods are newer
# than the world age of this top-level code.
cargo = Base.invokelatest(RustToolChain.cargo)
# When using RustToolChain's own artifact cargo, override any user rustc-wrapper
# settings from cargo config so broken wrappers do not affect the build.
if any(depot -> startswith(cargo.exec[1], joinpath(depot, "artifacts")), DEPOT_PATH)
    env = copy(cargo.env)
    push!(env, "RUSTC_WRAPPER=")
    push!(env, "RUSTC_WORKSPACE_WRAPPER=")
    cargo = setenv(Cmd(cargo.exec), env)
end

run(`$cargo build --release --locked --manifest-path $manifest --target-dir $target`)
const destination = joinpath(@__DIR__, "usr", "lib")
mkpath(destination)
const library = Sys.iswindows() ? "colmat_blas.dll" : "libcolmat_blas." * Libdl.dlext
cp(
    joinpath(target, "release", library),
    joinpath(destination, "libcolmat_blas." * Libdl.dlext);
    force=true,
)
println("Installed ", joinpath(destination, "libcolmat_blas." * Libdl.dlext))
