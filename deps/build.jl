using Libdl
using TOML

# Use RustToolChain's cargo when available; fall back to system cargo otherwise.
# Pkg.build runs deps/build.jl in a sandbox that may not see newly-added deps,
# so the fallback keeps the build working in that case.
const CARGO = let
    try
        RustToolChain = Base.require(
            Base.PkgId(
                Base.UUID("e9dc52e2-edb8-4742-9783-5e542d30dbb5"),
                "RustToolChain",
            ),
        )
        # Use invokelatest because the dynamically-loaded module's methods are
        # newer than the world age of this top-level code.
        cmd = Base.invokelatest(RustToolChain.cargo)
        # When using RustToolChain's own artifact cargo, override any user
        # rustc-wrapper settings from cargo config so broken wrappers do not
        # affect the build.
        if any(depot -> startswith(cmd.exec[1], joinpath(depot, "artifacts")), DEPOT_PATH)
            env = copy(cmd.env)
            push!(env, "RUSTC_WRAPPER=")
            push!(env, "RUSTC_WORKSPACE_WRAPPER=")
            setenv(Cmd(cmd.exec), env)
        else
            cmd
        end
    catch
        Sys.which("cargo") === nothing && error(
            "No cargo executable found. Install Rust or ensure RustToolChain is available in the build environment.",
        )
        `cargo`
    end
end

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
run(`$CARGO build --release --locked --manifest-path $manifest --target-dir $target`)
const destination = joinpath(@__DIR__, "usr", "lib")
mkpath(destination)
const library = Sys.iswindows() ? "colmat_blas.dll" : "libcolmat_blas." * Libdl.dlext
cp(
    joinpath(target, "release", library),
    joinpath(destination, "libcolmat_blas." * Libdl.dlext);
    force=true,
)
println("Installed ", joinpath(destination, "libcolmat_blas." * Libdl.dlext))
