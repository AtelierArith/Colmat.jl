using RustCall

const source = joinpath(@__DIR__, "t4a-faer")
const repository = "https://github.com/tensor4all/faer.git"

if !ispath(source)
    run(`git clone --depth=1 $repository $source`)
else
    isdir(joinpath(source, ".git")) || error("$source is not a Git checkout")
    isempty(read(`git -C $source status --porcelain`, String)) ||
        error("$source has local changes; commit or move them before rebuilding")
    # Follow the remote's current default branch, including branch renames.
    run(`git -C $source fetch --depth=1 origin HEAD`)
    run(`git -C $source checkout --detach FETCH_HEAD`)
end
println("Building colmat revision ", strip(read(`git -C $source rev-parse HEAD`, String)))

# Use the same crate and cache as the package's @rust_crate binding.
RustCall.load_crate_bindings(joinpath(@__DIR__, "rust"))
