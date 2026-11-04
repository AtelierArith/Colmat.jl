using RustCall
println("Julia: ", VERSION)
println("RustCall: ", pkgversion(RustCall))
info = RustCall.scan_crate(@__DIR__)
println("Methods: ", [(s.name,[m.name for m in s.methods]) for s in info.julia_structs])
bindings = RustCall.load_crate_bindings(@__DIR__)
@assert bindings.read(bindings.Destination(),bindings.Counter()) == 42
