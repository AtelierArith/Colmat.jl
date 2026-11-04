# RustCall glob-imported borrowed owner

Reported as [RustCall.jl #569](https://github.com/AtelierArith/RustCall.jl/issues/569).

RustCall 0.9.2 rejects a valid borrowed opaque-owner argument when its name is imported via `use super::*`. This is a scoped follow-up to #565: the same-module and explicit-import cases work; the unambiguous glob case does not. The FFI layout guard prevents an invalid call, but its suggestion to register a by-value layout is inappropriate for a borrowed generated owner.

Environment executed: Julia 1.13.3, RustCall.jl 0.9.2 (latest release v0.9.2), rustcall_julia_macros 0.6.0, rustc 1.99.1 (58cf8f9c3 2026-10-13), Linux x86_64. Initial direct cdylib build and a cached repeat both reproduce, no faer dependency, 2026-11-05. Current upstream main `47c3ed21dbae0fca78a735198e99cb82d4c29189` was inspected, not executed; its `deps/rustcall_julia_core/src/paths.rs` is byte-identical to the installed file. Searches for glob/borrowed/path/module reports found closed #565 but no report covering this scoped import case.

Create a directory containing these complete files:

`Cargo.toml`:
```toml
[package]
name = "rustcall-path-mwe"
version = "0.1.0"
edition = "2021"
[lib]
crate-type = ["cdylib", "rlib"]
[dependencies]
rustcall_julia_macros = "=0.6.0"
```

`src/lib.rs`:
```rust
use rustcall_julia_macros::julia;
#[julia]
pub struct Counter { value: i32 }
#[julia]
impl Counter {
    #[julia]
    pub fn new() -> Self { Self { value: 42 } }
}
#[julia]
pub struct Destination { value: i32 }
#[julia]
impl Destination {
    #[julia]
    pub fn new() -> Self { Self { value: 0 } }
}
mod operations;
#[test]
fn rust_answer() { assert_eq!(Destination::new().read(&Counter::new()), 42); }
```

`src/operations.rs`:
```rust
use super::*;
#[julia]
impl Destination {
    #[julia]
    pub fn read(&mut self, source: &Counter) -> i32 { self.value = source.value; self.value }
}
```

`repro.jl`:
```julia
using RustCall
println("Julia: ", VERSION)
println("RustCall: ", pkgversion(RustCall))
info = RustCall.scan_crate(@__DIR__)
println("Methods: ", [(s.name,[m.name for m in s.methods]) for s in info.julia_structs])
bindings = RustCall.load_crate_bindings(@__DIR__)
@assert bindings.read(bindings.Destination(),bindings.Counter()) == 42
```

Run from that directory (Julia environment may live elsewhere):

```sh
julia --project=env -e 'using Pkg; Pkg.add(PackageSpec(name="RustCall", version="0.9.2"))'
cargo test
julia --project=env repro.jl
```

Executed results: Rust unit test `rust_answer` passes (exit 0). Julia scans `Counter: new` and `Destination: new, read`, builds the library, then exits 1:

```text
RustError: cannot pass ...RustcallPathMwe.Counter to Rust by value as argument 2:
RustCall has no layout assertion for it (#245).
ffi_check_by_value_signature(::Type{Int32},
    ::Type{Tuple{Ptr{Nothing}, ...RustcallPathMwe.Counter}})
```

Expected: `read(Destination(), Counter()) == Int32(42)` using checked immutable borrow of Counter and checked mutable Destination borrow. Replacing only `use super::*;` with the following makes the exact Julia assertion pass (exit 0):

```rust
use super::{Counter, Destination};
use rustcall_julia_macros::julia;
```

### Cause and proposed correction

`paths::locate_borrowed_type` uses `locate_with_fallback(..., Fallback::None)`, which deliberately stops before glob imports. Impl ownership can still be located, so the method exists, while its borrowed argument lacks handle lowering. The generator subsequently routes the wrapper value to the by-value FFI guard. This is a safe rejection, not evidence of memory corruption.

Proposed (not implemented/tested upstream): add conservative name resolution through glob-imported module scopes. Resolve each applicable glob's target scope for the exact argument identifier, retain local-item and explicit-import precedence, include plain/unannotated types and aliases in shadowing/ambiguity checks, and accept only a unique proven annotated owner. Do not enable the existing global unique-name fallback or guess across the selected structs: those are not sound Rust name resolution. For unresolved/ambiguous cases, emit an explicit unsupported borrowed-owner diagnostic instead of suggesting by-value registration. Keep existing lifetime/library-generation/alive/alias checks unchanged.

Regression: the provided cross-module glob case, its explicit-import control, `&mut Counter`, named import precedence, a plain local Counter shadowing the annotated one, conflicting glob candidates, and reexport/glob cycles. Ensure both expression and written binding emitters consume the same proven handle metadata. Colmat uses explicit imports as the local workaround; no raw pointers or by-value registration are needed.
