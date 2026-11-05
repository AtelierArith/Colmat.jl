# Householder upgrade per-call allocation

Standalone four-scalar MWE for the compact-WY allocation recorded in
[tensor4all/faer#93](https://github.com/tensor4all/faer/issues/93).

Run `cargo run --release --locked --manifest-path docs/reproducers/householder-upgrade-allocation/Cargo.toml` from the Colmat root.
The dependency pins faer main `cd516af670ee23cd28aa7a1499592936ec5d7488`;
Rust 1.99.1 / Linux x86_64 / threadteam 0.3.0 were executed. After three warm-up
calls, every counted width-2 call allocates one partial buffer: 8192/16384/16384/
32768 bytes for f32/f64/c32/c64. Width 1 allocates zero. Results pass independent
strict-upper Gram checks; no timing/scaling claim is made. The counter includes
all threads and excludes setup and validation.

The retained Cargo.lock captures the executed resolution. Kernel source is
identical to Colmat's selected clean faer `19053d8` (0.29.0). The public upgrade
API has no scratch argument or query. Caller-owned workspace for this allocation
remains an upstream requirement; this MWE does not implement a fix.

Executed evidence and proposed correction: https://github.com/tensor4all/faer/issues/93#issuecomment-6648218975
