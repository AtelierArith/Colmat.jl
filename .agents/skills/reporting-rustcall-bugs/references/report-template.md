# RustCall issue body

Adapt this outline to the actual defect. Fill it using observed evidence, not
hypothetical output. No fixed test count or dependency version is prescribed.

## Problem and expected behavior

Name the valid Rust method or Julia call, the intended result, and the observed
wrong result/error/crash. Explain why the responsibility lies in RustCall rather
than the wrapper or Rust library. Include a standalone Rust comparison when it
helps establish this distinction.

## Environment

- Julia version/platform and Rust compiler version.
- RustCall.jl release/resolved commit; macro crate version.
- Complete relevant Cargo features/dependency resolution.
- Tested revision and date; latest-version/source check and duplicate-issue search.
- Fresh build versus reused cached library, if relevant to reproduction.

## Complete executable MWE

Include minimal `Project.toml`/Julia package setup, full Julia script, and full
Rust `Cargo.toml`/source if not embedded in the Julia script. Give exact creation,
instantiation, build and run commands. A maintainer must not need the original
application. Pin or identify the reproduced versions while avoiding a universal
recommendation to pin every patch version in a production package.

## Observed result

Include the relevant actual output, complete short stack trace or selected
frames with the full trace available as an attachment/artifact, and process exit
status. Include expected values independently, with assertions when practical.

## Cause and concrete proposed solution

Identify the source method/generator/macro/conversion and failure mechanism.
Describe an implementation change or include a small proposed diff. Distinguish
source-based hypotheses from experimentally verified causes and tested fixes.
Describe how the change preserves correct ownership and error behavior.

## Regression and workaround

Specify a regression based on observable behavior: supported return type,
error propagation, independent owned result, checked reference, wrong-owner
rejection, use-after-close, mutable alias rejection, or cache invalidation as
relevant. Include the minimal successful and failing cases needed for the defect.
State whether a wrapper workaround exists and its limitations.

Past Colmat reports #564 and #565 were resolved in RustCall 0.9.2. Treat that as
historical context; verify current issue status and versions rather than filing
those same reports again.
