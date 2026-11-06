---
name: reporting-rustcall-bugs
description: Investigate RustCall.jl binding defects or demonstrated API gaps, execute minimal Julia/Rust reproductions, and file actionable issues in AtelierArith/RustCall.jl. Use for failures found during binding work involving generated code, ownership, conversions, macros or diagnostics; distinguish wrapper and Rust dependency bugs first.
---

# Report RustCall.jl defects

Turn an observed RustCall failure or reproducible API limitation into a report
that a maintainer can run and act on. Target
<https://github.com/AtelierArith/RustCall.jl/issues>.

During an authorized task, report relevant confirmed defects discovered along
the way; do not wait for a separate request for each report. This does not expand
the task into an unrelated audit. Read applicable repository AGENTS.md and
reporting conventions before preparing the report.

## Establish responsibility

Inspect the actual Julia RustCall package, generated binding signatures, macro
crate version and Rust adapter source. Capture Julia/Rust/toolchain/platform,
resolved package versions or commit SHAs, Cargo features, build invocation and
whether an existing library/cache was reused.

Reduce the failing behavior independently:

- A Rust-library operation that fails in standalone Rust usually belongs in that
  library's issue tracker, not RustCall's.
- A wrapper supplying a wrong dtype, stale shape, invalid pointer or expired
  owner should be corrected in the wrapper.
- A valid Rust method that RustCall binds incorrectly, an erroneous checked
  owner/reference conversion, a result/return conversion defect, or inadequate
  diagnostics for a supported usage is a RustCall candidate.

Compare generated signatures and documented supported usage with the installed
source. Reproduce with a tiny custom Rust crate or inline Rust function rather
than requiring Colmat or a large numerical library. Preserve valid lifetimes;
never invoke undefined behavior to demonstrate an ownership problem. Identify
limitations as limitations and requests as requests when correctness is not
violated.

## Produce the evidence

Read [references/report-template.md](references/report-template.md) for the
report structure. Execute the exact commands and code included in the report.
Include complete Julia environment setup and Rust sources/Cargo manifest when
needed, imports, expected behavior, actual output/error, and exit status. Remove
machine-specific paths, secrets, and dependencies on unprovided local files.
Use separate processes for crashes and verify any edited reproducer again.

Check the latest available RustCall release/source and search existing issues.
State exactly which revision was executed versus merely inspected. If fixed in
a compatible release, prefer migrating and verifying the fix rather than opening
a duplicate. Do not claim a defect remains in latest without evidence.

Propose a concrete correction at the responsible binding generator, macro,
conversion, or owner-validation code. Explain the intended behavior, relevant
source location, why the change addresses the MWE, and a regression test. Label
untested patches as proposals. A workaround alone is not a proposed upstream fix.
Avoid bypassing checked borrows to make a failing ownership test appear to pass.

## Publish and retain

When the active task/session authorizes reporting confirmed RustCall bugs,
create the issue without asking for duplicate confirmation. This skill's presence
alone does not authorize external writes in unrelated tasks. Prepare a complete
reviewable draft even when posting is not yet authorized.

Keep API requests distinct from correctness bugs. Permission to report in
AtelierArith/RustCall.jl does not authorize merging PRs, publishing crates or
contacting other projects. For a standalone faer failure, use
`reporting-faer-bugs` when available rather than filing it against RustCall.

Prefer a sufficient connected GitHub tool. An authenticated `gh` fallback is
appropriate when connector capabilities or repository access are insufficient.
Use a structured body or a temporary file with `--body-file`, preserving actual
newlines and literal code. For example:

```sh
gh issue create --repo AtelierArith/RustCall.jl \
  --title 'Concrete binding operation and observed defect' \
  --body-file /tmp/rustcall-issue.md
```

Confirm the returned issue URL, read back the report, and retain the executed MWE plus link in the
calling project's reproducer notes when that fits the task. Clearly report any
posting failure and keep the draft; do not claim publication without a confirmed
result. Do not close an upstream issue merely because the wrapper has a workaround.

If creation times out or returns an ambiguous transport error, check the target
repository for the submitted title/body before retrying. Reuse an existing report
for the same root cause instead of creating duplicates. Finish with the confirmed
issue URL, tested versions, proposed correction, and any unverified parts.
