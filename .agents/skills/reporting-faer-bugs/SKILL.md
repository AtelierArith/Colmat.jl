---
name: reporting-faer-bugs
description: Investigate defects or missing public capabilities in tensor4all/faer, execute standalone Rust reproductions, and file actionable issues. Use for faer failures found during numerical or binding work; distinguish wrapper and RustCall defects first.
---

# Report tensor4all/faer defects

Turn a confirmed faer defect or demonstrated public API gap into an actionable
report at <https://github.com/tensor4all/faer/issues>. During an authorized task,
report relevant defects discovered along the way; do not wait for a separate
request for each report. This does not authorize an unrelated audit or external
writes in another session without task/session authorization.

## Establish responsibility and repository context

Inspect the actual resolved faer source revision and Cargo features, native
adapter, build script and calling API. Record the executed checkout, not just a
manifest version: a wrapper build may fetch a newer default-branch revision.
Read applicable AGENTS.md and review rules in the faer checkout and the shared
rules they reference before preparing a report.

- Reproduce a suspected numerical or Rust API defect in standalone Rust without
  Julia/RustCall. A panic alone does not establish faer responsibility: verify
  documented preconditions, shapes, initialized storage, strides and lifetimes.
- Correct invalid wrapper inputs or stale ownership/metadata in the wrapper.
  If valid Rust works but generated bindings fail, investigate RustCall instead.
- Label a missing capability as an API request rather than a correctness bug.
  Explain the supported operation it prevents and propose a concrete API.
- The reporting destination is the tensor4all fork. Its upstream may be
  Codeberg; permission to report here does not authorize contacting Codeberg or
  other trackers. Follow the checkout's upstream-feedback rules.

## Execute a safe minimal reproduction

Use a small independent crate with complete Cargo.toml, source, pinned dependency
and exact commands. Avoid unpublished local patches, machine-specific paths and
unprovided fixtures. Retain Cargo.lock when it matters for reproduction.

Never manufacture undefined behavior: do not read uninitialized memory, violate
unsafe preconditions or use expired references. Isolate panics/crashes/hangs in
child processes; bound suspected hangs and record timeout versus actual exit.

Assert an independent expected result or invariant. For numerical defects use
residual/reconstruction/orthogonality and precision/scale-aware checks; finite
values or correct shapes alone do not prove correctness. Include the dtype,
complex phases, layout, dimension or Seq/Team configuration that triggers the
failure, and a nearby positive control when useful. Distinguish output-owner
allocations from temporary workspace; allocation counts do not prove a timing
regression. A missing public method can be demonstrated by a safe compile-only
probe with the actual compiler diagnostics.

Search existing issues and inspect current fork main. State separately which
revisions were executed and which were only inspected. If already fixed, adopt
and verify the fix when in scope rather than file a duplicate. Re-run edited
reproducer code before presenting it as executed evidence.

## Prepare and publish the report

Write the issue in English, following the fork's review conventions. Include:

- Concrete trigger and expected versus actual behavior, error/output and status.
- Exact faer commit, toolchain/platform/features and execution commands.
- Complete runnable MWE and any input generator/environment setup it needs.
- Relevant source location and a concrete proposed correction or API, with a
  regression-test suggestion. Label untested fixes as proposals.
- Wrapper impact/workaround and its limits, if applicable; link related reports.

When the active task/session authorizes faer issue reporting, prepare and post
the report without duplicate confirmation. The skill itself grants no permission
to merge, publish crates or contact other projects. Prefer sufficient connected
GitHub tools; use authenticated gh when repository access/capabilities are
insufficient. Use a structured body or a UTF-8 file with --body-file, preserving
literal code and newlines.

Confirm the created issue URL and read back the report. On an ambiguous creation
failure, search for the submitted title/body before retrying. Reuse or extend an
existing report for the same root cause. Keep a complete draft if posting fails
and report the actual failure rather than claiming publication.

Retain the executed crate and issue link in the caller project's reproducer
notes when appropriate. A wrapper workaround does not resolve the faer defect;
remove it only after selecting the fixed revision and verifying the original
MWE and wrapper regression. Report tested versions, the confirmed URL and any
unverified parts to the user.
