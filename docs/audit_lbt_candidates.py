"""Reconcile the immutable #262 cohort with recorded issue scopes and current policy.

Inputs are explicit local snapshots; this command never writes to GitHub.
"""
import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

BASELINE_REVISION = "3c850a88a05b3c09eadd84d3722b1704cf1421f1"
BASELINE_NATIVE = "e1c83482f5a61449d30e989808c3ba5f751ec391"

# The pinned #262 baseline predates the English labels; accept both spellings.
NEGATIVE_API_LABELS = {"No independent public API", "No independent public API (within reviewed scope)",
                       "対応する独立公開 API なし"}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def reconcile(baseline, current, issues, rook_issue):
    old = {row["symbol"]: row for row in baseline}
    now = {row["symbol"]: row for row in current}
    if len(old) != len(baseline) or len(now) != len(current) or old.keys() != now.keys():
        raise ValueError("Canonical coverage changed or contains duplicates")
    if set(issues) != set(range(264, 315)):
        raise ValueError("The original 51 issue snapshots are required")
    if "LAPACKE" not in issues[288]["body"] or "CBLAS" not in issues[271]["body"]:
        raise ValueError("Shared C API scope needs reassessment")
    mentions = {n: set(re.findall(r"[A-Za-z][A-Za-z0-9_]*", item["body"])) for n, item in issues.items()}
    original_positive = {s for s, row in old.items() if row["colmat_api"] not in NEGATIVE_API_LABELS}
    original_candidates = {s for s in original_positive if not (
        old[s]["adapter_lp64"] == "true" and old[s]["adapter_ilp64"] == "true")}
    positive = {s for s, row in now.items() if row["capability_status"] not in ("unsupported", "unverified")}
    additions = positive - original_positive
    removed = original_positive - positive
    coverage = []
    for symbol, row in sorted(now.items()):
        interface = row["interface"]
        historical = symbol in original_candidates
        added = symbol in additions
        owners = {n for n, tokens in mentions.items() if symbol in tokens} if historical else set()
        numeric = set(owners)
        basis = "explicit canonical symbol" if owners else "outside original candidate cohort"
        if historical or added:
            if interface == "LAPACKE":
                owners.add(288)
                name = symbol.removeprefix("LAPACKE_").removesuffix("_work") + "_"
                numeric = {n for n, tokens in mentions.items() if name in tokens}
                basis = "LAPACKE shared ABI #288; numerical owners from Fortran counterpart"
                if not numeric and name in old and old[name]["adapter_lp64"] == old[name]["adapter_ilp64"] == "true":
                    numeric.add(262)
                    basis = "LAPACKE shared ABI #288; numerical counterpart already registered at #262 (not full-contract verification)"
                if historical and not numeric:
                    raise ValueError(f"Unowned numerical counterpart: {symbol}")
            elif interface == "CBLAS":
                owners.add(271)
                basis = "CBLAS shared ABI #271"
            if added:
                if not row["family"].endswith("_rook"):
                    raise ValueError(f"New positive capability needs an issue: {symbol}")
                owners.add(rook_issue)
                numeric.add(rook_issue)
                basis = "rook API omission corrected; dedicated follow-up and shared C ABI scope"
            if not owners:
                raise ValueError(f"Unowned candidate: {symbol}")
        coverage.append(dict(symbol=symbol, interface=interface,
                             historical_candidate=str(historical).lower(),
                             reassessment_addition=str(added).lower(),
                             capability_status=row["capability_status"],
                             owner_issues=" ".join(map(str, sorted(owners))),
                             numerical_issues=" ".join(map(str, sorted(numeric))), coverage_basis=basis))
    return coverage, dict(original_candidates=len(original_candidates),
                          original_positive=len(original_positive),
                          reassessed_positive=len(positive),
                          additions=sorted(additions), removed=sorted(removed),
                          unverified_original_exclusions=sum(s not in original_positive and r["capability_status"] == "unverified" for s, r in now.items()),
                          additions_by_interface=dict(Counter(now[s]["interface"] for s in additions)),
                          baseline_numerical_lapacke=sum(row["historical_candidate"] == "true" and row["interface"] == "LAPACKE" and row["numerical_issues"] == "262" for row in coverage))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-tsv", type=Path, required=True)
    parser.add_argument("--issue-snapshot", type=Path, required=True)
    parser.add_argument("--rook-issue", type=int, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    assets = root / "docs/src/assets"
    baseline_data = args.baseline_tsv.read_bytes()
    baseline = list(csv.DictReader(baseline_data.decode().splitlines(), delimiter="\t"))
    current = list(csv.DictReader((assets / "lbt-symbol-inventory.tsv").open(), delimiter="\t"))
    snapshot = json.loads(args.issue_snapshot.read_text())
    issues = {int(n): value for n, value in snapshot.get("issues", snapshot).items()}
    coverage, summary = reconcile(baseline, current, issues, args.rook_issue)
    if len(current) != 5146 or summary["original_candidates"] != 2910:
        raise ValueError("The recorded #262 baseline does not match its original scope")
    output = assets / "lbt-candidate-coverage.tsv"
    with output.open("w") as file:
        writer = csv.DictWriter(file, fieldnames=list(coverage[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(coverage)
    policy = json.loads((root / "docs/lbt-capabilities.json").read_text())
    report = dict(baseline_wrapper_revision=BASELINE_REVISION,
                  baseline_native_revision=BASELINE_NATIVE,
                  baseline_tsv_sha256=digest(baseline_data),
                  reassessment_native_revision=policy["assessment_revision"],
                  policy_sha256=digest((root / "docs/lbt-capabilities.json").read_bytes()),
                  generated_at=datetime.now(timezone.utc).isoformat(),
                  issue_snapshot_saved_at=snapshot.get("saved_at", "not_recorded"),
                  issue_snapshot_sha256=digest(args.issue_snapshot.read_bytes()),
                  canonical_symbols=len(current), original_implementation_issue_count=51,
                  rook_followup_issue=args.rook_issue, summary=summary,
                  coverage_tsv_sha256=digest(output.read_bytes()),
                  issue_snapshots=[dict(number=n, url=item["html_url"], updated_at=item["updated_at"],
                                        body_sha256=digest(item["body"].encode())) for n, item in sorted(issues.items())])
    (assets / "lbt-candidate-audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
