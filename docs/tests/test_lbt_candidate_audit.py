import csv
import hashlib
import json
import sys
import unittest
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DOCS))
from audit_lbt_candidates import reconcile


class CandidateAuditTests(unittest.TestCase):
    def fixture(self):
        symbols = [("dsytrf_", "Fortran", "sytrf", False),
                   ("LAPACKE_dsytrf_work", "LAPACKE", "sytrf", False),
                   ("dsytrf_rook_", "Fortran", "sytrf_rook", True)]
        old = [dict(symbol=s, interface=i, family=f, colmat_api="No independent public API" if rook else "Public feature available",
                    adapter_lp64="false", adapter_ilp64="false") for s, i, f, rook in symbols]
        current = [{**row, "capability_status": "representation"} for row in old]
        issues = {n: {"body": ""} for n in range(264, 315)}
        issues[266]["body"] = "dsytrf_"
        issues[271]["body"] = "CBLAS shared ABI"
        issues[288]["body"] = "LAPACKE shared ABI"
        return old, current, issues

    def test_shared_abi_is_distinct_from_numerical_ownership(self):
        old, current, issues = self.fixture()
        coverage, summary = reconcile(old, current, issues, 372)
        lookup = {row["symbol"]: row for row in coverage}
        self.assertEqual(summary["original_candidates"], 2)
        self.assertEqual(summary["additions"], ["dsytrf_rook_"])
        self.assertEqual(lookup["LAPACKE_dsytrf_work"]["owner_issues"], "288")
        self.assertEqual(lookup["LAPACKE_dsytrf_work"]["numerical_issues"], "266")
        self.assertEqual(lookup["dsytrf_rook_"]["owner_issues"], "372")

    def test_already_connected_numerics_are_attributed_to_baseline(self):
        old, current, issues = self.fixture()
        old.append(dict(symbol="dgetrf_", interface="Fortran", family="getrf", colmat_api="Public feature available", adapter_lp64="true", adapter_ilp64="true"))
        old.append(dict(symbol="LAPACKE_dgetrf_work", interface="LAPACKE", family="getrf", colmat_api="Public feature available", adapter_lp64="false", adapter_ilp64="false"))
        current += [{**row, "capability_status": "representation"} for row in old[-2:]]
        coverage, summary = reconcile(old, current, issues, 372)
        entry=next(row for row in coverage if row["symbol"]=="LAPACKE_dgetrf_work")
        self.assertEqual(entry["owner_issues"],"288")
        self.assertEqual(entry["numerical_issues"],"262")
        self.assertEqual(summary["baseline_numerical_lapacke"],1)

    def test_unowned_fortran_and_duplicate_canonical_are_rejected(self):
        old, current, issues = self.fixture()
        issues[266]["body"] = ""
        with self.assertRaises(ValueError):
            reconcile(old, current, issues, 372)
        old, current, issues = self.fixture()
        with self.assertRaises(ValueError):
            reconcile(old + old[:1], current, issues, 372)

    def test_recorded_full_cohort_and_hashes(self):
        assets = DOCS / "src/assets"
        report = json.loads((assets / "lbt-candidate-audit.json").read_text())
        snapshot_data=(DOCS / "audits/issue-263-scope-snapshot.json").read_bytes()
        snapshot=json.loads(snapshot_data)
        self.assertEqual(hashlib.sha256(snapshot_data).hexdigest(),report["issue_snapshot_sha256"])
        for item in report["issue_snapshots"]:
            body=snapshot["issues"][str(item["number"])]["body"]
            self.assertEqual(hashlib.sha256(body.encode()).hexdigest(),item["body_sha256"])
        coverage_data = (assets / "lbt-candidate-coverage.tsv").read_bytes()
        coverage = list(csv.DictReader(coverage_data.decode().splitlines(), delimiter="\t"))
        self.assertEqual(hashlib.sha256(coverage_data).hexdigest(), report["coverage_tsv_sha256"])
        self.assertEqual(hashlib.sha256((DOCS / "lbt-capabilities.json").read_bytes()).hexdigest(), report["policy_sha256"])
        self.assertEqual(len(coverage), 5146)
        self.assertEqual(len({row["symbol"] for row in coverage}), 5146)
        historical = [row for row in coverage if row["historical_candidate"] == "true"]
        self.assertEqual(len(historical), 2910)
        self.assertTrue(all(row["owner_issues"] for row in historical))
        self.assertTrue(all(row["numerical_issues"] for row in historical if row["interface"] == "LAPACKE"))
        self.assertEqual(report["summary"]["baseline_numerical_lapacke"],148)
        self.assertEqual(report["summary"]["removed"], [])
        self.assertEqual(len(report["summary"]["additions"]), 40)
        self.assertEqual(report["summary"]["additions_by_interface"], {"Fortran": 20, "LAPACKE": 20})
        self.assertEqual({i["number"] for i in report["issue_snapshots"]}, set(range(264, 315)))


if __name__ == "__main__":
    unittest.main()
