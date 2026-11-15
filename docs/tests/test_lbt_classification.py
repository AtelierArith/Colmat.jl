"""Regression tests use synthetic assessments; they do not assert API absence."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lbt_classification import classify


class ClassificationTests(unittest.TestCase):
    revision = "a" * 40

    def evidence(self):
        return dict(revision=self.revision, scalars=["c", "z"], storage="dense symmetric",
                    options="transpose rather than adjoint factorization", method="semantic public contract review",
                    sources=["factor.rs", "solve.rs"], reason="reviewed API requires adjoint symmetry")

    def test_unregistered_names_are_not_absence_proofs(self):
        for family in ("brand_new", "new_rook", "new_2stage", "la_unknown", "tp_unknown", "nancheck"):
            self.assertEqual(classify(family, "d", {}, self.revision)["status"], "unverified")

    def test_old_absence_rule_without_evidence_remains_unverified(self):
        groups = {"old": dict(status="unsupported", api="—", source="mod.rs", note="old claim")}
        self.assertEqual(classify("old", "d", groups, self.revision)["status"], "unverified")

    def test_partial_and_precision_specific_assessments(self):
        groups = {"factor": dict(status="representation", api="factor", source="factor.rs", note="representation"),
                  "expert": dict(status="conditional", api="basic", source="expert.rs", note="basic only")}
        overrides = [dict(routines=["factor"], scalars=["c", "z"], status="unsupported", api="—",
                          source="factor.rs", note="different symmetry", evidence=self.evidence())]
        self.assertEqual(classify("factor", "d", groups, self.revision, overrides)["status"], "representation")
        self.assertEqual(classify("factor", "c", groups, self.revision, overrides)["status"], "unsupported")
        self.assertEqual(classify("expert", "d", groups, self.revision, overrides)["status"], "conditional")
        self.assertEqual(classify("factor", "c", groups, self.revision, overrides)["algorithm_status"], "not_assessed")

    def test_single_and_double_precision_can_have_distinct_option_support(self):
        groups={"factor":dict(status="native",api="synthetic factor",source="factor.rs",note="synthetic double-precision contract")}
        overrides=[dict(routines=["factor"],scalars=["s"],status="conditional",api="synthetic restricted factor",source="factor.rs",note="synthetic single-precision option restriction")]
        self.assertEqual(classify("factor","s",groups,self.revision,overrides)["status"],"conditional")
        self.assertEqual(classify("factor","d",groups,self.revision,overrides)["status"],"native")

    def test_current_policy_preserves_precision_and_rook_distinctions(self):
        import json
        policy = json.loads((Path(__file__).resolve().parents[1] / "lbt-capabilities.json").read_text())
        import re
        upstream=(Path(__file__).resolve().parents[2] / "deps/upstream.toml").read_text()
        pin=re.search(r'^revision\s*=\s*"([a-f0-9]{40})"',upstream,re.MULTILINE)
        self.assertIsNotNone(pin)
        self.assertEqual(policy["assessment_revision"],pin.group(1))
        groups = {name: group for group in policy["groups"] for name in group["routines"]}
        def current(family, scalar):
            return classify(family, scalar, groups, policy["assessment_revision"], policy["scalar_overrides"])
        self.assertEqual(current("sytrf", "d")["status"], "representation")
        self.assertEqual(current("sytrf", "c")["status"], "unsupported")
        self.assertEqual(current("sytrf_rook", "d")["status"], "representation")
        self.assertEqual(current("hetrf_rook", "z")["status"], "representation")
        self.assertEqual(current("sytrf_rook", "z")["status"], "unsupported")
        self.assertEqual(current("geevx", "d")["status"], "conditional")
        self.assertEqual(current("future_routine", "d")["status"], "unverified")

    def test_absence_evidence_is_revision_and_scope_bound(self):
        evidence = self.evidence()
        group = dict(status="unsupported", api="—", source="factor.rs", note="review", evidence=evidence)
        for mutate in (lambda e: e.update(revision="b"*40), lambda e: e.pop("method"), lambda e: e.update(scalars=["z"])):
            changed = {**evidence}
            mutate(changed)
            with self.assertRaises(ValueError):
                classify("factor", "c", {"factor": {**group, "evidence": changed}}, self.revision)


if __name__ == "__main__":
    unittest.main()
