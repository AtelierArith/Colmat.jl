"""Pure, conservative capability classification; forwarding is recorded elsewhere."""

REQUIRED_EVIDENCE = {"revision", "scalars", "storage", "options", "method", "sources", "reason"}


def classify(family, scalar, groups, revision, overrides=()):
    """Missing rules and unsubstantiated absence claims remain unverified.

    An absence assessment is a bounded semantic review of a public contract,
    never a conclusion drawn from an unregistered name or a matching suffix.
    """
    group = groups.get(family)
    for override in overrides:
        if family in override["routines"] and scalar in override["scalars"]:
            group = override
            break
    if group is None:
        return dict(status="unverified", api="unverified", source="faer/src/linalg/mod.rs",
                    note="Rules not registered. The correspondence with the public API meaning, types, storage format, and options is unverified. Names and suffix/prefix are not evidence of absence.",
                    evidence_status="not_reviewed", evidence=None,
                    algorithm_status="not_assessed")
    result = dict(group)
    evidence = result.get("evidence")
    if evidence is not None:
        missing = REQUIRED_EVIDENCE - evidence.keys()
        if missing or any(not evidence[key] for key in REQUIRED_EVIDENCE):
            raise ValueError(f"Incomplete capability evidence for {family}: {missing}")
        if evidence["revision"] != revision:
            raise ValueError(f"Stale capability evidence for {family}")
        if scalar not in evidence["scalars"]:
            raise ValueError(f"Capability evidence does not cover scalar {scalar}")
    elif result["status"] == "unsupported":
        result.update(status="unverified", api="unverified",
                      note="The old rules tentatively classified this as unsupported, but the record of the revision, types, storage formats, options, and verification method is insufficient. Re-review target. Old note: " + result["note"])
    result["evidence_status"] = "scoped_review" if evidence else "not_catalogued"
    if result["status"] == "unverified":
        result["evidence_status"] = "not_reviewed"
    result["evidence"] = evidence
    result["algorithm_status"] = result.get("algorithm_status", "not_assessed")
    return result
