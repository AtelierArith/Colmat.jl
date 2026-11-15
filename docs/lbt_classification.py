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
        return dict(status="unverified", api="未照合", source="faer/src/linalg/mod.rs",
                    note="規則未登録。公開 API の意味・型・保存形式・オプションとの対応は未照合。名称や suffix/prefix は不存在の根拠ではない。",
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
        result.update(status="unverified", api="未照合",
                      note="旧規則では非対応と仮分類したが、revision・型・保存形式・オプションと確認方法の記録が不足している。再照合対象。旧メモ: " + result["note"])
    result["evidence_status"] = "scoped_review" if evidence else "not_catalogued"
    if result["status"] == "unverified":
        result["evidence_status"] = "not_reviewed"
    result["evidence"] = evidence
    result["algorithm_status"] = result.get("algorithm_status", "not_assessed")
    return result
