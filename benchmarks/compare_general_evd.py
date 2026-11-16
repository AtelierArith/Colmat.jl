"""Compare the same 768 general EVD conditions in two preserved reports."""
import gzip
import json
import statistics
import sys
import tomllib


def read(path):
    with (gzip.open(path, "rb") if path.endswith(".gz") else open(path, "rb")) as stream:
        return tomllib.load(stream)


before, after = (read(path) for path in sys.argv[1:3])
for field in (
    "native_revision", "tool_manifest_sha256", "julia_version", "rustc",
    "cpu_model", "platform", "validation_blas",
):
    assert before[field] == after[field], field
keys = ("scalar", "n", "left", "right", "balance", "participants", "fixture")


def key(row):
    return tuple(row[field] for field in keys)


b = {key(row): row for row in before["cases"]}
a = {key(row): row for row in after["cases"]}
assert len(before["cases"]) == len(after["cases"]) == len(a) == len(b) == 768
assert a.keys() == b.keys()
for case in a:
    assert a[case]["input_seed"] == b[case]["input_seed"]
for report in (a, b):
    assert all(
        row["info"] == 0 and row["samples"] == 10
        and row["julia_allocations"] == row["julia_memory_bytes"] == 0
        for row in report.values()
    )
result = {
    "before_revision": before["source_revision"],
    "after_revision": after["source_revision"],
    "cases": len(a),
    "groups": [],
}
for n in (32, 96, 256):
    for mode in ("NN", "VN", "NV", "VV"):
        cases = [k for k in a if a[k]["n"] == n and a[k]["left"] + a[k]["right"] == mode]
        row = {
            "n": n, "mode": mode,
            "median_before_over_after": statistics.median(
                b[k]["steady_median_ns"] / a[k]["steady_median_ns"] for k in cases
            ),
        }
        if "native_profile" in next(iter(a.values())):
            for metric in (
                "median_allocations", "median_allocated_bytes", "median_extra_peak_live_bytes",
            ):
                for label, report in (("before", b), ("after", a)):
                    row[metric + "_" + label] = statistics.median(
                        report[k]["native_profile"][metric] for k in cases
                    )
        result["groups"].append(row)
result["max_residuals"] = {
    scalar: max(
        row.get(side + "_residual", 0) for row in a.values()
        if row["scalar"] == scalar for side in ("left", "right")
    ) for scalar in ("Float32", "Float64", "ComplexF32", "ComplexF64")
}
print(json.dumps(result, indent=2))
