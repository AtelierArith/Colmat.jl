"""Validate archived EVD reports and quantify asynchronous capture boundaries."""
import gzip
import hashlib
import json
import pathlib
import sys
import tomllib

root=pathlib.Path(sys.argv[1])
manifest=json.loads((root/'manifest.json').read_text())
result={}
for filename, hashes in manifest.items():
    data=(root/filename).read_bytes()
    raw=gzip.decompress(data)
    assert hashlib.sha256(data).hexdigest()==hashes['compressed_sha256']
    assert hashlib.sha256(raw).hexdigest()==hashes['uncompressed_sha256']
    report=tomllib.loads(raw.decode())
    if not filename.startswith('profile'):
        continue
    count=deltas=mismatches=max_boundary=max_live=0
    widths={}
    for row in report['cases']:
        captures=row['native_profile']['captures']
        assert len(captures)==5
        for capture in captures:
            assert len(capture['phases'])==15
            for field in ('allocations','allocated_bytes','frees','freed_bytes'):
                assert sum(p[field] for p in capture['phases'])==capture[field]
            delta=capture['ending_live_bytes']-capture['baseline_live_bytes']
            accounted=capture['allocated_bytes']-capture['freed_bytes']
            deltas+=bool(delta)
            mismatches+=delta!=accounted
            max_boundary=max(max_boundary,abs(delta-accounted))
            max_live=max(max_live,abs(delta))
            width=widths.setdefault(str(row['participants']),{'captures':0,'boundary_mismatches':0,'nonzero_live_deltas':0})
            width['captures']+=1
            width['boundary_mismatches']+=delta!=accounted
            width['nonzero_live_deltas']+=bool(delta)
            count+=1
    assert count==3840
    result[filename]={'captures':count,'hashes_and_phase_totals':'passed','nonzero_live_deltas':deltas,'max_absolute_live_delta_bytes':max_live,'live_vs_request_boundary_mismatches':mismatches,'max_absolute_boundary_difference_bytes':max_boundary,'participants':widths}
print(json.dumps(result,indent=2))
