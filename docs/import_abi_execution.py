"""Import a successful full-suite ABI artifact after validating its provenance.

Run with Python 3.11+: python docs/import_abi_execution.py test-output/abi-execution.toml
Then regenerate the ordinary inventory and candidate ownership audit.
"""
from pathlib import Path
import csv
import json
import sys
import tomllib
from lbt_execution import SNAPSHOT, validate_execution, lapack_rows, cblas_rows, lapacke_rows

root = Path(__file__).resolve().parent.parent
rows = list(csv.DictReader((root / 'docs/src/assets/lbt-symbol-inventory.tsv').open(), delimiter='\t'))
revision = tomllib.loads((root / 'deps/upstream.toml').read_text())['revision']
data = tomllib.loads(Path(sys.argv[1]).read_text())
executed = validate_execution(root, rows, revision, data)
(root / SNAPSHOT).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
print(f'Validated {len(lapack_rows(rows))} LAPACK and {len(cblas_rows(rows))} CBLAS and {len(lapacke_rows(rows))} LAPACKE canonical entries; source revision {data["revision"]}')
