"""Validate completed-call evidence independently of capability classification."""
from pathlib import Path
import hashlib
import json
import re

BLAS_FAMILIES = set('scal real_scal copy swap axpy axpby dot dotu dotc asum nrm2 iamax iamin amax amin max min sum rot real_rot rotg gemv gbmv trmv tbmv tpmv trsv tbsv tpsv symv sbmv spmv hemv hbmv hpmv ger gerc geru syr spr syr2 spr2 her hpr her2 hpr2 gemm gemm3m gemmt gemmtr symm hemm syrk syr2k herk her2k trmm trsm imatcopy omatcopy geadd'.split())
REGISTRATION_FIELDS = ('symbol', 'interface', 'family', 'scalar', 'adapter_lp64', 'adapter_ilp64')
SNAPSHOT = 'docs/src/assets/lapack-abi-execution.json'


def registration_digest(rows):
    text = '\n'.join('\t'.join(row[field] for field in REGISTRATION_FIELDS)
                     for row in sorted(rows, key=lambda row: row['symbol'])) + '\n'
    return hashlib.sha256(text.encode()).hexdigest()


def lapack_rows(rows):
    return {row['symbol']: row for row in rows
            if row['interface'] == 'Fortran' and row['adapter_lp64'] == 'true'
            and row['family'] not in BLAS_FAMILIES}


SOURCE_FILES = (
    'Project.toml', 'test/Project.toml', 'deps/upstream.toml', 'deps/build.jl',
    'deps/rust/Cargo.toml', 'deps/rust/Cargo.lock', 'src/Colmat.jl',
    'src/backend.jl', 'deps/rust/src/lib.rs', 'test/runtests.jl',
    'test/suite.jl', 'test/abi_execution_evidence.jl',
)


def execution_source_files(root):
    files = set(SOURCE_FILES)
    for directory, suffixes in (('src', ('.jl',)), ('test', ('.jl', '.c')), ('deps/rust/src', ('.rs',))):
        files.update(path.relative_to(root).as_posix() for path in (root / directory).rglob('*')
                     if path.is_file() and path.suffix in suffixes)
    return files


def validate_execution(root, rows, native_revision, data):
    root = Path(root).resolve()
    if data.get('schema_version') != 1 or data.get('full_suite_passed') is not True:
        raise ValueError('Execution evidence must come from a successful full suite')
    if data.get('errors') != []:
        raise ValueError('Execution recorder reported errors')
    if data.get('native_revision') != native_revision:
        raise ValueError('Execution evidence uses a stale native revision')
    if data.get('inventory_registration_sha256') != registration_digest(rows):
        raise ValueError('Execution evidence uses a different registration inventory')
    if data.get('manifest_sha256') != hashlib.sha256((root / 'Manifest.toml').read_bytes()).hexdigest():
        raise ValueError('Execution evidence uses a different dependency manifest')
    if not data.get('julia_version') or not data.get('platform'):
        raise ValueError('Execution evidence omits its Julia version or platform')
    if not re.fullmatch(r'[0-9a-f]{64}', data.get('library_sha256', '')):
        raise ValueError('Execution evidence omits its exact native library hash')
    revision = data.get('revision', '')
    if len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
        raise ValueError('Published evidence needs an exact repository revision')
    sources = data.get('source_sha256', {})
    if set(sources) != execution_source_files(root):
        raise ValueError('Execution evidence omits current production, test or worker sources')
    for file, digest in sources.items():
        path = (root / file).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f'Invalid execution evidence source: {file}')
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f'Stale execution evidence source: {file}')
    expected = lapack_rows(rows)
    executed = {symbol: [] for symbol in expected}
    for witness in data.get('witnesses', []):
        symbol = witness.get('symbol')
        if symbol not in expected:
            continue  # This snapshot's published scope is Fortran LAPACK.
        if witness.get('file') == 'test/abi_execution_checks.jl':
            continue  # Recorder probes are not numerical contract evidence.
        if (witness.get('bits') not in (32, 64)
                or witness.get('route') not in ('dll', 'lbt_trampoline')
                or witness.get('file') not in sources
                or not isinstance(witness.get('line'), int) or witness['line'] < 1
                or not witness.get('signature')
                or not isinstance(witness.get('integer_arguments'), bool)):
            raise ValueError(f'Incomplete concrete-call evidence: {symbol}')
        widths = {int(bits) for bits in re.findall(r'(?:Ref|Ptr)\{U?Int(32|64)\}', witness['signature'])}
        if (witness['integer_arguments'] and widths != {witness['bits']}) or (not witness['integer_arguments'] and widths):
            raise ValueError(f'Concrete signature width disagrees with executed ABI: {symbol}')
        if (not witness['integer_arguments']
                and expected[symbol]['family'] not in ('lartg', 'lartgp', 'lartgs')):
            raise ValueError(f'Missing integer signature for {symbol}')
        executed[symbol].append(witness)
    for symbol, witnesses in executed.items():
        if {w['bits'] for w in witnesses} != {32, 64}:
            raise ValueError(f'Missing executed integer ABI for {symbol}')
    return executed


def load_execution(root, rows, native_revision):
    path = Path(root) / SNAPSHOT
    if not path.exists():
        return {}, None
    data = json.loads(path.read_text())
    return validate_execution(root, rows, native_revision, data), data


def execution_fields(symbol, executed, provenance):
    if symbol not in executed:
        return dict(execution_evidence_status='not_catalogued', execution_evidence='')
    witnesses = executed[symbol]
    evidence = dict(snapshot='assets/lapack-abi-execution.json',
                    revision=provenance['revision'], integer_abis=[32, 64],
                    integer_arguments=any(w['integer_arguments'] for w in witnesses),
                    routes=sorted({w['route'] for w in witnesses}),
                    sources=sorted({w['file'] for w in witnesses}))
    return dict(execution_evidence_status='direct_call_witness',
                execution_evidence=json.dumps(evidence, separators=(',', ':')))
