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


# Six CBLAS dots were connected before the frozen #271 scope.
PREEXISTING_CBLAS_DOTS = {
    'cblas_sdot', 'cblas_ddot', 'cblas_cdotu_sub', 'cblas_cdotc_sub',
    'cblas_zdotu_sub', 'cblas_zdotc_sub',
}
# (leading Cint enums, by-value signed blasint arguments). Enums retain
# their C width even when dimensions/strides use the ILP64 interface.
CBLAS_INTEGER_LAYOUT = {}
for families, enums, dimensions in (
    ('scal real_scal asum nrm2 iamax iamin amax amin sum', 0, 2),
    ('copy swap axpy axpby dotu dotc rot real_rot', 0, 3),
    ('rotg', 0, 0), ('gemm gemm3m', 3, 6), ('gemmt gemmtr', 4, 5),
    ('gemv', 2, 5), ('gbmv', 2, 7), ('trmv trsv', 4, 3),
    ('tbmv tbsv', 4, 4), ('tpmv tpsv', 4, 2), ('trmm trsm', 5, 4),
    ('symv hemv', 2, 4), ('sbmv hbmv', 2, 5), ('spmv hpmv', 2, 3),
    ('ger geru gerc', 1, 5), ('syr her', 2, 3), ('spr hpr', 2, 2),
    ('syr2 her2', 2, 4), ('spr2 hpr2', 2, 3), ('symm hemm', 3, 5),
    ('syrk herk', 3, 4), ('syr2k her2k', 3, 5),
    ('imatcopy omatcopy', 2, 4), ('geadd', 1, 4),
):
    for family in families.split():
        CBLAS_INTEGER_LAYOUT[family] = enums, dimensions


def cblas_rows(rows):
    return {row['symbol']: row for row in rows
            if row['interface'] == 'CBLAS'
            and row['adapter_lp64'] == row['adapter_ilp64'] == 'true'
            and row['symbol'] not in PREEXISTING_CBLAS_DOTS}


def validate_cblas_signature(row, witness):
    enums, dimensions = CBLAS_INTEGER_LAYOUT[row['family']]
    signature = witness['signature']
    if not (signature.startswith('(') and signature.endswith(')')):
        raise ValueError(f'Invalid CBLAS signature: {row["symbol"]}')
    arguments = [part.strip() for part in signature[1:-1].split(',') if part.strip()]
    integers = [part for part in arguments[enums:] if re.fullmatch(r'U?Int(?:32|64)', part)]
    if (arguments[:enums] != ['Int32'] * enums
            or len(integers) != dimensions
            or any(part != f'Int{witness["bits"]}' for part in integers)
            or re.search(r'(?:Ref|Ptr)\{U?Int(?:32|64)\}', signature)
            or witness['integer_arguments'] != bool(dimensions)):
        raise ValueError(f'CBLAS enum/dimension signature disagrees with ABI: {row["symbol"]}')


# Policy helpers have no numerical dimensions and are validated by the separate
# C policy tests. The numerical LAPACKE evidence scope includes every other
# registered LAPACKE entry, including any newly added families.
LAPACKE_POLICY_HELPERS = {'LAPACKE_get_nancheck', 'LAPACKE_set_nancheck'}


def lapacke_rows(rows):
    return {row['symbol']: row for row in rows
            if row['interface'] == 'LAPACKE'
            and row['adapter_lp64'] == row['adapter_ilp64'] == 'true'
            and row['symbol'] not in LAPACKE_POLICY_HELPERS}


def validate_lapacke_signature(row, witness):
    integer = f'Int{witness["bits"]}'
    scalar = {'s': 'Float32', 'd': 'Float64', 'c': 'ComplexF32', 'z': 'ComplexF64'}[row['scalar']]
    t, pivot = f'Ptr{{{scalar}}}', f'Ptr{{{integer}}}'
    real_scalar = {'s': 'Float32', 'd': 'Float64', 'c': 'Float32', 'z': 'Float64'}[row['scalar']]
    real = f'Ptr{{{real_scalar}}}'
    args = {
        'getrf': ['Int32', integer, integer, t, integer, pivot],
        'getf2': ['Int32', integer, integer, t, integer, pivot],
        'getrf2': ['Int32', integer, integer, t, integer, pivot],
        'getrs': ['Int32', 'Int8', integer, integer, t, integer, pivot, t, integer],
        'gesv': ['Int32', integer, integer, t, integer, pivot, t, integer],
        'getri': ['Int32', integer, t, integer, pivot],
        'potrf': ['Int32', 'Int8', integer, t, integer],
        'potrf2': ['Int32', 'Int8', integer, t, integer],
        'potri': ['Int32', 'Int8', integer, t, integer],
        'lauum': ['Int32', 'Int8', integer, t, integer],
        'lacpy': ['Int32', 'Int8', integer, integer, t, integer, t, integer],
        'laset': ['Int32', 'Int8', integer, integer, scalar, scalar, t, integer],
        'laswp': ['Int32', integer, t, integer, integer, integer, pivot, integer],
        'tgevc': ['Int32', 'Int8', 'Int8', pivot, integer, t, integer, t, integer, t, integer, t, integer, integer, pivot],
        'lange': ['Int32', 'Int8', integer, integer, t, integer],
        'lansy': ['Int32', 'Int8', 'Int8', integer, t, integer],
        'lantr': ['Int32', 'Int8', 'Int8', 'Int8', integer, integer, t, integer],
        'lapmr': ['Int32', integer, integer, integer, t, integer, pivot],
        'lapmt': ['Int32', integer, integer, integer, t, integer, pivot],
        'orgtr': ['Int32', 'Int8', integer, t, integer, t],
        'ungtr': ['Int32', 'Int8', integer, t, integer, t],
        'ormtr': ['Int32', 'Int8', 'Int8', 'Int8', integer, integer, t, integer, t, t, integer],
        'unmtr': ['Int32', 'Int8', 'Int8', 'Int8', integer, integer, t, integer, t, t, integer],
        'orgbr': ['Int32', 'Int8', integer, integer, integer, t, integer, t],
        'ungbr': ['Int32', 'Int8', integer, integer, integer, t, integer, t],
        'orghr': ['Int32', integer, integer, integer, t, integer, t],
        'unghr': ['Int32', integer, integer, integer, t, integer, t],
        'ormbr': ['Int32', 'Int8', 'Int8', 'Int8', integer, integer, integer, t, integer, t, t, integer],
        'unmbr': ['Int32', 'Int8', 'Int8', 'Int8', integer, integer, integer, t, integer, t, t, integer],
        'ormhr': ['Int32', 'Int8', 'Int8', integer, integer, integer, integer, t, integer, t, t, integer],
        'unmhr': ['Int32', 'Int8', 'Int8', integer, integer, integer, integer, t, integer, t, t, integer],
        'trevc': ['Int32', 'Int8', 'Int8', pivot, integer, t, integer, t, integer, t, integer, integer, pivot],
        'trexc': [],
        'trsen': [],
        'lascl': ['Int32', 'Int8', integer, integer, real_scalar, real_scalar, integer, integer, t, integer],
        'lacgv': [integer, t, integer],
        'larfg': [integer, t, t, integer, t],
        'larft': ['Int32', 'Int8', 'Int8', integer, integer, t, integer, t, t, integer],
        'larfb': ['Int32', 'Int8', 'Int8', 'Int8', 'Int8', integer, integer, integer, t, integer, t, integer, t, integer],
        'lartgp': [real_scalar, real_scalar, real, real, real],
        'lartgs': [real_scalar, real_scalar, real_scalar, real, real],
        'gees': [],
        'geesx': [],
        'ggev': [],
        'ggev3': [],
        'ggevx': [],
        'gghrd': ['Int32', 'Int8', 'Int8', integer, integer, integer, t, integer, t, integer, t, integer, t, integer],
        'gghd3': ['Int32', 'Int8', 'Int8', integer, integer, integer, t, integer, t, integer, t, integer, t, integer],
        'hgeqz': [],
        'upgtr': ['Int32', 'Int8', integer, t, t, t, integer],
        'upmtr': ['Int32', 'Int8', 'Int8', 'Int8', integer, integer, t, t, t, integer],
        'lacp2': ['Int32', 'Int8', integer, integer, real, integer, t, integer],
        'tzrzf': ['Int32', integer, integer, t, integer, t],
        'ggqrf': ['Int32', integer, integer, integer, t, integer, t, t, integer, t],
        'ggrqf': ['Int32', integer, integer, integer, t, integer, t, t, integer, t],
        'gglse': ['Int32', integer, integer, integer, t, integer, t, integer, t, t, t],
        'ggglm': ['Int32', integer, integer, integer, t, integer, t, integer, t, t, t],
        'gels': ['Int32', 'Int8', integer, integer, integer, t, integer, t, integer],
        'gelsy': ['Int32', integer, integer, integer, t, integer, t, integer, pivot, real_scalar, pivot],
        'gelss': ['Int32', integer, integer, integer, t, integer, t, integer, real, real_scalar, pivot],
        'gelsd': ['Int32', integer, integer, integer, t, integer, t, integer, real, real_scalar, pivot],
        'tpqrt': ['Int32', integer, integer, integer, integer, t, integer, t, integer, t, integer],
        'tpqrt2': ['Int32', integer, integer, integer, t, integer, t, integer, t, integer],
        'tpmqrt': ['Int32', 'Int8', 'Int8', integer, integer, integer, integer, integer, t, integer, t, integer, t, integer, t, integer],
        'tprfb': ['Int32', 'Int8', 'Int8', 'Int8', 'Int8', integer, integer, integer, integer, t, integer, t, integer, t, integer, t, integer],
        'ormrz': ['Int32', 'Int8', 'Int8', integer, integer, integer, integer, t, integer, t, t, integer],
        'unmrz': ['Int32', 'Int8', 'Int8', integer, integer, integer, integer, t, integer, t, t, integer],
        'geqp3': ['Int32', integer, integer, t, integer, pivot, t],
        'geqpf': ['Int32', integer, integer, t, integer, pivot, t],
        'geqrt': ['Int32', integer, integer, integer, t, integer, t, integer],
        'geqrt2': ['Int32', integer, integer, t, integer, t, integer],
        'geqrt3': ['Int32', integer, integer, t, integer, t, integer],
        'gemqrt': ['Int32', 'Int8', 'Int8', integer, integer, integer, integer, t, integer, t, integer, t, integer],
        'geqrf': ['Int32', integer, integer, t, integer, t],
        'geqrfp': ['Int32', integer, integer, t, integer, t],
        'geqr2': ['Int32', integer, integer, t, integer, t],
        'gelqf': ['Int32', integer, integer, t, integer, t],
        'gelq2': ['Int32', integer, integer, t, integer, t],
        'geqlf': ['Int32', integer, integer, t, integer, t],
        'gerqf': ['Int32', integer, integer, t, integer, t],
        'orgqr': ['Int32', integer, integer, integer, t, integer, t],
        'ungqr': ['Int32', integer, integer, integer, t, integer, t],
        'orglq': ['Int32', integer, integer, integer, t, integer, t],
        'unglq': ['Int32', integer, integer, integer, t, integer, t],
        'orgql': ['Int32', integer, integer, integer, t, integer, t],
        'ungql': ['Int32', integer, integer, integer, t, integer, t],
        'orgrq': ['Int32', integer, integer, integer, t, integer, t],
        'ungrq': ['Int32', integer, integer, integer, t, integer, t],
        'ormqr': ['Int32', 'Int8', 'Int8', integer, integer, integer, t, integer, t, t, integer],
        'unmqr': ['Int32', 'Int8', 'Int8', integer, integer, integer, t, integer, t, t, integer],
        'ormlq': ['Int32', 'Int8', 'Int8', integer, integer, integer, t, integer, t, t, integer],
        'unmlq': ['Int32', 'Int8', 'Int8', integer, integer, integer, t, integer, t, t, integer],
        'ormql': ['Int32', 'Int8', 'Int8', integer, integer, integer, t, integer, t, t, integer],
        'unmql': ['Int32', 'Int8', 'Int8', integer, integer, integer, t, integer, t, t, integer],
        'ormrq': ['Int32', 'Int8', 'Int8', integer, integer, integer, t, integer, t, t, integer],
        'unmrq': ['Int32', 'Int8', 'Int8', integer, integer, integer, t, integer, t, t, integer],
        'trtri': ['Int32', 'Int8', 'Int8', integer, t, integer],
        'trtrs': ['Int32', 'Int8', 'Int8', 'Int8', integer, integer, t, integer, t, integer],
        'potrs': ['Int32', 'Int8', integer, integer, t, integer, t, integer],
        'posv': ['Int32', 'Int8', integer, integer, t, integer, t, integer],
        'syr': ['Int32', 'Int8', integer, scalar, t, integer, t, integer],
        'gesvd': ['Int32', 'Int8', 'Int8', integer, integer, t, integer, real, t, integer, t, integer],
        'gesdd': ['Int32', 'Int8', integer, integer, t, integer, real, t, integer, t, integer],
        'geev': ['Int32', 'Int8', 'Int8', integer, t, integer, t]
                + ([real] if row['scalar'] in ('s', 'd') else [])
                + [t, integer, t, integer],
        'geevx': ['Int32', 'Int8', 'Int8', 'Int8', 'Int8', integer, t, integer, t]
                 + ([real] if row['scalar'] in ('s', 'd') else [])
                 + [t, integer, t, integer, pivot, pivot, real, real, real, real],
        'syev': ['Int32', 'Int8', 'Int8', integer, t, integer, real],
        'heev': ['Int32', 'Int8', 'Int8', integer, t, integer, real],
        'syevd': ['Int32', 'Int8', 'Int8', integer, t, integer, real],
        'heevd': ['Int32', 'Int8', 'Int8', integer, t, integer, real],
        'syevr': ['Int32', 'Int8', 'Int8', 'Int8', integer, t, integer, real_scalar, real_scalar, integer, integer, real_scalar, pivot, real, t, integer, pivot],
        'heevr': ['Int32', 'Int8', 'Int8', 'Int8', integer, t, integer, real_scalar, real_scalar, integer, integer, real_scalar, pivot, real, t, integer, pivot],
    }
    if row['family'] == 'syr' and row['scalar'] not in ('c', 'z'):
        raise ValueError('LAPACKE complex SYR contract requires a complex scalar')
    if row['family'] not in args:
        raise ValueError(f'LAPACKE signature contract missing for {row["symbol"]}')
    expected = args[row['family']]
    if row['family'] in ('geqrf', 'geqrfp', 'geqr2', 'gelqf', 'gelq2', 'geqlf', 'gerqf', 'tzrzf') and row['symbol'].endswith('_work'):
        expected = expected + [t] + ([integer] if row['family'] not in ('geqr2', 'gelq2') else [])
    if row['family'] in ('orgqr', 'ungqr', 'orglq', 'unglq', 'orgql', 'ungql', 'orgrq', 'ungrq') and row['symbol'].endswith('_work'):
        expected = expected + [t, integer]
    if row['family'] in ('ormqr', 'unmqr', 'ormlq', 'unmlq', 'ormql', 'unmql', 'ormrq', 'unmrq', 'ormrz', 'unmrz') and row['symbol'].endswith('_work'):
        expected = expected + [t, integer]
    if row['family'] in ('geqp3', 'geqpf') and row['symbol'].endswith('_work'):
        expected = expected + [t] + ([integer] if row['family'] == 'geqp3' else [])
        if row['scalar'] in ('c', 'z'):
            expected = expected + [real]
    if row['family'] in ('tpqrt', 'tpmqrt', 'tprfb') and row['symbol'].endswith('_work'):
        expected = expected + [t] + ([integer] if row['family'] == 'tprfb' else [])
    if row['family'] in ('ggqrf', 'ggrqf', 'gglse', 'ggglm') and row['symbol'].endswith('_work'):
        expected = expected + [t, integer]
    if row['family'] == 'gels' and row['symbol'].endswith('_work'):
        expected = expected + [t, integer]
    if row['family'] in ('gelsy', 'gelss', 'gelsd') and row['symbol'].endswith('_work'):
        expected = expected + [t, integer] + ([real] if row['scalar'] in ('c', 'z') else [])
        if row['family'] == 'gelsd':
            expected = expected + [pivot]
    if row['family'] in ('geqrt', 'gemqrt') and row['symbol'].endswith('_work'):
        expected = expected + [t]
    if row['family'] == 'getri' and row['symbol'].endswith('_work'):
        expected = expected + [t, integer]
    if row['family'] == 'trsen':
        if row['scalar'] in ('s', 'd'):
            expected = ['Int32', 'Int8', 'Int8', pivot, integer, t, integer, t, integer, t, t, pivot, real, real]
        else:
            expected = ['Int32', 'Int8', 'Int8', pivot, integer, t, integer, t, integer, t, pivot, real, real]
    if row['family'] == 'trsen' and row['symbol'].endswith('_work'):
        expected = expected + ([t, integer, pivot, integer] if row['scalar'] in ('s', 'd') else [t, integer])
    if row['family'] == 'trexc':
        if row['scalar'] in ('s', 'd'):
            expected = ['Int32', 'Int8', integer, t, integer, t, integer, pivot, pivot]
        else:
            expected = ['Int32', 'Int8', integer, t, integer, t, integer, integer, integer]
    if row['family'] == 'trexc' and row['symbol'].endswith('_work'):
        expected = expected + [t]
    if row['family'] == 'trevc' and row['symbol'].endswith('_work'):
        expected = expected + [t] + ([real] if row['scalar'] in ('c', 'z') else [])
    if row['family'] in ('upgtr', 'upmtr') and row['symbol'].endswith('_work'):
        expected = expected + [t]
    if row['family'] == 'hgeqz':
        if row['scalar'] in ('s', 'd'):
            expected = ['Int32', 'Int8', 'Int8', 'Int8', integer, integer, integer, t, integer, t, integer, t, t, t, t, integer, t, integer]
        else:
            expected = ['Int32', 'Int8', 'Int8', 'Int8', integer, integer, integer, t, integer, t, integer, t, t, t, integer, t, integer]
        if row['symbol'].endswith('_work'):
            expected = expected + [t, integer] + ([real] if row['scalar'] in ('c', 'z') else [])
    if row['family'] == 'gghd3' and row['symbol'].endswith('_work'):
        expected = expected + [t, integer]
    if row['family'] == 'ggevx':
        if row['scalar'] in ('s', 'd'):
            expected = ['Int32', 'Int8', 'Int8', 'Int8', 'Int8', integer, t, integer, t, integer, t, t, t, t, integer, t, integer, pivot, pivot, t, t, t, t, t, t]
        else:
            expected = ['Int32', 'Int8', 'Int8', 'Int8', 'Int8', integer, t, integer, t, integer, t, t, t, integer, t, integer, pivot, pivot, real, real, real, real, real, real]
        if row['symbol'].endswith('_work'):
            expected = expected + [t, integer] + ([real, pivot, pivot] if row['scalar'] in ('c', 'z') else [pivot, pivot])
    if row['family'] in ('ggev', 'ggev3'):
        if row['scalar'] in ('s', 'd'):
            expected = ['Int32', 'Int8', 'Int8', integer, t, integer, t, integer, t, t, t, t, integer, t, integer]
        else:
            expected = ['Int32', 'Int8', 'Int8', integer, t, integer, t, integer, t, t, t, integer, t, integer]
        if row['symbol'].endswith('_work'):
            expected = expected + [t, integer] + ([real] if row['scalar'] in ('c', 'z') else [])
    if row['family'] == 'geesx':
        expected = ['Int32', 'Int8', 'Int8', 'Ptr{Nothing}', 'Int8', integer, t, integer, pivot, t] + \
            ([] if row['scalar'] in ('c', 'z') else [t]) + [t, integer, real, real]
        if row['symbol'].endswith('_work'):
            expected = expected + [t, integer] + ([real, pivot] if row['scalar'] in ('c', 'z') else [pivot, integer, pivot])
    if row['family'] == 'gees':
        expected = ['Int32', 'Int8', 'Int8', 'Ptr{Nothing}', integer, t, integer, pivot, t] + \
            ([] if row['scalar'] in ('c', 'z') else [t]) + [t, integer]
        if row['symbol'].endswith('_work'):
            expected = expected + [t, integer] + ([real, pivot] if row['scalar'] in ('c', 'z') else [pivot])
    if row['family'] == 'larfb' and row['symbol'].endswith('_work'):
        expected = expected + [t, integer]
    if row['family'] in ('orgtr', 'ungtr', 'ormtr', 'unmtr', 'orgbr', 'ungbr', 'orghr', 'unghr', 'ormbr', 'unmbr', 'ormhr', 'unmhr') and row['symbol'].endswith('_work'):
        expected = expected + [t, integer]
    if row['family'] in ('lange', 'lansy', 'lantr') and row['symbol'].endswith('_work'):
        expected = expected + [real]
    if row['family'] == 'geev' and row['symbol'].endswith('_work'):
        expected = expected + [t, integer]
        if row['scalar'] in ('c', 'z'):
            expected = expected + [real]
    if row['family'] == 'geevx' and row['symbol'].endswith('_work'):
        expected = expected + [t, integer, pivot if row['scalar'] in ('s', 'd') else real]
    if row['family'] in ('gesvd', 'gesdd'):
        if row['symbol'].endswith('_work'):
            expected = expected + [t, integer]
            if row['scalar'] in ('c', 'z'):
                expected = expected + [real]
            if row['family'] == 'gesdd':
                expected = expected + [pivot]
        elif row['family'] == 'gesvd':
            expected = expected + [real]
    if row['family'] in ('syev', 'heev') and row['symbol'].endswith('_work'):
        expected = expected + [t, integer]
        if row['family'] == 'heev':
            expected = expected + [real]
    if row['family'] in ('syevd', 'heevd') and row['symbol'].endswith('_work'):
        expected = expected + [t, integer]
        if row['family'] == 'heevd':
            expected = expected + [real, integer]
        expected = expected + [pivot, integer]
    if row['family'] in ('syevr', 'heevr') and row['symbol'].endswith('_work'):
        expected = expected + [t, integer]
        if row['family'] == 'heevr':
            expected = expected + [real, integer]
        expected = expected + [pivot, integer]
    actual = witness['signature'].replace(' ', '')
    expected_return = real_scalar if row['family'] in ('lange', 'lansy', 'lantr') else integer
    expected_integer_arguments = row['family'] not in ('lartg', 'lartgp', 'lartgs')
    if (actual != '(' + ','.join(expected) + ')'
            or witness.get('return_type') != expected_return
            or witness['integer_arguments'] is not expected_integer_arguments):
        raise ValueError(f'LAPACKE layout/flag/integer/return signature disagrees with ABI: {row["symbol"]}')


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
    lapack = lapack_rows(rows)
    cblas = cblas_rows(rows)
    lapacke = lapacke_rows(rows)
    expected = lapack | cblas | lapacke
    if (cblas or 'cblas_target_symbols' in data) and data.get('cblas_target_symbols') != sorted(cblas):
        raise ValueError('CBLAS execution scope differs from registered targets')
    if (lapacke or "lapacke_target_symbols" in data) and data.get("lapacke_target_symbols") != sorted(lapacke):
        raise ValueError("LAPACKE execution scope differs from registered targets")
    executed = {symbol: [] for symbol in expected}
    for witness in data.get('witnesses', []):
        symbol = witness.get('symbol')
        if symbol not in expected:
            continue  # Other BLAS calls are outside the published witness scope.
        if witness.get('file') == 'test/abi_execution_checks.jl':
            continue  # Recorder probes are not numerical contract evidence.
        if (witness.get('bits') not in (32, 64)
                or witness.get('route') not in ('dll', 'lbt_trampoline')
                or witness.get('file') not in sources
                or not isinstance(witness.get('line'), int) or witness['line'] < 1
                or not witness.get('signature')
                or not isinstance(witness.get('integer_arguments'), bool)):
            raise ValueError(f'Incomplete concrete-call evidence: {symbol}')
        if symbol in lapacke:
            if witness['file'] in ('test/lapacke_execution_checks.jl', 'test/lapacke_calls.jl'):
                raise ValueError(f'LAPACKE registration/recorder probe is not execution: {symbol}')
            validate_lapacke_signature(lapacke[symbol], witness)
            executed[symbol].append(witness)
            continue
        if symbol in cblas:
            if witness['file'] in ('test/cblas.jl', 'test/cblas_calls.jl', 'test/cblas_execution_checks.jl'):
                raise ValueError(f'CBLAS registration/recorder probe is not execution: {symbol}')
            validate_cblas_signature(cblas[symbol], witness)
            executed[symbol].append(witness)
            continue
        widths = {int(bits) for bits in re.findall(r'(?:Ref|Ptr)\{U?Int(32|64)\}', witness['signature'])}
        if (witness['integer_arguments'] and widths != {witness['bits']}) or (not witness['integer_arguments'] and widths):
            raise ValueError(f'Concrete signature width disagrees with executed ABI: {symbol}')
        if (not witness['integer_arguments']
                and expected[symbol]['family'] not in ('lartg', 'lartgp', 'lartgs')):
            raise ValueError(f'Missing integer signature for {symbol}')
        executed[symbol].append(witness)
    for symbol, witnesses in executed.items():
        if symbol in (cblas | lapacke) and {(w['bits'], w['route']) for w in witnesses} != {
                (bits, route) for bits in (32, 64) for route in ('dll', 'lbt_trampoline')}:
            raise ValueError(f'Missing completed {"LAPACKE" if symbol in lapacke else "CBLAS"} ABI/route for {symbol}')
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
