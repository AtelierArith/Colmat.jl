"""Regenerate every canonical LBT symbol and its pinned colmat API assessment.

Run from the repository root after building Colmat. Python 3.11+ and Julia are
needed only for this audit command, not to load or build the package.
"""
from pathlib import Path
from collections import defaultdict, Counter
import csv
import json
import re
import subprocess
import tomllib
from lbt_classification import classify as classify_capability
from lbt_execution import load_execution, execution_fields

ROOT = Path(__file__).resolve().parent.parent
UPSTREAM = tomllib.loads((ROOT / 'deps/upstream.toml').read_text())
SOURCE = ROOT / 'deps/colmat-rs'
POLICY = json.loads((ROOT / 'docs/lbt-capabilities.json').read_text())
if POLICY['assessment_revision'] != UPSTREAM['revision']:
    raise RuntimeError('Capability policy needs review for the current native revision')
GROUPS = {}
for group in POLICY['groups']:
    for name in group['routines']:
        GROUPS[name] = group

LABELS = {
    "native': 'Public numerical capability available",
    "representation': 'Supported; storage conversion required",
    "composition': 'Composition candidate; no direct API",
    "conditional': 'Partial; additional contract implementation required",
    "unsupported': 'No independent public API (within reviewed scope)",
    "unverified': 'Unverified; insufficient evidence",
}
SPECIAL = {
    'scnrm2': ('nrm2', 'c'), 'dznrm2': ('nrm2', 'z'),
    'scasum': ('asum', 'c'), 'dzasum': ('asum', 'z'),
    'csscal': ('real_scal', 'c'), 'zdscal': ('real_scal', 'z'),
    'csrot': ('real_rot', 'c'), 'zdrot': ('real_rot', 'z'),
}


def routine(symbol):
    interface = 'Fortran'
    name = symbol.rstrip('_')
    if name.startswith('LAPACKE_'):
        interface = 'LAPACKE'
        name = name[8:]
        if name in ('get_nancheck', 'set_nancheck'):
            return interface, name, 'other'
        if name.endswith('_work'):
            name = name[:-5]
    elif name.startswith('cblas_'):
        interface = 'CBLAS'
        name = name[6:]
        if name.endswith('_sub'):
            name = name[:-4]
    if name in SPECIAL:
        family, scalar = SPECIAL[name]
        return interface, family, scalar
    match = re.fullmatch(r'i([sdcz])a(max|min)', name)
    if match:
        return interface, 'ia' + match[2], match[1]
    if len(name) > 1 and name[0] in 'sdcz':
        return interface, name[1:], name[0]
    return interface, name, 'other'


def classify(family, scalar):
    return classify_capability(family, scalar, GROUPS, UPSTREAM['revision'],
                               POLICY.get('scalar_overrides', []))


raw = subprocess.check_output(
    ['julia','--startup-file=no','--project','docs/dump_lbt_inventory.jl'], cwd=ROOT, text=True
)
rows = []
for line in raw.splitlines():
    symbol, lp, ilp = line.split('\t')
    interface, family, scalar = routine(symbol)
    group = classify(family, scalar)
    source = group['source']
    if not (SOURCE / source).is_file():
        raise RuntimeError(f'Missing evidence source: {source}')
    if group['evidence']:
        for evidence_source in group['evidence']['sources']:
            if not (SOURCE / evidence_source).is_file():
                raise RuntimeError(f'Missing reviewed source: {evidence_source}')
    note = group['note']
    if interface == 'CBLAS' and lp == ilp == 'true':
        note += '; C ABI connected for LP64/ILP64 with row/column-major and explicit enum/coefficient/return conversions; #271 tests DLL and LBT routes'
    elif interface == 'LAPACKE' and lp == ilp == 'true':
        note += '; C ABI connected for LP64/ILP64; #288 tests explicit layout/char/integer/INFO contracts and DLL/LBT routes; NaN policy helpers retain C-int flags in both interfaces'
    elif interface != 'Fortran':
        note += '; C ABI/row-major/enum/return values require separate conversion'
    if family == 'geevx' and lp == ilp == 'true':
        note += '; the current native path is SENSE=N. Condition estimation is delegated to the saved original backend. If there is no original LP64 implementation, integers are converted to ILP64. Configurations lacking the original ILP64 GEEVX are rejected before LBT registration'
    rows.append(dict(symbol=symbol, interface=interface, family=family, scalar=scalar,
        colmat_api=LABELS[group['status']], adapter_lp64=lp, adapter_ilp64=ilp,
        entry=group['api'], source=source, contract=note,
        capability_status=group['status'], evidence_status=group['evidence_status'],
        capability_evidence=json.dumps(group['evidence'], ensure_ascii=False, separators=(',', ':')) if group['evidence'] else '',
        execution_evidence_status='not_catalogued', execution_evidence='',
        independent_algorithm_status=group['algorithm_status']))
assert len(rows) == len({r['symbol'] for r in rows})
executed, execution_provenance = load_execution(ROOT, rows, UPSTREAM['revision'])
for row in rows:
    row.update(execution_fields(row['symbol'], executed, execution_provenance))
assets = ROOT / 'docs/src/assets'
assets.mkdir(exist_ok=True)
with (assets/'lbt-symbol-inventory.tsv').open('w') as out:
    writer = csv.DictWriter(out, fieldnames=list(rows[0]), delimiter='\t', lineterminator='\n')
    writer.writeheader(); writer.writerows(rows)
meta = dict(julia=subprocess.check_output(['julia','--startup-file=no','-e','print(VERSION)'],text=True),
    revision=UPSTREAM['revision'], repository=UPSTREAM['repository'], symbols=len(rows),
    interfaces=dict(Counter(r['interface'] for r in rows)),
    lp64=sum(r['adapter_lp64']=='true' for r in rows),
    ilp64=sum(r['adapter_ilp64']=='true' for r in rows),
    capability_assessments=dict(Counter(r['capability_status'] for r in rows)),
    execution_evidence=dict(Counter(r['execution_evidence_status'] for r in rows)),
    execution_source_revision=execution_provenance['revision'] if execution_provenance else None,
    independent_algorithm_assessments=dict(Counter(r['independent_algorithm_status'] for r in rows)))
(assets/'lbt-symbol-inventory.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
lines = ['# Full LBT and colmat-rs routine mapping','',
    "## Scope and classification','",
    f"Lists **all {len(rows):,} canonical symbols** exposed by LBT in Julia {meta['julia']}.",
    'LP64 / ILP64 are separate tables with the same canonical names. Types and ABIs are not omitted to claim blanket support.',
    f"The colmat-rs target is `{UPSTREAM['revision']}` in `deps/upstream.toml`.",
    'This is a static audit of connection candidates based on the public source. It is not the result of executing every candidate below.',
    "Results of actual Julia calls are recorded separately in [the measured Julia API mapping](@ref).','",
    '- **Public numerical capability available**: Numerical kernels such as LU, SVD, and eigenvalue decomposition are public. This does not mean the ABI is identical.',
    '- **Storage conversion required**: Differences exist in reflectors, permutations, band/packed storage, and so on. An adapter is not necessarily available yet.',
    '- **Composition candidate**: A candidate whose mathematical output can be produced by combining public kernels. Equivalence in LAPACK-specific algorithms, accuracy guarantees, or complexity is not claimed.',
    '- **Partial**: Basic factorizations exist, but condition estimates, fixed pivots, expert flags, and similar features are missing.',
    '- **No independent public API (within reviewed scope)**: A determination within the recorded scope of revisions, types, storage formats, options, and semantic verification methods. It does not mean the mathematics cannot be implemented.',
    '- **Unverified; insufficient evidence**: Unregistered families, guesses based on names, and old unsupported classifications with insufficient confirmation records. Does not mean the public API is absent.',
    '- `capability_status` and `capability_evidence` record the public-feature assessment; `adapter_lp64/ilp64` records the actual ABI connection; `execution_evidence_status` records the registration state of per-entry execution evidence; and `independent_algorithm_status` records verification of independent algorithms, all separately. `not_catalogued` does not mean untested. Do not infer the presence of an independent algorithm from `not_assessed`.',
    "- `adapter` is the actual LBT registration. Do not confuse the existence of a numerical feature, the ABI connection, and completed verification.','",
    "## Machine-readable list of all symbols','",
    '[The all-symbol TSV](../assets/lbt-symbol-inventory.tsv) contains **one row per symbol**, with the name, C/Fortran ABI, type, public-feature classification, LP64/ILP64 connection status, source evidence, and restrictions.',
    "[The generation-metadata JSON](../assets/lbt-symbol-inventory.json) is also saved.','",
    '| ABI | Canonical symbols |','|---|---|']
for interface,count in meta['interfaces'].items(): lines.append(f'| {interface} | {count:,} |')
lines += ['',f"The current adapter registers LP64 **{meta['lp64']}** and ILP64 **{meta['ilp64']}** symbols.",
    "GEEVX condition estimation is delegated to the existing backend rather than the native one. This is not a change that provides all of LAPACKE.','",
    "## LAPACK connected in this round','",
    "| Family | Numerical processing | Restrictions |','|---|---|---|",
    '| GEQRT / GEQRT2 / GEQRT3 / GEMQRT | Native QR coefficient conversion and block reflector application | Allocates internal arrays. GEQRT3 uses the common path |',
    '| GETRF / GETRF2 / GETF2 / GETRS / GETRI / GESV | colmat-rs LU and triangular solve | Converts the final permutation to IPIV |',
    '| POTRF / POTRF2 / POTF2 / POTRS / POTRI / POSV | colmat-rs LLT and triangular solve | Converts UPLO and INFO |',
    '| GESDD / GESVD | colmat-rs SVD | Converts full/thin/overwrite and VT. Both drivers connect to the common native SVD |',
    '| GEEV / GEEVX | colmat-rs balancing, EVD, and Schur | GEEVX uses native SENSE=N. Expert modes use the original implementation; LP64 converts integers to ILP64 as needed. Requires an ILP64 implementation before registration |',
    '| GGEV / GGEV3 / GEGV / GGEVX | colmat-rs GEVD, homogeneous alpha/beta, and left/right eigenvectors | GGEVX uses native BALANC=N/SENSE=N. Other modes use the existing ILP64 backend with integer ABI conversion |',
    '| SYEV / HEEV / SYEVD / HEEVD | colmat-rs self-adjoint EVD | No guarantee of running the same Netlib algorithm as the driver name |',
    '| LARFG / LARFGP / LARF / LARFX / LARFY / LARFT / LARFB | Rust reflector generation and native GEMV/SYMV/GEMM | Converts LAPACK coefficients, F/B, and C/R. LARFB allocates internal arrays |',
    '| GGQRF / GGRQF / GGLSE / GGGLM | native-backed QR/RQ, Q application, and triangular solve composition | External reflector compatibility, equality constraints, minimum norm, and INFO=1/2 for exact rank failure. Allocates internal coefficients and scratch |',
    '| TREXC / LAEXC / TRSEN / GEESX | Native checked Schur moves and the existing GEES | Condition estimation uses the saved original ILP64 routines. Converts rejection indices, the total selected SDIM, integer work arrays, and the callback ABI |',
    '| TREVC / TREVC3 | Public Schur eigenvectors, reversed adjoint, and back-transformation | SIDE L/R/B, HOWMNY A/B/S, SELECT/M for real pairs, and max-component normalization. Native rejections use the saved original LAPACK |',
    '| LACPY / LASET / LASWP | Native LACPY/LASET/LASWP and direct row-major conversion | Reads and writes only the logical rectangle and selected triangle. LASWP swaps whole rows and preserves Fortran ordering for INCX<0 |',
    "| TGEVC | Eigenvector back-substitution from an existing S/P Schur pencil and Q/Z back-transformation | Ports LAPACK's guarded small solves to Rust. SIDE L/R/B, HOWMNY A/S/B, real conjugate pairs, infinite eigenvalues, and max-component normalization. C API is #288 |",
    '| LANGE / LANSY / LANTR | Native matrix norms and row-major transpose views | M/1/O/I/F/E, symmetric/triangular storage, and the unit diagonal. Returns a real value, not INFO |',
    '| GESVDQ | SVD with native QR preprocessing | Transposes A and writes it back (QR factors); U is M×M for JOBU=A, N×N for F, and M×N otherwise (R has NUMRANK columns); V is written back as the logical NUMRANK or N rows by N columns. High-level entry points query IWORK/WORK/RWORK |',
    '| GGESX | Native generalized Schur decomposition (reordering and condition estimation) | Passes the SELECTCTG callback to native reordering and transposes A/B/VSL/VSR before writing everything back. Queries WORK/LWORK/LIWORK, and the high-level entry adds slack to LWORK for the TGSYL stage of SENSE |',
    '| TGSEN | Native cluster reordering of a generalized Schur pair | SELECT is a LOGICAL (integer) array. Transposes A/B/Q/Z and writes everything back, querying WORK/LWORK/LIWORK. The TGSYL stage for IJOB>=1 needs at least one more LWORK element than the queried minimum, so the high-level entry adds slack |',
    '| GESVDX | Native selected singular value decomposition | RANGE=A/V/I; row-major U is M×NS and VT is NS×N, writing back only the selected NS pairs. The high-level entry queries WORK/LWORK and uses SUPERB as integer workspace |',
    '| BDSQR / BDSDC / BDSVDX / LASQ1 | Native bidiagonal SVD and selection | U/VT/C application, COMPQ N/I, RANGE A/V/I, and ascending output. The compact P of BDSDC is delegated to the saved original ILP64 LAPACK |',
    '| LAPMR / LAPMT | Native row/column permutation | The row-major path converts only the logical rectangle. Preserves K and forward/backward |',
    '| ORGTR/UNGTR and ORMTR/UNMTR | Native tridiagonal Q formation and application | The row-major path preserves the selected triangle and converts the logical rectangle. Converts UPLO/SIDE/TRANS and the lwork query |',
    '| ORGBR/ORMBR and ORGHR/ORMHR | Native bidiagonal/Hessenberg Q formation and application | Converts VECT/SIDE/TRANS/ILO/IHI and the lwork query. The row-major path handles only the logical rectangle |',
    '| TREVC | Native Schur eigenvectors | SIDE/HOWMNY/SELECT/M, real conjugate pairs, and row-major logical-rectangle conversion. C ABI is #288 |',
    '| TREXC | Native Schur block move | COMPQ V/N, IFST/ILST, and the ABI difference between real pointers and complex by-value arguments. C ABI is #288 |',
    '| LASCL | Native safe-range scaling | TYPE G/L/U/H/B/Q/Z, the number of row-major band rows, CFROM/CTO, and INFO. C ABI is #288 |',
    '| LACGV / LACP2 | Native conjugation and real-to-complex copy | INCX and the selected triangle, plus row-major conversion from real A to complex B. LACGV has no layout argument |',
    '| TRSEN | Native checked Schur selection (JOB=N) | SELECT/M, reordering, and row-major T/Q. JOB E/V/B use the saved original LAPACK |',
    '| LARFG | Native Householder generator | No layout argument; converts INCX and TAU/alpha/x directly |',
    '| LARFT | Native block reflector triangular factor | DIRECT/STOREV and row-major V/T conversion. Preserves the transpose convention of T |',
    '| LARFB | Native block reflector application | SIDE/TRANS/DIRECT/STOREV and row-major V/T/C conversion. WORK length is K |',
    '| LARTGP / LARTGS | Native Givens rotation with scalar guards | No layout argument; converts by-value f/g/sigma and the NaN ordering |',
    '| GEES | Native Schur driver (SORT=N/S) | Passes the SELECT callback directly and converts row-major A/VS, WORK/BWORK, and SDIM |',
    '| GEESX | Native/saved Schur expert driver | SENSE, RCONDE/RCONDV, WORK/IWORK/RWORK, and row-major A/VS |',
    '| GGEV / GGEV3 | native GEVD driver | JOBVL/JOBVR、alpha/beta、row-major A/B/VL/VR、WORK/RWORK |',
    '| GGEVX | Native/saved GEVD expert driver | BALANC/SENSE, condition numbers, ILO/IHI, IWORK/BWORK, and row-major A/B/VL/VR |',
    '| GGHRD / GGHD3 | Native generalized Hessenberg reduction | COMPQ/COMPZ V/N, ILO/IHI, row-major A/B/Q/Z, and the GGHD3 work |',
    '| HGEQZ | native generalized Schur driver (JOB=E) | COMPQ/COMPZ、alpha/beta、row-major H/T/Q/Z、WORK/RWORK |',
    '| UPGTR / UPMTR | Native packed tridiagonal Q formation and application | C/Z only. Row-major packed AP reordering, SIDE/TRANS, and row-major C |',
    '| OPGTR / OPMTR | Native packed tridiagonal Q formation and application | S/D only. Row-major packed AP reordering, SIDE/TRANS, and row-major C |',
    '| PTTRF / PTTRS | Native positive tridiagonal factorization/solve | PTTRF without layout, row-major B for PTTRS, and real D with complex E for complex types |',
    '| PBTRF / PBTRS | Native banded positive-definite factorization/solve | Row-major band AB (kd+1 rows × LDAB columns), row-major B, and LDAB/LDB conversion |',
    '| SBTRD / HBTRD | Native banded tridiagonal reduction | Row-major band AB, Q for VECT=V, D/E, and LDAB/LDQ conversion |',
    '| PTEQR | Native tridiagonal eigensolver | COMPZ N/V/I, row-major Z, real WORK of length 4N-4, and D/E |',
    '| PPTRF / PPTRS | Native packed positive-definite factorization/solve | Row-major packed AP with pp_trans reordering, row-major B, and no LDAP |',
    '| GBTRF / GBTRS / GBSV | Native general band LU | Row-major band AB (2KL+KU+1 rows × LDAB columns), IPIV, row-major B, and TRANS |',
    '| GBEQU / GBEQUB | Native band equilibration | Analysis band AB (KL+KU+1 rows), R/C/ROWCND/COLCND/AMAX, and row-major conversion |',
    '| GTTRF / GTTRS / GTSV | Native tridiagonal LU | GTTRF without layout, DU2/IPIV, TRANS, and row-major B |',
    '| SPTRF / SPTRS / SPSV | Native packed Bunch-Kaufman LU | S/D only. Row-major packed AP with sp_trans reordering, IPIV, and row-major B |',
    '| SYTRF/SYTRS/SYSV and HETRF/HETRS/HESV | Native dense Bunch-Kaufman LU | UPLO, IPIV, WORK query, and row-major A/B. SY uses S/D and HE uses C/Z |',
    '| GEBAL / GEBAK | native balancing and back-transform | JOB N/P/S/B、ILO/IHI、real SCALE、row-major A/V、SIDE L/R |',
    '| TBTRS and TPTRS/TPTRI | Native triangular band/packed solve and inverse | UPLO/TRANS/DIAG, row-major band AB and packed AP reordering, and row-major B |',
    '| LANGB / LANHE | Native band and Hermitian norms | Row-major conversion of the analysis band AB (KL+KU+1 rows) and of the selected triangle. Row-major returns the same values as column-major |',
    '| GEHRD / GEBRD / LARFX | Native Hessenberg reduction, bidiagonalization, and reflector application | ILO/IHI, D/E/TAUQ/TAUP, work queries, row-major A/C, and the unblocked LARFX path for small dimensions |',
    '| GEEQU / GEEQUB | Native dense equilibration | R/C/ROWCND/COLCND/AMAX and row-major logical-rectangle conversion of A. GEEQUB scales by powers of two |',
    '| GGES / GGES3 | Native generalized Schur driver | JOBVSL/JOBVR/SORT, alpha/beta, row-major A/B/VSL/VSR, and the GGES3 work query |',
    '| HSEQR | native Hessenberg Schur decomposition | JOB E/S、COMPZ N/I/V、ILO/IHI、WR/WI vs W、row-major H/Z |',
    '| GEQR/GELQ and GEMQR/GEMLQ | Native blocked QR/LQ interfaces | T/TSIZE descriptors, row-major A (R×K and K×R), row-major C, and work queries |',
    '| PBEQU / PBSTF / PBSV | Native positive-definite band extras | S/SCOND/AMAX, the split Cholesky factor, row-major band AB (KD+1 rows), and B |',
    '| PFTRF / PFTRI / PFTRS | Native RFP Cholesky | TRANSR/UPLO, transposition of the RFP rectangle (ROW×COL), and row-major B |',
    '| TRTTF/TFTTR, TFTTP/TPTTF, TPTTR/TRTTP | Native full/RFP/packed conversion | Transposition of the RFP rectangle (dtf_trans), packed pp_trans reordering, and row-major A/LDA |',
    '| BDSQR, BDSDC, BDSVDX | Native bidiagonal SVD | Transposes and writes back VT/U/C; BDSDC validates COMPQ=N/I (COMPQ=P is only passed to the native compact fallback); BDSVDX writes back only the columns returned in the 2N-row [U;V] Z |',
    '| GETSLS, GETSQRHRT, ORGTSQR_ROW/UNGTSQR_ROW, ORHR_COL/UNHR_COL | Native tall-skinny least squares and TSQR Householder reconstruction | T is an NB×(N*number of slots) array and the row-major LDT must be at least the number of columns. GETSLS/GETSQRHRT/ORGTSQR_ROW query WORK/LWORK and transpose A/B/T before writing back |',
    '| LAG2C/LAG2D/LAG2S/LAG2Z, LACRM/LARCM, PSTRF | Native precision conversion, complex×real matrix products, and pivoted Cholesky | Row-major conversion and write-back. LAG2* overflow reports a positive INFO, LACRM/LARCM have no INFO (column-major argument errors come only from the native checker), and PSTRF writes back only the triangle and passes through PIV/RANK |',
    '| SYTRD/HETRD and SYEQUB/HEEQUB | Native dense tridiagonalization and diagonal scaling | NaN checks and write-back only on the UPLO triangle; SYTRD/HETRD query WORK/LWORK; SYEQUB/HEEQUB treat A as read-only and use CABS1 for complex types |',
    '| STEVX/STEVR/STEGR/STEMR | Native selected tridiagonal eigenvalues | NaN checks only for D/E (N elements for E in STEMR); row-major Z only for returned columns; STEVR/STEGR/STEMR query WORK/IWORK; STEMR passes NZC/TRYRAC |',
    '| **Not connected (no independent native public API): GEJSV/GESVJ** | None | The S/D/C/Z GEJSV and GESVJ (Jacobi SVD) have no numerical entry point on the colmat-rs/Colmat.jl side. Ordinary SVD is not substituted to make them succeed, and they are not registered with LBT (the standard LAPACK is left in charge). |',
    '| **Not connected (different contract): complex symmetric CSY*/CSP*** and ZSY*/ZSP*** (SYTRF/SYTRS/SYSV/SYTRI/SYTRI2/SYTRI2X/SYTRS2, SPTRF/SPTRS/SPSV/SPTRI, and their ROOK variants) | None | Native Bunch-Kaufman has a Hermitian LDL^H contract, so the factors and IPIV of complex symmetric LDL^T are different. The Hermitian version is not substituted, and these are not registered with LBT (the standard LAPACK is left in charge). |',
    '| **Unverified modes** | — | BDSDC COMPQ=P (compact form) only forwards to the native compact fallback. The row-major paths of BDSQR/BDSDC/BDSVDX verify COMPQ=N/I and all modes of BDSQR/BDSVDX. |',
    '| SYEVX/HEEVX, SYGVX/HEGVX, SBGVX/HBGVX, SPGVX/HPGVX | Native selected eigenvalues (dense, band, and packed, including generalized) | NaN checks and write-back only on the UPLO triangle; row-major Z only for returned columns; SYEVX/SYGVX query WORK/LWORK; the band Q is transposed only for JOBZ=V |',
    '| SBEVX/HBEVX, SPEVX/HPEVX, SPTRD/HPTRD | Native band/packed selected eigenvalues and packed tridiagonalization | Converts band AB and packed AP to column-major and writes back; row-major Q/Z only for JOBZ=V, and Z writes back only returned columns. WORK/IWORK(/RWORK) are allocated at fixed length |',
    '| SBGST/HBGST/SPGST/HPGST, SBGV/HBGV/SPGV/HPGV, SBGVD/HBGVD/SPGVD/HPGVD | Native band/packed generalized self-adjoint | Converts band AB/BB (KA+1/KB+1 rows) and packed AP/BP to column-major and writes back, transposing row-major Z/X. GST reads the split or ordinary Cholesky factor of B read-only. GVD variants query WORK/IWORK(/RWORK) |',
    "| SYGST/HEGST, SYGV/HEGV, SYGVD/HEGVD | Native generalized self-adjoint definite problems | ITYPE 1–3. NaN checks only on the UPLO triangle; row-major A is the full matrix for JOBZ=V and triangular otherwise; B's triangle is written back. GST reads B read-only. GVD variants query WORK/IWORK(/RWORK) |",
    '| TGEXC | Native reordering of a generalized Schur pair (QZ reorder) | Real IFST/ILST are input/output pointers, while complex ones are passed by value. Transposes A/B/Q/Z and queries the native routine directly |',
    '| SYTRF/SYTRS/SYSV_ROOK and HETRF/HETRS_ROOK | Native rook Bunch-Kaufman | Factors and IPIV (negative 2×2 pivots) are used interchangeably in row/column-major. Complex symmetric CSYTRF_ROOK and HESV_ROOK are out of scope |',
    '| SFRK/HFRK | Native RFP rank-k update | alpha/beta are passed by value, and the RFP rectangle (ROW×COL) and row-major A are transposed. No INFO is returned, so column-major argument errors come only from the native checker |',
    '| SBEV/HBEV/SPEV/HPEV and SBEVD/HBEVD/SPEVD/HPEVD | Native band/packed full eigen-decomposition | Converts band AB (KD+1 rows) and packed AP to column-major and transposes row-major Z. EVD variants allocate through WORK/IWORK(/RWORK) queries |',
    '| STERF/STEQR/STEV/STEVD/STEDC | Native tridiagonal EVD (QL/QR, divide and conquer) | STERF has no matrix_layout. Row-major Z is read as input only for COMPZ=V, and STEDC/STEVD allocate through WORK/IWORK(/RWORK) queries |',
    '| SPTRI, HPTRF/HPTRS/HPSV/HPTRI, SYTRS2/HETRS2 | Native packed Bunch-Kaufman and dense solves | Reorders packed AP with pp_trans and excludes the imaginary part of the Hermitian diagonal from NaN checks. SYTRS2 allocates N elements without using WORK |',
    '| SYTRI/HETRI, SYTRI2/HETRI2, SYTRI2X/HETRI2X | Native Bunch-Kaufman inverse | Factors and IPIV are used interchangeably. The row-major path checks NaN and writes back only the UPLO triangle; TRI2 queries WORK; TRI2X uses a work area of (N+NB+1)*(NB+3) |',
    '| POEQU/POEQUB, PPEQU, PPSV/PPTRI | Native positive-definite equilibration and packed extras | S/SCOND/AMAX (dense/packed), factor write-back for PPSV, and factor to inv(A) for PPTRI |',
    '| TFSM / TFTRI and PTSV | Native RFP triangular operations and positive-definite tridiagonal solves | SIDE/UPLO/TRANS/DIAG, row-major RFP rectangle transposition and B, and D/E with row-major B for PTSV |',
    '| TGEXC / TGEX2 / TGSEN / GGESX | Public QZ reorder and eigenvalue extraction from saved factors | Condition and separation estimates use the existing ILP64 LAPACK. Converts integer arrays and callback LOGICAL, and allocates internal matrices |',
    '| SBTRD/HBTRD/SPTRD/HPTRD and OPGTR/UPGTR/OPMTR/UPMTR | Native tridiagonalization / Householder | Band/packed expansion, external AP/TAU interoperability, and the X*Q update for VECT=U. Not a dedicated band bulge chase |',
    '| GST/GV/GVD/GVX for SBG/HBG/SPG/HPG | Native triangular/split Cholesky, congruence transformation, and full EVD | Converts packed ITYPE and the band split factor. The GVX Q is the full generalized eigenbasis, and positive ABSTOL and boundary guards use the saved original LAPACK |',
    "| SYEVR / HEEVR | Full EVD and A/I/V range selection | Does not provide native MRRR / Sturm count |','",
    "## Full list of connection candidates and unsupported routines','",
    'The following is grouped by family. For each row, all applicable symbols are listed explicitly. Types are kept in the TSV as `s/d/c/z`, and everything else as `other`.',
    'Evidence links point to the public source at the pinned revision. Restrictions within a row do not mean that they have been verified to satisfy every LAPACK contract.','']
grouped=defaultdict(list)
for row in rows: grouped[(row['colmat_api'],row['family'],row['entry'],row['source'],row['contract'])].append(row)
for status in LABELS.values():
    lines += ['### '+status,'','| Family / ABI | All symbols | Public feature and evidence | Contract differences |','|---|---|---|---|']
    for (classification,family,entry,source,note),items in sorted(grouped.items()):
        if classification!=status:continue
        symbols=', '.join('`'+r['symbol']+'`' for r in items)
        abi='/'.join(sorted({r['interface'] for r in items}))
        url=f"https://github.com/tensor4all/colmat-rs/blob/{UPSTREAM['revision']}/{source}"
        lines.append(f'| `{family}` / {abi} | {symbols} | [{entry}]({url}) | {note} |')
    lines += ['']
lines += ['## Direct ABI execution evidence (#317, #271)','',
    f"`direct_call_witness` covers {len(executed):,} canonical entries, with LP64 and ILP64 recorded separately.",
    'The [ABI evidence snapshot](../assets/lapack-abi-execution.json) identifies the exact source/native revisions, manifest and test source SHA-256 hashes. Stale evidence is rejected.',
    'CBLAS records completed named DLL/LBT calls explicitly: enums stay Cint, dimensions/strides use the selected integer ABI, and ROTG has no integer arguments.',
    'LARTG/LARTGP/LARTGS have no integer arguments; their entry names still distinguish the two LBT tables. Shared native addresses do not identify individual aliases; named LBT stubs provide that evidence.',
    "The collector observes project-owned pointer ccalls. High-level standard-library calls, isolated child processes and static symbol/library ccalls are excluded. Consult the individual source/signature/route witnesses and the numerical tests together.','",
    '```sh',
    "julia --project -e 'using Pkg; Pkg.test(; allow_reresolve=false)'",
    'python3 docs/import_abi_execution.py test-output/abi-execution.toml',
    "```','",
    'Linux/macOS CI collects the same evidence during the ordinary full suite and saves separate ABI artifacts.','']
lines += ['## Original candidate set and the 51 re-reviews','',
    'Retains the history of the 2,910 candidates and 51 items from the original #262 snapshot (`3c850a88a05b3c09eadd84d3722b1704cf1421f1` / native `e1c83482f5a61449d30e989808c3ba5f751ec391`).',
    '[The all-canonical per-owner re-review TSV](../assets/lbt-candidate-coverage.tsv) and [the JSON of audit conditions and original Issue body hashes](../assets/lbt-candidate-audit.json) are recorded separately.',
    'Fortran/CBLAS are mapped to the original explicit lists, and LAPACKE to the shared ABI scope of #288 and the numerical Issues of the Fortran counterparts. The numerical counterparts of the 148 LAPACKE candidates belong to #262, which was connected at the baseline. The ownership of the shared ABI and the registration of numerical counterparts are not proof that all contracts are implemented or verified.',
    'Adds the 40 candidates (20 Fortran and 20 LAPACKE) that overlooked public rook pivoting, tracked in [Issue #372](https://github.com/AtelierArith/Colmat.jl/issues/372). This is a classification correction because public enums/fields also exist in the original baseline, not the addition of a new native API. With the correction, the original candidate set contains 2,950 entries.',
    "No symbols were removed from the original positive API determinations. Of the old unsupported classifications, 1,918 were separated as unverified and are not dropped from the candidates as absent independent public APIs. They are not judged to be numerically implemented or confirmed as additional features.','",
    "### Separate columns for classification and evidence','",
    "| Column | Meaning |','|---|---|",
    '| `capability_status` / `colmat_api` | Classifies semantic correspondence of public numerical features, storage format, partial support, and unverified status |',
    '| `capability_evidence` / `evidence_status` | Verified revision, types, storage, options, verification method, public source, and reason. Absence determinations with insufficient records are not adopted |',
    '| `adapter_lp64` / `adapter_ilp64` | Measured current LBT registrations. Separate from whether a public API exists |',
    '| `execution_evidence_status` / `execution_evidence` | `direct_call_witness` records a completed concrete call through a DLL or LBT entry. It does not certify every option, numerical contract or independent algorithm. `not_catalogued` does not mean untested. |',
    '| `independent_algorithm_status` | Verification status of a dedicated independent algorithm. Currently `not_assessed`. Do not infer a dedicated algorithm from the existence of a basic solver or API |','']
lines += ['## Regeneration and completeness checks','',
    "The public API assessments are kept in `docs/lbt-capabilities.json`, and the actual symbol set is obtained from LBT. Routines with no registered rule and old unsupported determinations with insufficient evidence are retained as \"Unverified; insufficient evidence\". Suffixes/prefixes are not evidence of absence.','",
    "```sh','python3 docs/generate_lbt_inventory.py','julia --project=test test/lbt_inventory.jl','python3 -m unittest discover -s docs/tests -v','```','",
    "To regenerate the candidate re-review, pass the TSV of the pinned snapshot (`git show 3c850a88a05b3c09eadd84d3722b1704cf1421f1:docs/src/assets/lbt-symbol-inventory.tsv`) and a JSON containing the original 51 Issue bodies to `python3 docs/audit_lbt_candidates.py --baseline-tsv ORIGINAL.tsv --issue-snapshot docs/audits/issue-263-scope-snapshot.json --rook-issue 372`. The output records body hashes and retrieval times, and does not confuse past bodies with the current snapshot.','",
    'This audit is a public API mapping of a pinned source. If public features are added later, re-read the classification rules and restrictions and update it.']
(ROOT/'docs/src/reference/lbt-inventory.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(meta,ensure_ascii=False))
