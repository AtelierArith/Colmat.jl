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
    'native': '対応する公開数値機能あり',
    'representation': '対応機能あり・保存形式の変換が必要',
    'composition': '公開機能の合成候補・直接 API はなし',
    'conditional': '一部機能のみ・追加の契約実装が必要',
    'unsupported': '対応する独立公開 API なし（確認範囲付き）',
    'unverified': '未照合・根拠不足',
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
        note += '; C ABI/row-major/enum/戻り値を別途変換する必要がある'
    if family == 'geevx' and lp == ilp == 'true':
        note += '; 現行 native は SENSE=N。条件数推定は保存した元の backend へ委譲。LP64 の元の実装がなければ ILP64 に整数を変換。元の ILP64 GEEVX が不足する構成は LBT 登録前に拒否'
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
lines = ['# LBT と colmat-rs の全ルーチン対応表','',
    '## 調査範囲と判定','',
    f"Julia {meta['julia']} の LBT が公開する **{len(rows):,} 個の全 canonical シンボル**を列挙します。",
    'LP64 / ILP64 は同じ canonical 名の別テーブルです。型と ABI を省略して「全部対応」とは判定しません。',
    f"colmat-rs の対象は `deps/upstream.toml` の `{UPSTREAM['revision']}` です。",
    'これは公開ソースに基づく接続候補の静的監査です。以下の全候補を実行した検証結果ではありません。',
    '実際の Julia 呼び出しの結果は [Julia API の実測対応表](@ref)に分けて記録します。','',
    '- **対応する公開数値機能あり**: LU・SVD・固有値分解などの数値 kernel が公開されている。ABI 同一という意味ではない。',
    '- **保存形式の変換が必要**: reflector・置換・band/packed 等の差がある。まだ adapter があるとは限らない。',
    '- **合成候補**: 公開 kernel を組み合わせれば数学的出力を作れる候補。LAPACK 固有の算法・精度保証・計算量まで同等とは判定しない。',
    '- **一部機能のみ**: 基本分解はあるが condition estimate・固定 pivot・expert flag 等が不足する。',
    '- **対応する独立公開 API なし（確認範囲付き）**: revision・型・保存形式・オプション・意味上の確認方法を記録した範囲での判定。数学的に自作できないという意味ではない。',
    '- **未照合・根拠不足**: 未登録 family、名前に基づく推測、確認記録が不足する旧非対応分類。公開 API の不存在を意味しない。',
    '- `capability_status` と `capability_evidence` は公開機能の判定、`adapter_lp64/ilp64` は実際の ABI 接続、`execution_evidence_status` は入口ごとの実行証拠の登録状態、`independent_algorithm_status` は独立算法の確認を別々に記録する。`not_catalogued` は未実行という意味ではない。`not_assessed` から独立算法の有無を推論しない。',
    '- `adapter` は実際の LBT 登録内容。数値機能の存在、ABI の接続、実行済み検証を混同しない。','',
    '## 全シンボルの機械可読一覧','',
    '[全シンボル TSV](../assets/lbt-symbol-inventory.tsv) は **1シンボル1行**で、名称・C/Fortran ABI・型・公開機能判定・LP64/ILP64 接続状態・ソース根拠・制約を収録します。',
    '[生成条件 JSON](../assets/lbt-symbol-inventory.json) も保存します。','',
    '| ABI | canonical シンボル数 |','|---|---|']
for interface,count in meta['interfaces'].items(): lines.append(f'| {interface} | {count:,} |')
lines += ['',f"現行 adapter は LP64 **{meta['lp64']}**、ILP64 **{meta['ilp64']}** シンボルを登録します。",
    'GEEVX の条件数推定は native ではなく既存 backend へ委譲します。LAPACKE 全体を提供する変更ではありません。','',
    '## 今回接続した LAPACK','',
    '| 系列 | 数値処理 | 制約 |','|---|---|---|',
    '| GEQRT / GEQRT2 / GEQRT3 / GEMQRT | native QR の係数変換とブロック反射適用 | 内部配列を確保。GEQRT3 は共通経路を使用 |',
    '| GETRF / GETRF2 / GETF2 / GETRS / GETRI / GESV | colmat-rs LU・三角 solve | 最終置換を IPIV に変換 |',
    '| POTRF / POTRF2 / POTF2 / POTRS / POTRI / POSV | colmat-rs LLT・三角 solve | UPLO と INFO を変換 |',
    '| GESDD / GESVD | colmat-rs SVD | full/thin/overwrite と VT を変換。両 driver は共通 native SVD に接続 |',
    '| GEEV / GEEVX | colmat-rs balance・EVD・Schur | GEEVX は SENSE=N が native。expert は元の実装、LP64 は必要に応じ ILP64 へ整数変換。ILP64 実装を登録前に要求 |',
    '| GGEV / GGEV3 / GEGV / GGEVX | colmat-rs GEVD・homogeneous α/β・左右固有ベクトル | GGEVX は BALANC=N/SENSE=N が native。その他は既存 ILP64 backend と整数 ABI 変換 |',
    '| SYEV / HEEV / SYEVD / HEEVD | colmat-rs 自己随伴 EVD | ドライバー名と同じ Netlib 算法を実行する保証はない |',
    '| LARFG / LARFGP / LARF / LARFX / LARFY / LARFT / LARFB | Rust reflector 生成・native GEMV/SYMV/GEMM | LAPACK 係数・F/B・C/R を変換。LARFB は内部配列を確保 |',
    '| GGQRF / GGRQF / GGLSE / GGGLM | native-backed QR/RQ・Q 適用・三角 solve の合成 | 外部 reflector 互換、等式制約と最小ノルム、厳密な rank failure の INFO=1/2。内部係数・scratch を確保 |',
    '| TREXC / LAEXC / TRSEN / GEESX | native checked Schur moves・既存 GEES | 条件推定は保存した元の ILP64 routine。拒否時の index と全選択 SDIM、整数作業配列と callback の ABI を変換 |',
    '| TREVC / TREVC3 | public Schur eigenvectors・反転随伴・back-transform | SIDE L/R/B・HOWMNY A/B/S・実数対の SELECT/M と max-component 正規化。native 拒否は保存した元の LAPACK |',
    '| LACPY / LASET / LASWP | native LACPY/LASET/LASWP と直接 row-major 変換 | 論理矩形・選択三角だけを読み書き。LASWP は行全体を交換し、INCX<0 の Fortran 順序を保持 |',
    '| TGEXC / TGEX2 / TGSEN / GGESX | public QZ reorder・保存因子の固有値抽出 | 条件・分離推定は既存 ILP64 LAPACK。整数配列・callback LOGICAL を変換し、内部行列を確保 |',
    '| SBTRD/HBTRD/SPTRD/HPTRD・OPGTR/UPGTR/OPMTR/UPMTR | native tridiag / Householder | band/packed 展開、外部 AP/TAU 相互利用、VECT=U の X*Q 更新。専用 band bulge chase ではない |',
    '| SBG/HBG/SPG/HPG の GST/GV/GVD/GVX | native triangular/split Cholesky・合同変換・全 EVD | packed ITYPE と band split 因子を変換。GVX の Q は全一般化固有基底、正の ABSTOL と境界 guard は保存した元の LAPACK |',
    '| SYEVR / HEEVR | 全 EVD と A/I/V 範囲の選択 | native MRRR / Sturm count は提供しない |','',
    '## 接続候補と未対応ルーチンの全一覧','',
    '以下は系列ごとの集約です。各行に該当する全シンボルを明示します。型は `s/d/c/z`、それ以外は `other` として TSV に保持します。',
    '根拠リンクは固定 revision の公開ソースです。行内の制約が LAPACK の全契約を満たすと検証済み、という意味ではありません。','']
grouped=defaultdict(list)
for row in rows: grouped[(row['colmat_api'],row['family'],row['entry'],row['source'],row['contract'])].append(row)
for status in LABELS.values():
    lines += ['### '+status,'','| 系列 / ABI | 全シンボル | 公開機能・根拠 | 契約上の差 |','|---|---|---|---|']
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
    'The collector observes project-owned pointer ccalls. High-level standard-library calls, isolated child processes and static symbol/library ccalls are excluded. Consult the individual source/signature/route witnesses and the numerical tests together.','',
    '```sh',
    "julia --project -e 'using Pkg; Pkg.test(; allow_reresolve=false)'",
    'python3 docs/import_abi_execution.py test-output/abi-execution.toml',
    '```','',
    'Linux/macOS CI collects the same evidence during the ordinary full suite and saves separate ABI artifacts.','']
lines += ['## 元の候補集合と51件の再照合','',
    '元の #262 snapshot (`3c850a88a05b3c09eadd84d3722b1704cf1421f1` / native `e1c83482f5a61449d30e989808c3ba5f751ec391`) の2,910候補・51件の履歴を保持します。',
    '[全 canonical の担当再照合 TSV](../assets/lbt-candidate-coverage.tsv) と [監査条件・元の Issue 本文 hash JSON](../assets/lbt-candidate-audit.json) を別に記録します。',
    'Fortran/CBLAS は元の明示リスト、LAPACKE は #288 の共通 ABI 範囲と Fortran counterpart の数値 Issue に対応付けます。148件の LAPACKE 候補の数値 counterpart は基準時点で接続済みの #262 に帰属します。共通 ABI の担当と、数値 counterpart の登録は、全契約の実装・検証の証明ではありません。',
    '公開 rook pivoting を見落としていた40候補（Fortran 20・LAPACKE 20）を追加し、[Issue #372](https://github.com/AtelierArith/Colmat.jl/issues/372) で追跡します。元の基準にも公開 enum/field が存在する分類訂正であり、新たな native API の追加ではありません。旧基準の候補集合は訂正により2,950になります。',
    '元の肯定的な API 判定から除外したシンボルは0です。旧非対応分類のうち1,918件は未照合へ分離し、独立公開 API の不存在として候補から落としません。これらを数値実装済み・追加機能確認済みとは判定しません。','',
    '### 判定と証拠の別列','',
    '| 列 | 意味 |','|---|---|',
    '| `capability_status` / `colmat_api` | 公開数値機能の意味上の対応・保存形式・部分対応・未照合を分類 |',
    '| `capability_evidence` / `evidence_status` | 確認 revision・型・格納・オプション・確認方法・公開ソース・理由。記録不足の不存在判定は採用しない |',
    '| `adapter_lp64` / `adapter_ilp64` | 現在の LBT 登録内容を実測。公開 API の有無とは別 |',
    '| `execution_evidence_status` / `execution_evidence` | `direct_call_witness` records a completed concrete call through a DLL or LBT entry. It does not certify every option, numerical contract or independent algorithm. `not_catalogued` does not mean untested. |',
    '| `independent_algorithm_status` | 専用の独立算法の確認状態。現在は `not_assessed`。基本 solver や API の存在から専用算法を推論しない |','']
lines += ['## 再生成と網羅性検査','',
    '公開 API の判断は `docs/lbt-capabilities.json` に保持し、実際のシンボル集合は LBT から取得します。規則未登録のルーチンと根拠が不足する旧非対応判定は「未照合・根拠不足」として保持します。suffix/prefix は不存在の証拠にしません。','',
    '```sh','python3 docs/generate_lbt_inventory.py','julia --project=test test/lbt_inventory.jl','python3 -m unittest discover -s docs/tests -v','```','',
    '候補再照合の再生成には、固定 snapshot の TSV（`git show 3c850a88a05b3c09eadd84d3722b1704cf1421f1:docs/src/assets/lbt-symbol-inventory.tsv`）と元の51件の Issue 本文を保存した JSON を `python3 docs/audit_lbt_candidates.py --baseline-tsv ORIGINAL.tsv --issue-snapshot docs/audits/issue-263-scope-snapshot.json --rook-issue 372` に渡します。出力は本文 hash と取得時点を記録し、過去の本文と現行 snapshot を混同しません。','',
    'この監査は固定ソースの公開 API 対応表です。公開機能が後から追加された場合は分類規則と制約を読み直して更新してください。']
(ROOT/'docs/src/reference/lbt-inventory.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(meta,ensure_ascii=False))
