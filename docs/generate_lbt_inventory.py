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

ROOT = Path(__file__).resolve().parent.parent
UPSTREAM = tomllib.loads((ROOT / 'deps/upstream.toml').read_text())
SOURCE = ROOT / 'deps/colmat-rs'
POLICY = json.loads((ROOT / 'docs/lbt-capabilities.json').read_text())
GROUPS = {}
for group in POLICY['groups']:
    for name in group['routines']:
        GROUPS[name] = group

LABELS = {
    'native': '対応する公開数値機能あり',
    'representation': '対応機能あり・保存形式の変換が必要',
    'composition': '公開機能の合成候補・直接 API はなし',
    'conditional': '一部機能のみ・追加の契約実装が必要',
    'unsupported': '対応する独立公開 API なし',
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
    group = GROUPS.get(family)
    if group is not None:
        group = dict(group)
        if scalar in 'cz' and family.startswith(('sy','sp')) and family in {
            'sytrf','sytf2','sytrs','sytri','sysv','sytri2','sytri2x','sytrs2','sygv','sygvd','sygvx','syev','syevd','syevr','syevx','sytrd','sytd2','sptrf','sptrs','sptri','spsv'
        }:
            group.update(status='unsupported', api='—', note='複素対称（非 Hermitian）のこの契約を満たす公開 solver はない。複素 Hermitian HE 系と区別する')
        return group
    # The public API catalogue has no independently callable analogue for these
    # auxiliaries/algorithms. No inference from "SVD internally uses it" is made.
    if family.endswith('_2stage') or '_aa' in family or family.endswith(('_rk','_rook','_3','_3x')):
        note = '特定の two-stage / Aasen / rook / rank-k 保存方式の公開 API はない。標準 driver の数学的存在と区別する'
    elif family.startswith(('la','il','iparm','xerbla','lsame','orbdb','unbdb','tp','tz')):
        note = 'LAPACK 固有の補助処理・内部アルゴリズムまたは保存形式に対応する独立公開 API は確認できない'
    elif family.endswith(('_trans','_nancheck')) or family == 'nancheck':
        note = 'LAPACKE の入力検査・row/column-major 変換 helper は colmat の公開ルーチンとして提供されない。adapter 側の処理'
    else:
        note = '固定 revision の公開 API catalogue に同じ契約の独立ルーチンはない。数学的に自作・合成できないという意味ではない'
    return dict(status='unsupported', api='—', source='faer/src/linalg/mod.rs', note=note)


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
    note = group['note']
    if interface != 'Fortran':
        note += '; C ABI/row-major/enum/戻り値を別途変換する必要がある'
    if family == 'geevx' and lp == ilp == 'true':
        note += '; 現行 native は SENSE=N。条件数推定は登録時に保持した backend へ委譲。該当 ABI に既存転送先がない場合は利用不可'
    rows.append(dict(symbol=symbol, interface=interface, family=family, scalar=scalar,
        colmat_api=LABELS[group['status']], adapter_lp64=lp, adapter_ilp64=ilp,
        entry=group['api'], source=source, contract=note))
assert len(rows) == len({r['symbol'] for r in rows})
assets = ROOT / 'docs/src/assets'
assets.mkdir(exist_ok=True)
with (assets/'lbt-symbol-inventory.tsv').open('w') as out:
    writer = csv.DictWriter(out, fieldnames=list(rows[0]), delimiter='\t', lineterminator='\n')
    writer.writeheader(); writer.writerows(rows)
meta = dict(julia=subprocess.check_output(['julia','--startup-file=no','-e','print(VERSION)'],text=True),
    revision=UPSTREAM['revision'], repository=UPSTREAM['repository'], symbols=len(rows),
    interfaces=dict(Counter(r['interface'] for r in rows)),
    lp64=sum(r['adapter_lp64']=='true' for r in rows),
    ilp64=sum(r['adapter_ilp64']=='true' for r in rows))
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
    '- **対応する独立公開 API なし**: 同じ契約を持つ公開関数は提供されていない。内部 kernel の存在や自作可能性と区別する。',
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
    '| GEEV / GEEVX | colmat-rs balance・EVD・Schur | GEEVX は SENSE=N が native。その他は既存 backend |',
    '| GGEV / GGEV3 / GEGV / GGEVX | colmat-rs GEVD・homogeneous α/β・左右固有ベクトル | GGEVX は BALANC=N/SENSE=N が native。その他は既存 ILP64 backend と整数 ABI 変換 |',
    '| SYEV / HEEV / SYEVD / HEEVD | colmat-rs 自己随伴 EVD | ドライバー名と同じ Netlib 算法を実行する保証はない |',
    '| LARFG / LARFGP / LARF / LARFX / LARFY / LARFT / LARFB | Rust reflector 生成・native GEMV/SYMV/GEMM | LAPACK 係数・F/B・C/R を変換。LARFB は内部配列を確保 |',
    '| TGEXC / TGEX2 / TGSEN / GGESX | public QZ reorder・保存因子の固有値抽出 | 条件・分離推定は既存 ILP64 LAPACK。整数配列・callback LOGICAL を変換し、内部行列を確保 |',
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
lines += ['## 再生成と網羅性検査','',
    '公開 API の判断は `docs/lbt-capabilities.json` に保持し、実際のシンボル集合は LBT から取得します。分類にない補助ルーチンは「独立公開 API なし」として列挙し、黙って除外しません。','',
    '```sh','python3 docs/generate_lbt_inventory.py','julia --project=test test/lbt_inventory.jl','```','',
    'この監査は固定ソースの公開 API 対応表です。公開機能が後から追加された場合は分類規則と制約を読み直して更新してください。']
(ROOT/'docs/src/reference/lbt-inventory.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(meta,ensure_ascii=False))
