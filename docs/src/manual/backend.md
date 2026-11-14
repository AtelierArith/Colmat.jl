# バックエンドとフォールバック

## 呼び出しの流れ

```text
Julia Matrix / Vector と LinearAlgebra
                  ↓
           libblastrampoline
                  ↓
        libcolmat_blas の BLAS ABI
                  ↓
       Rust colmat / Rust 互換層
```

`Colmat.__init__()` は `Colmat.load_backend()` を実行します。
ライブラリの存在と互換層の ABI バージョンを確認し、ILP64、LP64 の順に
`BLAS.lbt_forward` で登録します。

Julia の配列は呼び出し中だけ Rust が借用します。Julia 側に Rust の行列所有権や
寿命管理用ハンドルを公開しません。`mul!` などの変更規則は Julia の API に従います。

## 未提供ルーチンの扱い

登録には `clear=false` を指定します。Colmat が提供するシンボルを置き換え、
未提供の BLAS / LAPACK の転送先は保持します。

| 呼び出し | 実行先 |
|---|---|
| 対応する型の GEMM / GEMV など | Colmat |
| 帯・packed 格納や未提供 BLAS | 既存の転送先 |
| LAPACK の LU / SVD / 一般・自己随伴固有値 / Cholesky | Colmat |
| QR / Schur / 一般化固有値など、まだ未接続の LAPACK | 既存の転送先 |
| 整数や BigFloat など、この BLAS の対象外の型 | Julia の型ごとの既存実装 |

フォールバック側の共有ライブラリ内部から呼ぶ BLAS が LBT を経由するかは、
そのライブラリのリンク構成に依存します。LBT に入口が登録されたことと、
他の未接続分解の内部計算まで Colmat に切り替わることは同じではありません。
疎行列ソルバーの選択も、このパッケージのロードだけでは置き換えません。

## 切り替えの範囲と順序

変更は Julia プロセス全体に適用され、LBT を経由する他パッケージにも影響します。
計算ごと、配列ごと、Julia Task ごとの設定ではありません。

別の BLAS プロバイダーを後からロードすると、そのプロバイダーが転送先を変更する
場合があります。Colmat を再選択するときは次を実行します。

```julia
Colmat.load_backend()
```

この呼び出しも未提供ルーチンの現在の転送先を保持します。最初の OpenBLAS 構成へ
戻す操作や、フォールバックを特定ライブラリに固定する操作ではありません。

バックエンドの切り替えは並列 BLAS 計算を開始する前に済ませてください。
Colmat 内部の設定ロックは、他パッケージや直接の LBT 操作を同期しません。


## LAPACK の変換と未対応オプション

LU の最終置換を LAPACK の IPIV に、SVD の V を VT に変換します。
workspace query は LAPACK の `lwork=-1` 等を受け付けますが、native の作業領域は
現在 adapter 内部で確保します。LAPACK の work 配列だけで割り当てなしに動く
実装ではありません。

一般 EVD は native balance、左右固有ベクトル計算、Schur 計算を使用します。
現在は返却する Schur 形式を得るための処理を追加実行するため、性能上の制約があります。

GEEVX の `SENSE=N` は native、`SENSE=E/V/B` の条件数推定は登録前の転送先を呼びます。
その整数 ABI に既存 GEEVX の転送先がなければ、expert モードは利用できず負の INFO
で戻ります。`load_backend()` を再実行しても Colmat 自身を fallback として保存しません。

SYEVR/HEEVR は全スペクトルを計算して `RANGE=A/I/V` の結果を選択します。
colmat-rs に MRRR や Sturm-count の公開 API があるという意味ではありません。
ドライバーごとの数値実装・保存形式・未対応機能は
[LBT と colmat-rs の全ルーチン対応表](@ref)で確認できます。
