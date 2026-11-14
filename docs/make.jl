using Documenter
using DocumenterVitepress

using Colmat

makedocs(;
    root=@__DIR__,
    sitename="Colmat.jl",
    authors="Satoshi Terasaki",
    format=DocumenterVitepress.MarkdownVitepress(;
        repo="github.com/AtelierArith/Colmat.jl",
        devbranch="main",
        devurl="dev",
        description="Julia ユーザのための Colmat.jl 日本語マニュアル",
    ),
    modules=[Colmat],
    pages=[
        "ホーム" => "index.md",
        "インストール" => "installation.md",
        "はじめに" => "getting-started.md",
        "使い方と設計" => [
            "バックエンドとフォールバック" => "manual/backend.md",
            "スレッド設定" => "manual/threading.md",
        ],
        "リファレンス" =>
            ["BLAS 対応範囲" => "reference/coverage.md", "公開 API" => "reference/api.md"],
        "トラブルシューティング" => "troubleshooting.md",
        "0.2 への設計変更" => "migration.md",
        "開発" => "development.md",
    ],
)

if "--deploy" in ARGS
    DocumenterVitepress.deploydocs(;
        repo="github.com/AtelierArith/Colmat.jl",
        target=joinpath(@__DIR__, "build"),
        devbranch="main",
        push_preview=false,
    )
end
