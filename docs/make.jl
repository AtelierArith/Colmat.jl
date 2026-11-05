using Documenter
using DocumenterVitepress

# The manual contains ordinary Julia code fences, not executed examples. Building
# it does not load Colmat or compile/update the native Rust dependency.
makedocs(;
    root = @__DIR__,
    sitename = "Colmat.jl",
    authors = "Satoshi Terasaki",
    format = DocumenterVitepress.MarkdownVitepress(;
        repo = "github.com/AtelierArith/Colmat.jl",
        devbranch = "main",
        devurl = "dev",
    ),
    pages = [
        "ホーム" => "index.md",
        "インストール" => "installation.md",
        "マニュアル" => [
            "配列・型・所有権" => "manual/arrays.md",
            "密行列の分解と求解" => "manual/dense.md",
            "固有値・Schur・QZ" => "manual/spectral.md",
            "疎行列と直接法" => "manual/sparse.md",
            "疎 QR" => "manual/sparse-qr.md",
            "線形作用素・反復法" => "manual/operators.md",
            "反射・回転・縮約" => "manual/reductions.md",
            "作業領域とチーム" => "manual/resources.md",
            "統計・乱数・NPY" => "manual/utilities.md",
        ],
        "API 案内" => "reference/api.md",
        "開発ガイド" => "development.md",
    ],
)

if "--deploy" in ARGS
    DocumenterVitepress.deploydocs(;
        repo = "github.com/AtelierArith/Colmat.jl",
        target = joinpath(@__DIR__, "build"),
        devbranch = "main",
        push_preview = false,
    )
end
