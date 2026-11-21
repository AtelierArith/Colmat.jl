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
        description="Colmat.jl manual for Julia users",
    ),
    modules=[Colmat],
    pages=[
        "Home" => "index.md",
        "Installation" => "installation.md",
        "Getting started" => "getting-started.md",
        "Usage and design" => [
            "Backends and fallbacks" => "manual/backend.md",
            "Thread settings" => "manual/threading.md",
        ],
        "Reference" => [
            "Full LBT routine mapping" => "reference/lbt-inventory.md",
            "BLAS coverage" => "reference/coverage.md",
            "Measured Julia API mapping" => "reference/support-matrix.md",
            "Public API" => "reference/api.md",
        ],
        "Troubleshooting" => "troubleshooting.md",
        "0.2 redesign" => "migration.md",
        "Development" => "development.md",
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
