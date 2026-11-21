# Documentation development

The current public manual starts at [docs/src/index.md](src/index.md).
It is built with Documenter.jl and DocumenterVitepress.jl.
Run the following from the repository root.

```sh
julia --project=docs -e 'using Pkg; Pkg.instantiate(; workspace=true); Pkg.build("Colmat")'
julia --project=docs docs/make.jl
python3 -m http.server 8000 --bind 127.0.0.1 --directory docs/build/1
```

Open `http://127.0.0.1:8000` in a browser.
Because building the documentation loads the public API, Colmat must be built natively.
The documentation environment is part of the root Julia workspace and uses the root Manifest.

Manage the content in `src/`, the page structure in `make.jl`, and the display settings
in `src/.vitepress/config.mts`. `jldoctest` code examples and public API docstrings are
verified during the documentation build.

- [Implementation status](implementation-status.md) points to the current manual.
- [API mapping](public-api-mapping.md) also refers to the same public manual.
- For the old API ledger and reproducers, see the Git history described in [the redesign](src/migration.md).

Deployment to GitHub Pages is managed by `.github/workflows/documentation.yml`.
The local `docs/make.jl` does not publish anything; it only runs deployment when `--deploy` is given.
