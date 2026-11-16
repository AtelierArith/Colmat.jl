module ABIEvidence
using Libdl, Test, TOML, SHA
const source_hashes = Dict{String,String}()
const root = realpath(dirname(@__DIR__))
function repository_revision()
    return if ispath(joinpath(root, ".git"))
        strip(read(`git -C $root rev-parse HEAD`, String))
    else
        "source_archive"
    end
end
const initial_head = repository_revision()
function source_files()
    files=String[]
    for (directory, suffixes) in
        (("src", (".jl",)), ("test", (".jl", ".c")), ("deps/rust/src", (".rs",)))
        for (path, _, names) in walkdir(joinpath(root, directory)), name in names
            any(endswith(name, suffix) for suffix in suffixes) || continue
            push!(files, relpath(joinpath(path, name), root))
        end
    end
    append!(
        files,
        [
            "Project.toml",
            "test/Project.toml",
            "deps/upstream.toml",
            "deps/build.jl",
            "deps/rust/Cargo.toml",
            "deps/rust/Cargo.lock",
        ],
    )
    return sort!(unique(files))
end
for file in source_files()
    source_hashes[file]=bytes2hex(sha256(read(joinpath(root, file))))
end
const addresses = Dict{UInt,Vector{Tuple{String,Int,String}}}()
const witnesses = Set{Tuple{String,Int,String,String,Int,String,Bool}}()
const ambiguities = Set{String}()
const errors = String[]
const initialized = Ref(false)
const seen = Set{Tuple{UInt,Any,String,Int}}()
function initialize()
    native = Libdl.dlopen(Main.Colmat.backend_library())
    trampoline = Libdl.dlopen(Main.LinearAlgebra.BLAS.libblastrampoline)
    rows=readlines(joinpath(root, "docs/src/assets/lbt-symbol-inventory.tsv"))
    header=split(first(rows), '\t')
    for line in rows[2:end]
        row=Dict(zip(header, split(line, '\t'; keepempty=true)))
        row["interface"]=="Fortran" || continue
        for (bits, column) in ((32, "adapter_lp64"), (64, "adapter_ilp64"))
            row[column]=="true" || continue
            symbol=row["symbol"]*(bits==64 ? "64_" : "")
            for (handle, route) in ((native, "dll"), (trampoline, "lbt_trampoline"))
                pointer=Libdl.dlsym(handle, symbol)
                push!(
                    get!(addresses, UInt(pointer), Tuple{String,Int,String}[]),
                    (row["symbol"], bits, route),
                )
            end
        end
    end
    return initialized[]=true
end
function record(pointer, signature, file, line)
    try
        initialized[] || initialize()
        pointer isa Ptr || return nothing
        key=(UInt(pointer), signature, file, line)
        key in seen && return nothing
        push!(seen, key)
        entries=get(addresses, UInt(pointer), nothing)
        isnothing(entries) && return nothing
        widths=Set{Int}()
        for t in signature
            t <: Union{Ref{Int32},Ptr{Int32},Ref{UInt32},Ptr{UInt32}} && push!(widths, 32)
            t <: Union{Ref{Int64},Ptr{Int64},Ref{UInt64},Ptr{UInt64}} && push!(widths, 64)
        end
        filtered=isempty(widths) ? entries : filter(e->e[2] in widths, entries)
        length(unique((e[1], e[2]) for e in filtered))==1 || begin
            push!(
                ambiguities,
                string(relpath(file, root), ":", line, " ", entries, " widths=", widths),
            )
            return nothing
        end
        for (symbol, bits, route) in filtered
            push!(
                witnesses,
                (
                    symbol,
                    bits,
                    route,
                    relpath(file, root),
                    line,
                    string(signature),
                    !isempty(widths),
                ),
            )
        end
    catch e
        push!(errors, sprint(showerror, e))
    end
    return nothing
end
function rewrite(x, file, line=1)
    x isa Expr || return x
    x.head in (:inert,) && return x
    if x.head==:call && x.args[1]==:ccall
        original=x.args[2]
        # Static symbol/library syntax has special lowering and is not a pointer.
        (
            original isa QuoteNode ||
            original isa String ||
            (original isa Expr && original.head==:tuple)
        ) && return x
        p=gensym(:pointer)
        r=gensym(:result)
        call=Expr(:call, :ccall, p, x.args[3:end]...)
        sig=x.args[4]
        return quote
            let $p=$original
                local $r=$call
                $(GlobalRef(ABIEvidence, :record))($p, $sig, $file, $line)
                $r
            end
        end
    elseif x.head==:call && x.args[1]==:include && length(x.args)==2
        return :($(GlobalRef(ABIEvidence, :include_traced))(@__MODULE__, $(x.args[2])))
    end
    current=line
    newargs=Any[]
    for arg in x.args
        arg isa LineNumberNode && (current=arg.line)
        push!(newargs, rewrite(arg, file, current))
    end
    return Expr(x.head, newargs...)
end
function include_traced(mod, path)
    file=abspath(path)
    # Base.include keeps task-local include paths, but this wrapper must resolve
    # relative nested includes against the file currently being mapped.
    if !isabspath(path)
        parent=get(task_local_storage(), :abi_evidence_file, nothing)
        isnothing(parent) && (parent=joinpath(root, "test/runtests.jl"))
        file=normpath(joinpath(dirname(parent), path))
    end
    file=realpath(file)
    if !startswith(file, joinpath(root, "test")*"/")
        return Base.include(mod, file)
    end
    get!(source_hashes, relpath(file, root), bytes2hex(sha256(read(file))))
    old=get(task_local_storage(), :abi_evidence_file, nothing)
    task_local_storage(:abi_evidence_file, file)
    try
        Base.include(x->rewrite(x, file), mod, file)
    finally
        task_local_storage(:abi_evidence_file, old)
    end
end
# RFP SFRK/HFRK belong to LAPACK in the package's published counts.
const blas_families=Set(
    split(
        "scal real_scal copy swap axpy axpby dot dotu dotc asum nrm2 iamax iamin amax amin max min sum rot real_rot rotg gemv gbmv trmv tbmv tpmv trsv tbsv tpsv symv sbmv spmv hemv hbmv hpmv ger gerc geru syr spr syr2 spr2 her hpr her2 hpr2 gemm gemm3m gemmt gemmtr symm hemm syrk syr2k herk her2k trmm trsm imatcopy omatcopy geadd",
    ),
)
function inventory_rows()
    lines=readlines(joinpath(root, "docs/src/assets/lbt-symbol-inventory.tsv"))
    header=split(first(lines), '\t')
    return [Dict(zip(header, split(line, '\t'; keepempty=true))) for line in lines[2:end]]
end
function registration_digest()
    fields=("symbol", "interface", "family", "scalar", "adapter_lp64", "adapter_ilp64")
    rows=sort!(inventory_rows(); by=row->row["symbol"])
    data=join((join((row[field] for field in fields), '\t') for row in rows), '\n')*"\n"
    return bytes2hex(sha256(data))
end
function expected_lapack()
    return Set(
        row["symbol"] for row in inventory_rows() if row["interface"]=="Fortran" &&
            row["adapter_lp64"]=="true" &&
            !(row["family"] in blas_families)
    )
end
function finalize()
    expected=expected_lapack()
    cblas_sites=Main.CBLAS_271_CALL_SITES
    cblas_targets=Set(Main.CBLAS_271_TARGETS)
    @testset "Actual CBLAS foreign-call evidence" begin
        @test length(cblas_targets)==172
        for bits in (32, 64), route in (:dll, :lbt_trampoline)
            @test Set(s for (s, b, r, _, _, _) in cblas_sites if b==bits&&r==route)==cblas_targets
        end
    end
    @testset "Actual LAPACK foreign-call evidence" begin
        @test isempty(errors)
        @test length(expected)==1106
        for bits in (32, 64)
            executed=Set(e[1] for e in witnesses if e[2]==bits)
            missing=setdiff(expected, executed)
            isempty(missing) ||
                @error "LAPACK entries missing concrete calls" bits missing=sort!(
                    collect(missing)
                )
            @test isempty(missing)
        end
        @test repository_revision()==initial_head
        @test initial_head=="source_archive" ||
            isempty(read(`git -C $root status --porcelain --untracked-files=no`, String))
        @test Set(source_files())==Set(keys(source_hashes))
        if initial_head!="source_archive"
            tracked=Set(split(read(`git -C $root ls-files`, String), '\n'))
            @test Set(source_files())⊆tracked
        end
        for (file, digest) in source_hashes
            @test bytes2hex(sha256(read(joinpath(root, file))))==digest
        end
    end
    path=get(
        ENV, "COLMAT_ABI_EVIDENCE", joinpath(root, "test-output", "abi-execution.toml")
    )
    mkpath(dirname(path))
    upstream=TOML.parsefile(joinpath(root, "deps/upstream.toml"))
    payload=Dict(
        "schema_version"=>1,
        "full_suite_passed"=>true,
        "julia_version"=>string(VERSION),
        "platform"=>Sys.MACHINE,
        "revision"=>initial_head,
        "native_revision"=>upstream["revision"],
        "inventory_registration_sha256"=>registration_digest(),
        "manifest_sha256"=>bytes2hex(sha256(read(joinpath(root, "Manifest.toml")))),
        "library_sha256"=>bytes2hex(sha256(read(Main.Colmat.backend_library()))),
        "source_sha256"=>source_hashes,
        "cblas_target_symbols"=>sort!(collect(cblas_targets)),
        "witnesses"=>vcat(
            [
                Dict(
                    "symbol"=>s,
                    "bits"=>b,
                    "route"=>r,
                    "file"=>f,
                    "line"=>l,
                    "signature"=>t,
                    "integer_arguments"=>integers,
                ) for (s, b, r, f, l, t, integers) in sort!(collect(witnesses))
            ],
            [
                Dict(
                    "symbol"=>s,
                    "bits"=>b,
                    "route"=>string(r),
                    "file"=>relpath(realpath(f), root),
                    "line"=>l,
                    "signature"=>t,
                    "integer_arguments"=>!endswith(s, "rotg"),
                ) for (s, b, r, f, l, t) in sort!(collect(cblas_sites))
            ],
        ),
        "errors"=>errors,
        "ambiguous_shared_entries"=>sort!(collect(ambiguities)),
    )
    open(io->TOML.print(io, payload), path, "w")
    return println("Actual ABI execution evidence: ", path)
end
end
