# Run with the isolated BenchmarkTools/JET/AllocCheck environment documented in
# the accompanying report. No profiling dependency is added to package runtime.
module GeneralEVDBenchmark
using BenchmarkTools, JET, AllocCheck, Profile
using LinearAlgebra, Libdl, Random, TOML, SHA, Statistics
const ROOT=realpath(dirname(@__DIR__))

for T in (Float32, Float64, ComplexF32, ComplexF64)
    R=real(T)
    signature=Any[
        Ref{UInt8},
        Ref{UInt8},
        Ref{UInt8},
        Ref{UInt8},
        Ref{Int64},
        Ptr{T},
        Ref{Int64},
        Ptr{T},
    ]
    args=Any[:(x.bal), :(x.left), :(x.right), UInt8('N'), :(x.n), :a, :(x.n), :(x.w)]
    if T<:Real
        push!(signature, Ptr{R})
        push!(args, :(x.wi))
    end
    append!(
        signature,
        [
            Ptr{T},
            Ref{Int64},
            Ptr{T},
            Ref{Int64},
            Ref{Int64},
            Ref{Int64},
            Ptr{R},
            Ref{R},
            Ptr{R},
            Ptr{R},
            Ptr{T},
            Ref{Int64},
        ],
    )
    append!(
        args,
        [
            :(x.vl),
            :(size(x.vl, 1)),
            :(x.vr),
            :(size(x.vr, 1)),
            :(x.ilo),
            :(x.ihi),
            :(x.scale),
            :(x.abnrm),
            :(x.rconde),
            :(x.rcondv),
            :(x.work),
            :(length(x.work)),
        ],
    )
    push!(signature, T<:Real ? Ptr{Int64} : Ptr{R})
    push!(args, T<:Real ? :(x.iwork) : :(x.rwork))
    append!(signature, [Ref{Int64}, Clong, Clong, Clong, Clong])
    append!(args, [:(x.info), 1, 1, 1, 1])
    @eval function native_evd!(pointer, a::Matrix{$T}, x)
        ccall(pointer, Cvoid, ($(signature...),), $(args...))
        return x.info[]
    end
end

function workspace(::Type{T}, n, left, right, bal; fixture="scaled_dense") where {T}
    R=real(T)
    rng=MersenneTwister(318+n)
    original=randn(rng, T, n, n)
    if fixture=="scaled_isolated"
        original[1, :].=0
        original[:, 1].=0
        original[1, 1]=T(3)
    end
    if startswith(fixture, "scaled")
        scale=R.(exp2.(range(-8, 8; length=n)))
        for j in 1:n, i in 1:n
            original[i, j]*=scale[i]/scale[j]
        end
    end
    minimum=max(1, T<:Complex || (left=='N'&&right=='N') ? 2n : 3n)
    return (;
        n=Int64(n),
        original,
        a=copy(original),
        bal=UInt8(bal),
        left=UInt8(left),
        right=UInt8(right),
        w=zeros(T, n),
        wi=zeros(R, n),
        vl=zeros(T, left=='V' ? n : 1, left=='V' ? n : 1),
        vr=zeros(T, right=='V' ? n : 1, right=='V' ? n : 1),
        ilo=Ref{Int64}(-777),
        ihi=Ref{Int64}(-777),
        scale=zeros(R, n),
        abnrm=Ref{R}(0),
        rconde=zeros(R, n),
        rcondv=zeros(R, n),
        work=zeros(T, minimum),
        iwork=zeros(Int64, max(1, 2n)),
        rwork=zeros(R, max(1, 2n)),
        info=Ref{Int64}(-777),
    )
end
reset!(x) = copyto!(x.a, x.original)
function unpack(x, wi)
    z=complex.(x)
    j=1
    while j<=length(wi)
        if wi[j]>0
            z[:, j].=x[:, j] .+ im .* x[:, j + 1]
            z[:, j + 1].=conj.(z[:, j])
            j+=2
        else
            j+=1
        end
    end
    return z
end
function quality(::Type{T}, x) where {T}
    x.info[]==0 || error("Native EVD returned INFO=$(x.info[])")
    w=T<:Real ? complex.(x.w, x.wi) : x.w
    result=Dict{String,Any}(
        "info"=>x.info[], "ilo"=>x.ilo[], "ihi"=>x.ihi[], "abnrm"=>x.abnrm[]
    )
    for (name, job, vectors, op, values) in (
        ("left", x.left, x.vl, x.original', conj.(w)),
        ("right", x.right, x.vr, x.original, w),
    )
        job==UInt8('V') || continue
        z=T<:Real ? unpack(vectors, x.wi) : vectors
        residual=norm(op*z-z*Diagonal(values))/max(1, norm(x.original)*norm(z))
        normalization=maximum(abs(norm(v)-1) for v in eachcol(z))
        result[name * "_residual"]=residual
        result[name * "_normalization_error"]=normalization
        residual<=1000x.n*eps(real(T)) || error("Invalid $name residual: $residual")
    end
    return result
end
AllocCheck.@check_allocs checked_native!(pointer, a, x) = native_evd!(pointer, a, x)

function diagnostics(pointer, x, path)
    reset!(x)
    native_evd!(pointer, x.a, x)
    open(path, "w") do io
        println(io, "JET_AVAILABLE = ", JET.JET_AVAILABLE)
        println(
            io,
            "Julia caller analysis; foreign Rust allocations are not proven by these tools.",
        )
        JET.JET_AVAILABLE && show(
            io,
            MIME("text/plain"),
            JET.report_opt(native_evd!, (typeof(pointer), typeof(x.a), typeof(x))),
        )
        println(io)
        try
            reset!(x)
            result=checked_native!(pointer, x.a, x)
            println(io, "AllocCheck caller result: ", result)
        catch err
            println(io, "AllocCheck limitation/failure: ", sprint(showerror, err))
            if hasproperty(err, :errors)
                for item in err.errors
                    show(io, MIME("text/plain"), item)
                    println(io)
                end
            end
        end
        Profile.clear()
        Profile.@profile for _ in 1:40
            reset!(x)
            native_evd!(pointer, x.a, x)
        end
        Profile.print(io; format=:flat, sortedby=:count, C=true)
        Profile.Allocs.clear()
        reset!(x)
        Profile.Allocs.@profile sample_rate=1.0 native_evd!(pointer, x.a, x)
        println(io, "\nJulia allocation stacks (not native Rust allocation accounting):")
        allocations=Profile.Allocs.fetch()
        if isempty(allocations.allocs)
            println(
                io,
                "No Julia allocations sampled at sample_rate=1.0; native allocations remain unmeasured.",
            )
        else
            Profile.Allocs.print(io, allocations)
        end
        return nothing
    end
end

const PHASES=(
    "balance",
    "balanced_norm",
    "native_evd_combined",
    "hessenberg",
    "schur",
    "left_unbalance",
    "left_output_copy",
    "right_unbalance",
    "right_output_copy",
    "balance_metadata",
    "reserved10",
    "reserved11",
    "eigenvectors",
    "form_q",
    "total",
)
function capture_native(pointer, x, capture)
    raw=zeros(UInt64, 9+5length(PHASES))
    function once()
        reset!(x)
        ccall(capture.begin_capture, Cint, ())==0 || error("Native capture already active")
        try
            native_evd!(pointer, x.a, x)
        finally
            ccall(capture.end_capture, Cint, (Ptr{UInt64}, Csize_t), raw, length(raw))==0 ||
                error("Native capture snapshot failed")
        end
        x.info[]==0 || error("Native profiling call failed")
        raw[1]==1 && raw[2]==length(PHASES) || error("Native profiling schema mismatch")
        phases=[
            Dict(
                "name"=>name,
                "allocations"=>raw[10 + 5(i - 1)],
                "allocated_bytes"=>raw[11 + 5(i - 1)],
                "frees"=>raw[12 + 5(i - 1)],
                "freed_bytes"=>raw[13 + 5(i - 1)],
                "inclusive_ns"=>raw[14 + 5(i - 1)],
            ) for (i, name) in enumerate(PHASES)
        ]
        sum(p["allocations"] for p in phases)==raw[3] ||
            error("Allocation snapshot inconsistent")
        sum(p["allocated_bytes"] for p in phases)==raw[4] ||
            error("Allocation byte snapshot inconsistent")
        sum(p["frees"] for p in phases)==raw[5] || error("Free snapshot inconsistent")
        sum(p["freed_bytes"] for p in phases)==raw[6] ||
            error("Free byte snapshot inconsistent")
        return Dict(
            "allocations"=>raw[3],
            "allocated_bytes"=>raw[4],
            "frees"=>raw[5],
            "freed_bytes"=>raw[6],
            "baseline_live_bytes"=>raw[7],
            "ending_live_bytes"=>raw[8],
            "extra_peak_live_bytes"=>raw[9],
            "phases"=>phases,
        )
    end
    once() # Initialize profiler clocks and any remaining native caches.
    captures=[once() for _ in 1:5]
    return Dict(
        "captures"=>captures,
        "median_allocations"=>median([c["allocations"] for c in captures]),
        "median_allocated_bytes"=>median([c["allocated_bytes"] for c in captures]),
        "median_extra_peak_live_bytes"=>median([
            c["extra_peak_live_bytes"] for c in captures
        ]),
    )
end

function run_case(
    ::Type{T},
    pointer,
    x,
    left,
    right,
    bal,
    threads,
    fixture,
    cold,
    capture;
    samples=10,
    seconds=0.1,
) where {T}
    n=x.n
    quality(T, x)
    reset!(x)
    native_evd!(pointer, x.a, x)
    budget=max(seconds, 20cold.time)
    trial=@benchmark native_evd!($pointer, $(x.a), $x) setup=(reset!($x)) evals=1 samples=samples seconds=budget
    length(trial)==samples || error("Insufficient timing samples")
    quality(T, x)
    reset_trial=@benchmark reset!($x) evals=1 samples=samples seconds=seconds
    row=Dict{String,Any}(
        "scalar"=>string(T),
        "n"=>n,
        "input_seed"=>318+n,
        "left"=>string(left),
        "right"=>string(right),
        "balance"=>string(bal),
        "participants"=>threads,
        "fixture"=>fixture,
        "cold_seconds"=>cold.time,
        "steady_median_ns"=>median(trial).time,
        "steady_min_ns"=>minimum(trial).time,
        "julia_memory_bytes"=>trial.memory,
        "julia_allocations"=>trial.allocs,
        "input_reset_median_ns"=>median(reset_trial).time,
        "samples"=>length(trial),
        "native_allocation_measurement"=>"not instrumented in this pass",
    )
    hasproperty(cold, :compile_time) && (row["cold_compile_seconds"]=cold.compile_time)
    merge!(row, quality(T, x))
    row["timing_build"]=if isnothing(capture)
        "production"
    else
        "profiling feature, capture disabled during wall timing"
    end
    if !isnothing(capture)
        row["native_profile"]=capture_native(pointer, x, capture)
        row["native_allocation_measurement"]="Rust GlobalAlloc requests; not Julia allocations, RSS or OS stacks"
    end
    return row
end
function main(args=ARGS)
    quick="--quick" in args
    BLAS.set_num_threads(1)
    library=get(
        ENV,
        "COLMAT_EVD_LIBRARY",
        joinpath(ROOT, "deps/usr/lib/libcolmat_blas."*Libdl.dlext),
    )
    handle=Libdl.dlopen(library, Libdl.RTLD_LAZY|Libdl.RTLD_LOCAL)
    has_profile=Libdl.dlsym(handle, :colmat_evd_profile_begin; throw_error=false)
    has_profile = has_profile!==nothing && has_profile!=C_NULL
    if has_profile && !("--native-profile" in args)
        error(
            "Profiling library requires --native-profile; do not label its wall time as production",
        )
    end
    capture=if "--native-profile" in args
        (
            begin_capture=Libdl.dlsym(handle, :colmat_evd_profile_begin),
            end_capture=Libdl.dlsym(handle, :colmat_evd_profile_end),
        )
    else
        nothing
    end
    setter=Libdl.dlsym(handle, :colmat_blas_set_num_threads)
    getter=Libdl.dlsym(handle, :colmat_blas_get_num_threads)
    rows=Dict{String,Any}[]
    types=(Float32, Float64, ComplexF32, ComplexF64)
    sizes=quick ? (64,) : (32, 96, 256)
    fixtures=quick ? ("scaled_dense",) : ("dense", "scaled_isolated")
    for threads in (quick ? (1,) : (1, 2))
        ccall(setter, Cvoid, (Cint,), threads)
        ccall(getter, Cint, ())==threads || error("Native participant count mismatch")
        for T in types,
            n in sizes,
            fixture in fixtures,
            (left, right) in
            (quick ? (('V', 'V'),) : (('N', 'N'), ('V', 'N'), ('N', 'V'), ('V', 'V'))),
            bal in (quick ? ('B',) : ('N', 'P', 'S', 'B'))

            prefix=if T==Float32
                "s"
            elseif T==Float64
                "d"
            elseif T==ComplexF32
                "c"
            else
                "z"
            end
            pointer=Libdl.dlsym(handle, prefix*"geevx_64_")
            x=workspace(T, n, left, right, bal; fixture)
            # Invocation boundaries prevent inference of later warm calls from
            # compiling the foreign-call method before the cold timer starts.
            cold=@timed Base.invokelatest(native_evd!, pointer, x.a, x)
            push!(
                rows,
                Base.invokelatest(
                    run_case,
                    T,
                    pointer,
                    x,
                    left,
                    right,
                    bal,
                    threads,
                    fixture,
                    cold,
                    capture,
                ),
            )
            println(
                "Measured ",
                T,
                " n=",
                n,
                " ",
                left,
                right,
                " balance=",
                bal,
                " participants=",
                threads,
                " ",
                fixture,
            )
            flush(stdout)
        end
    end
    output=get(ENV, "COLMAT_EVD_OUTPUT", joinpath(tempdir(), "colmat-318-baseline.toml"))
    metadata=Dict(
        "source_revision"=>strip(read(`git -C $ROOT rev-parse HEAD`, String)),
        "native_revision"=>TOML.parsefile(joinpath(ROOT, "deps/upstream.toml"))["revision"],
        "julia_version"=>string(VERSION),
        "validation_blas"=>sprint(show, BLAS.get_config()),
        "platform"=>Sys.MACHINE,
        "library_sha256"=>bytes2hex(sha256(read(library))),
        "benchmark_sha256"=>bytes2hex(sha256(read(@__FILE__))),
        "tool_manifest_sha256"=>bytes2hex(
            sha256(read(joinpath(dirname(Base.active_project()), "Manifest.toml")))
        ),
        "rustc"=>read(`rustc -Vv`, String),
        "cases"=>rows,
    )
    Sys.isapple() &&
        (metadata["cpu_model"]=strip(read(`sysctl -n machdep.cpu.brand_string`, String)))
    open(io->TOML.print(io, metadata), output, "w")
    pointer=Libdl.dlsym(handle, :dgeevx_64_)
    Base.invokelatest(
        diagnostics,
        pointer,
        workspace(Float64, 64, 'V', 'V', 'B'),
        output*".diagnostics.txt",
    )
    return println("Saved ", output)
end
end
if abspath(PROGRAM_FILE)==@__FILE__
    GeneralEVDBenchmark.main()
end
