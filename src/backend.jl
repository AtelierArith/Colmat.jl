const _library = joinpath(
    @__DIR__, "..", "deps", "usr", "lib", "libcolmat_blas." * Libdl.dlext
)
const _handle = Ref{Ptr{Cvoid}}(C_NULL)
const _configuration_lock = ReentrantLock()

"""
    backend_library() -> String

Return the absolute path of Colmat's BLAS shared library.
"""
backend_library() = abspath(_library)

"""
    load_backend(; verbose=false)

Install Colmat's LP64 and ILP64 BLAS/LAPACK symbols in libblastrampoline for this process.
Existing forwards for unprovided routines are preserved. GEEVX uses the saved
backend for expert condition estimates; its SENSE=N path runs in colmat.
GGEVX runs BALANC=N/SENSE=N in colmat; other modes require a saved ILP64
expert backend, with integer conversion for LP64 callers. Such a backend must
be configured before loading Colmat (Julia normally provides OpenBLAS).
Tridiagonal EVD also saves ILP64 LAPACK entries for native nonconvergence and
the relative/absolute accuracy modes not provided by the native API.
Call again to select Colmat after loading another BLAS provider. Configure the
backend before starting concurrent BLAS work.
"""
function load_backend(; verbose::Bool=false)
    lock(_configuration_lock) do
        path = backend_library()
        isfile(path) || error(
            "Colmat's BLAS library is missing. Run `using Pkg; Pkg.build(\"Colmat\")` and restart Julia.",
        )
        if _handle[] == C_NULL
            _handle[] = Libdl.dlopen(path, Libdl.RTLD_LAZY | Libdl.RTLD_LOCAL)
        end
        abi = ccall(Libdl.dlsym(_handle[], :colmat_blas_abi_version), Cint, ())
        abi == 2 ||
            error("Unsupported Colmat BLAS ABI $abi; rebuild Colmat and restart Julia.")
        # Expert GEEVX condition estimates have no matching public colmat API.
        # Retain the previous implementation for SENSE=E/V/B; SENSE=N is native.
        register = Libdl.dlsym(_handle[], :colmat_lapack_set_expert_fallback)
        for (id, prefix) in enumerate(("s", "d", "c", "z")), interface in (:lp64, :ilp64)
            symbol = prefix * "geevx_"
            previous = BLAS.lbt_find_backing_library(symbol, interface)
            if previous !== nothing && realpath(previous.libname) != realpath(path)
                pointer = BLAS.lbt_get_forward(symbol, interface)
                ccall(
                    register,
                    Cvoid,
                    (Cint, Cint, Ptr{Cvoid}),
                    id - 1,
                    interface === :ilp64 ? 64 : 32,
                    pointer,
                )
            end
        end
        # Generalized expert balancing/condition estimates remain on the previous
        # ILP64 backend. Rust bridges LP64 calls and owns integer scratch buffers.
        register_generalized = Libdl.dlsym(_handle[], :colmat_ggevx_set_expert_fallback)
        has_generalized = Libdl.dlsym(_handle[], :colmat_ggevx_has_expert_fallback)
        for (id, prefix) in enumerate(("s", "d", "c", "z"))
            symbol = prefix * "ggevx_"
            previous = BLAS.lbt_find_backing_library(symbol, :ilp64)
            if previous !== nothing && realpath(previous.libname) != realpath(path)
                pointer = BLAS.lbt_get_forward(symbol, :ilp64)
                ccall(register_generalized, Cvoid, (Cint, Ptr{Cvoid}), id - 1, pointer)
            end
            ccall(has_generalized, Cint, (Cint,), id - 1) != 0 || error(
                "Colmat GGEVX requires an existing ILP64 expert backend; configure one before loading Colmat.",
            )
        end
        # Native tridiagonal EVD does not expose partial D/E on nonconvergence.
        # Save a LAPACK implementation before forwarding; LP64 widens integers.
        register_tridiagonal = Libdl.dlsym(_handle[], :colmat_tridiagonal_set_fallback)
        has_tridiagonal = Libdl.dlsym(_handle[], :colmat_tridiagonal_has_fallback)
        for (id, symbol) in enumerate((
            "ssteqr_",
            "dsteqr_",
            "csteqr_",
            "zsteqr_",
            "ssterf_",
            "dsterf_",
            "sstevd_",
            "dstevd_",
            "spteqr_",
            "dpteqr_",
            "cpteqr_",
            "zpteqr_",
            "sstedc_",
            "dstedc_",
            "cstedc_",
            "zstedc_",
            "sstevr_",
            "dstevr_",
            "sstevx_",
            "dstevx_",
            "sstemr_",
            "dstemr_",
            "cstemr_",
            "zstemr_",
        ))
            previous = BLAS.lbt_find_backing_library(symbol, :ilp64)
            if previous !== nothing && realpath(previous.libname) != realpath(path)
                pointer = BLAS.lbt_get_forward(symbol, :ilp64)
                ccall(register_tridiagonal, Cvoid, (Cint, Ptr{Cvoid}), id - 1, pointer)
            end
            ccall(has_tridiagonal, Cint, (Cint,), id - 1) != 0 || error(
                "Colmat tridiagonal EVD requires an existing ILP64 LAPACK backend; configure one before loading Colmat.",
            )
        end
        # LBT appends the suffix to the complete Fortran name: dgemm_ + 64_.
        # Detect ILP64 first, then LP64, preserving the existing fallback table.
        for suffix in ("64_", "")
            count = BLAS.lbt_forward(path; clear=false, suffix_hint=suffix, verbose)
            count > 0 ||
                error("Could not register Colmat BLAS symbols (suffix=$(repr(suffix))).")
        end
        return ccall(
            (:lbt_register_thread_interface, BLAS.libblastrampoline),
            Cvoid,
            (Cstring, Cstring),
            "colmat_blas_get_num_threads",
            "colmat_blas_set_num_threads",
        )
    end
    return nothing
end

"""
    get_num_threads() -> Int

Return the configured number of participants in Colmat matrix multiplication.
The initial value is one. Other fallback libraries have their own configuration.
"""
function get_num_threads()
    return Int(ccall((:colmat_blas_get_num_threads, _library), Cint, ()))
end

"""
    set_num_threads(n::Integer)

Set the number of participants in Colmat matrix multiplication. Pools are created
lazily per calling OS thread and replaced when the configured count changes.
`LinearAlgebra.BLAS.set_num_threads(n)` configures Colmat and the loaded fallback
libraries together.
"""
function set_num_threads(n::Integer)
    1 <= n <= typemax(Cint) ||
        throw(ArgumentError("thread count must be in 1:$(typemax(Cint))"))
    ccall((:colmat_blas_set_num_threads, _library), Cvoid, (Cint,), n)
    return nothing
end
