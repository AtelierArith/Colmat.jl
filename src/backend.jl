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

# LAPACK entry points that colmat cannot satisfy for every argument combination
# keep the implementation that was registered before Colmat so the Rust shim can
# delegate the remaining modes. `group` is the registry index in
# `deps/rust/src/lib.rs`; the fallback id is the 0-based position in `symbols`.
struct _FallbackGroup
    description::String
    requirement::String
    group::Int
    symbols::Tuple{Vararg{String}}
end

function _FallbackGroup(description, group, symbols)
    return _FallbackGroup(description, "ILP64 LAPACK backend", group, symbols)
end

function _fallback_group(description::AbstractString)
    return only(group for group in _FALLBACK_GROUPS if group.description == description)
end

# Group the names by type family for readability; fallback ids still follow the
# flattened order.
_symbols(rows...) = Tuple(Iterators.flatten(rows))

# Registration order only determines which error is reported first.
const _FALLBACK_GROUPS = (
    _FallbackGroup(
        "GGEVX", "ILP64 expert backend", 0, ("sggevx_", "dggevx_", "cggevx_", "zggevx_")
    ),
    _FallbackGroup(
        "tridiagonal EVD",
        1,
        _symbols(
            ("ssteqr_", "dsteqr_", "csteqr_", "zsteqr_"),
            ("ssterf_", "dsterf_"),
            ("sstevd_", "dstevd_"),
            ("spteqr_", "dpteqr_", "cpteqr_", "zpteqr_"),
            ("sstedc_", "dstedc_", "cstedc_", "zstedc_"),
            ("sstevr_", "dstevr_"),
            ("sstevx_", "dstevx_"),
            ("sstemr_", "dstemr_", "cstemr_", "zstemr_"),
        ),
    ),
    _FallbackGroup(
        "generalized self-adjoint EVD",
        2,
        _symbols(
            ("ssygv_", "dsygv_", "chegv_", "zhegv_"),
            ("ssygvd_", "dsygvd_", "chegvd_", "zhegvd_"),
            ("ssygvx_", "dsygvx_", "chegvx_", "zhegvx_"),
        ),
    ),
    _FallbackGroup(
        "advanced SVD",
        3,
        _symbols(
            ("sgesvdx_", "dgesvdx_", "cgesvdx_", "zgesvdx_"),
            ("sgesvdq_", "dgesvdq_", "cgesvdq_", "zgesvdq_"),
        ),
    ),
    _FallbackGroup(
        "selected self-adjoint EVD", 4, ("ssyevx_", "dsyevx_", "cheevx_", "zheevx_")
    ),
    _FallbackGroup(
        "range self-adjoint EVD", 5, ("ssyevr_", "dsyevr_", "cheevr_", "zheevr_")
    ),
    _FallbackGroup(
        "complex triangular products", 6, ("clauum_", "zlauum_", "clauu2_", "zlauu2_")
    ),
    _FallbackGroup(
        "band/packed selected EVD",
        7,
        _symbols(
            ("ssbevx_", "dsbevx_", "chbevx_", "zhbevx_"),
            ("sspevx_", "dspevx_", "chpevx_", "zhpevx_"),
        ),
    ),
    _FallbackGroup(
        "QZ reordering",
        8,
        _symbols(
            ("stgsen_", "dtgsen_", "ctgsen_", "ztgsen_"),
            ("sggesx_", "dggesx_", "cggesx_", "zggesx_"),
        ),
    ),
    _FallbackGroup(
        "structured generalized EVD",
        9,
        _symbols(
            ("sspgv_", "dspgv_", "chpgv_", "zhpgv_"),
            ("sspgvd_", "dspgvd_", "chpgvd_", "zhpgvd_"),
            ("sspgvx_", "dspgvx_", "chpgvx_", "zhpgvx_"),
            ("ssbgst_", "dsbgst_", "chbgst_", "zhbgst_"),
            ("ssbgv_", "dsbgv_", "chbgv_", "zhbgv_"),
            ("ssbgvd_", "dsbgvd_", "chbgvd_", "zhbgvd_"),
            ("ssbgvx_", "dsbgvx_", "chbgvx_", "zhbgvx_"),
        ),
    ),
    _FallbackGroup(
        "Schur eigenvectors",
        10,
        _symbols(
            ("strevc_", "dtrevc_", "ctrevc_", "ztrevc_"),
            ("strevc3_", "dtrevc3_", "ctrevc3_", "ztrevc3_"),
        ),
    ),
    _FallbackGroup("BDSDC compact mode", 11, ("sbdsdc_", "dbdsdc_")),
    _FallbackGroup(
        "Schur expert estimates",
        12,
        _symbols(
            ("strsen_", "dtrsen_", "ctrsen_", "ztrsen_"),
            ("sgeesx_", "dgeesx_", "cgeesx_", "zgeesx_"),
        ),
    ),
)

# Save the current ILP64 implementation for every symbol that still points at
# another library, then require the Rust shim to have a fallback for each one.
function _install_fallbacks!(own::AbstractString, group::_FallbackGroup, config)
    register = Libdl.dlsym(_handle[], :colmat_fallback_set)
    has = Libdl.dlsym(_handle[], :colmat_fallback_has)
    id = Cint(group.group)
    for (slot, symbol) in enumerate(group.symbols)
        previous = BLAS.lbt_find_backing_library(symbol, :ilp64; config)
        if previous !== nothing && realpath(previous.libname) != own
            pointer = BLAS.lbt_get_forward(symbol, :ilp64)
            ccall(register, Cvoid, (Cint, Cint, Ptr{Cvoid}), id, Cint(slot - 1), pointer)
        end
        ccall(has, Cint, (Cint, Cint), id, Cint(slot - 1)) != 0 || error(
            "Colmat $(group.description) requires an existing $(group.requirement); configure one before loading Colmat.",
        )
    end
    return nothing
end

# GEEVX is the only group that keeps both interfaces and distinguishes the
# integer width. Its LP64 slot may fall back to the saved ILP64 implementation,
# so availability is checked at ILP64 only.
function _install_geevx_fallbacks!(own::AbstractString, config)
    register = Libdl.dlsym(_handle[], :colmat_lapack_set_expert_fallback)
    has = Libdl.dlsym(_handle[], :colmat_lapack_has_expert_fallback)
    for (id, prefix) in enumerate(("s", "d", "c", "z")), interface in (:lp64, :ilp64)
        symbol = prefix * "geevx_"
        previous = BLAS.lbt_find_backing_library(symbol, interface; config)
        if previous !== nothing && realpath(previous.libname) != own
            pointer = BLAS.lbt_get_forward(symbol, interface)
            bits = interface === :ilp64 ? 64 : 32
            ccall(register, Cvoid, (Cint, Cint, Ptr{Cvoid}), id - 1, bits, pointer)
        end
    end
    for id in 0:3
        ccall(has, Cint, (Cint, Cint), id, 64) != 0 || error(
            "Colmat GEEVX requires an existing ILP64 expert backend; configure one before loading Colmat.",
        )
    end
    return nothing
end

# Capture fallback providers before either interface changes the LBT table.
# Unlike a complete provider, Colmat must preserve forwards for missing routines.
function _forward_to_colmat(path; verbose::Bool=false)
    own = realpath(path)
    config = BLAS.lbt_get_config()
    _install_geevx_fallbacks!(own, config)
    for group in _FALLBACK_GROUPS
        _install_fallbacks!(own, group, config)
    end
    # LBT appends the suffix to the complete Fortran name: dgemm_ + 64_.
    for suffix in ("64_", "")
        count = BLAS.lbt_forward(path; clear=false, suffix_hint=suffix, verbose)
        count > 0 ||
            error("Could not register Colmat BLAS symbols (suffix=$(repr(suffix))).")
    end
    return nothing
end

"""
    load_backend(; verbose=false)

Install Colmat's LP64 and ILP64 BLAS/LAPACK symbols in libblastrampoline for this process.
Existing forwards for unprovided routines are preserved.
GEEVX uses saved original backends for expert condition estimates. LP64 calls
use integer conversion to the saved ILP64 backend when no original LP64 routine exists;
SENSE=N runs in colmat. An ILP64 GEEVX backend must exist before registration.
GGEVX runs BALANC=N/SENSE=N in colmat; other modes require a saved ILP64
expert backend, with integer conversion for LP64 callers. Such a backend must
be configured before loading Colmat (Julia normally provides OpenBLAS).
SYEVR/HEEVR save ILP64 range drivers for positive ABSTOL, native failures and
value intervals near rounded native eigenvalues; LP64 integer outputs are converted.
Tridiagonal EVD also saves ILP64 LAPACK entries for native nonconvergence and
the relative/absolute accuracy modes not provided by the native API.
Generalized self-adjoint drivers save ILP64 entries for native failures and
positive-ABSTOL range calculations. SYEVX/HEEVX also save ILP64 entries for
positive ABSTOL and native failures.
Band/packed selected EVD also retains ILP64 expert entries for positive ABSTOL,
native failures and value-range endpoints within a precision-based guard of native eigenvalues.
Selected and QR-preconditioned SVD save ILP64 entries for native failures and
GESVDQ condition estimation; independent Jacobi drivers remain on the existing backend.
TGSEN reorder-only and GGESX SENSE=N calls run in colmat, with ILP64 LAPACK entries for
condition and separation estimates, with integer conversion for LP64 callers.
Ordinary Schur reordering uses native block swaps; TRSEN JOB=N and GEESX SENSE=N
run natively, with saved ILP64 entries for other condition estimates. LP64
integer workspaces and LOGICAL SELECT callbacks are converted explicitly.
Structured generalized EVD also saves original ILP64 entries for numerical recovery,
positive ABSTOL, and precision guards at selected value boundaries.
Call again to select Colmat after loading another BLAS provider. Configure the
backend before starting concurrent BLAS work.
"""
function load_backend(; verbose::Bool=false)
    lock(_configuration_lock) do
        path = backend_library()
        isfile(path) || error(
            "Colmat's BLAS library is missing. Run `using Pkg; Pkg.build(\"Colmat\")` and restart Julia.",
        )
        first_load = _handle[] == C_NULL
        if first_load
            _handle[] = Libdl.dlopen(path, Libdl.RTLD_LAZY | Libdl.RTLD_LOCAL)
        end
        abi = ccall(Libdl.dlsym(_handle[], :colmat_blas_abi_version), Cint, ())
        abi == 2 ||
            error("Unsupported Colmat BLAS ABI $abi; rebuild Colmat and restart Julia.")
        _forward_to_colmat(path; verbose)
        # Colmat's pool starts at one participant. On first load, adopt the
        # current BLAS width so loading the backend does not silently serialize
        # threaded work. Reloading keeps any narrower setting already chosen.
        threads = first_load ? BLAS.get_num_threads() : nothing
        ccall(
            (:lbt_register_thread_interface, BLAS.libblastrampoline),
            Cvoid,
            (Cstring, Cstring),
            "colmat_blas_get_num_threads",
            "colmat_blas_set_num_threads",
        )
        if threads !== nothing
            ccall((:colmat_blas_set_num_threads, _library), Cvoid, (Cint,), threads)
        end
        return nothing
    end
    return nothing
end

"""
    get_num_threads() -> Int

Return the configured number of participants in Colmat matrix multiplication.
The initial value is the BLAS width observed when Colmat was loaded. Other
fallback libraries have their own configuration.
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
