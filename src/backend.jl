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
Existing forwards for unprovided routines are preserved.
GEEVX uses saved original backends for expert condition estimates. LP64 calls
use integer conversion to the saved ILP64 backend when no original LP64 routine exists;
SENSE=N runs in colmat. An ILP64 GEEVX backend must exist before registration.
GGEVX runs BALANC=N/SENSE=N in colmat; other modes require a saved ILP64
expert backend, with integer conversion for LP64 callers. Such a backend must
be configured before loading Colmat (Julia normally provides OpenBLAS).
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
        # Expert GEEVX condition estimates have no matching public colmat API.
        # Retain the previous implementation for SENSE=E/V/B; SENSE=N is native.
        register = Libdl.dlsym(_handle[], :colmat_lapack_set_expert_fallback)
        has_expert = Libdl.dlsym(_handle[], :colmat_lapack_has_expert_fallback)
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
        for id in 0:3
            ccall(has_expert, Cint, (Cint, Cint), id, 64) != 0 || error(
                "Colmat GEEVX requires an existing ILP64 expert backend; configure one before loading Colmat.",
            )
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
        # Retain complete generalized self-adjoint drivers for native failure
        # recovery and the expert absolute-tolerance mode of the range drivers.
        register_selfadjoint = Libdl.dlsym(_handle[], :colmat_selfadjoint_gv_set_fallback)
        has_selfadjoint = Libdl.dlsym(_handle[], :colmat_selfadjoint_gv_has_fallback)
        for (id, symbol) in enumerate((
            "ssygv_",
            "dsygv_",
            "chegv_",
            "zhegv_",
            "ssygvd_",
            "dsygvd_",
            "chegvd_",
            "zhegvd_",
            "ssygvx_",
            "dsygvx_",
            "chegvx_",
            "zhegvx_",
        ))
            previous=BLAS.lbt_find_backing_library(symbol, :ilp64)
            if previous!==nothing&&realpath(previous.libname)!=realpath(path)
                pointer=BLAS.lbt_get_forward(symbol, :ilp64)
                ccall(register_selfadjoint, Cvoid, (Cint, Ptr{Cvoid}), id-1, pointer)
            end
            ccall(has_selfadjoint, Cint, (Cint,), id-1)!=0||error(
                "Colmat generalized self-adjoint EVD requires an existing ILP64 LAPACK backend; configure one before loading Colmat.",
            )
        end
        # Preserve the expert estimate and failure-recovery drivers before forwarding.
        register_svd=Libdl.dlsym(_handle[], :colmat_advanced_svd_set_fallback)
        has_svd=Libdl.dlsym(_handle[], :colmat_advanced_svd_has_fallback)
        for (id, symbol) in enumerate((
            "sgesvdx_",
            "dgesvdx_",
            "cgesvdx_",
            "zgesvdx_",
            "sgesvdq_",
            "dgesvdq_",
            "cgesvdq_",
            "zgesvdq_",
        ))
            previous=BLAS.lbt_find_backing_library(symbol, :ilp64)
            if previous!==nothing&&realpath(previous.libname)!=realpath(path)
                pointer=BLAS.lbt_get_forward(symbol, :ilp64)
                ccall(register_svd, Cvoid, (Cint, Ptr{Cvoid}), id-1, pointer)
            end
            ccall(has_svd, Cint, (Cint,), id-1)!=0||error(
                "Colmat advanced SVD requires an existing ILP64 LAPACK backend; configure one before loading Colmat.",
            )
        end
        register_vx=Libdl.dlsym(_handle[], :colmat_selfadjoint_vx_set_fallback)
        has_vx=Libdl.dlsym(_handle[], :colmat_selfadjoint_vx_has_fallback)
        for (id, symbol) in enumerate(("ssyevx_", "dsyevx_", "cheevx_", "zheevx_"))
            previous=BLAS.lbt_find_backing_library(symbol, :ilp64)
            if previous!==nothing && realpath(previous.libname)!=realpath(path)
                pointer=BLAS.lbt_get_forward(symbol, :ilp64)
                ccall(register_vx, Cvoid, (Cint, Ptr{Cvoid}), id-1, pointer)
            end
            ccall(has_vx, Cint, (Cint,), id-1)!=0 || error(
                "Colmat selected self-adjoint EVD requires an existing ILP64 LAPACK backend; configure one before loading Colmat.",
            )
        end
        register_structured = Libdl.dlsym(_handle[], :colmat_structured_evd_set_fallback)
        has_structured = Libdl.dlsym(_handle[], :colmat_structured_evd_has_fallback)
        for (id, symbol) in enumerate((
            "ssbevx_",
            "dsbevx_",
            "chbevx_",
            "zhbevx_",
            "sspevx_",
            "dspevx_",
            "chpevx_",
            "zhpevx_",
        ))
            previous = BLAS.lbt_find_backing_library(symbol, :ilp64)
            if previous !== nothing && realpath(previous.libname) != realpath(path)
                pointer = BLAS.lbt_get_forward(symbol, :ilp64)
                ccall(register_structured, Cvoid, (Cint, Ptr{Cvoid}), id - 1, pointer)
            end
            ccall(has_structured, Cint, (Cint,), id - 1) != 0 || error(
                "Colmat band/packed selected EVD requires an existing ILP64 LAPACK backend; configure one before loading Colmat.",
            )
        end
        # TGSEN and GGESX condition estimates use the original ILP64 LAPACK entry;
        # Rust widens LP64 logical selections and integer workspace.
        register_qz = Libdl.dlsym(_handle[], :colmat_qz_reorder_set_fallback)
        has_qz = Libdl.dlsym(_handle[], :colmat_qz_reorder_has_fallback)
        for (id, symbol) in enumerate((
            "stgsen_",
            "dtgsen_",
            "ctgsen_",
            "ztgsen_",
            "sggesx_",
            "dggesx_",
            "cggesx_",
            "zggesx_",
        ))
            previous = BLAS.lbt_find_backing_library(symbol, :ilp64)
            if previous !== nothing && realpath(previous.libname) != realpath(path)
                pointer = BLAS.lbt_get_forward(symbol, :ilp64)
                ccall(register_qz, Cvoid, (Cint, Ptr{Cvoid}), id - 1, pointer)
            end
            ccall(has_qz, Cint, (Cint,), id - 1) != 0 || error(
                "Colmat QZ reordering requires an existing ILP64 LAPACK backend; configure one before loading Colmat.",
            )
        end
        # Structured generalized routines retain original entries for native refusal.
        register_structured_gv = Libdl.dlsym(_handle[], :colmat_structured_gv_set_fallback)
        has_structured_gv = Libdl.dlsym(_handle[], :colmat_structured_gv_has_fallback)
        for (id, symbol) in (
            (0, "sspgv_"),
            (1, "dspgv_"),
            (2, "chpgv_"),
            (3, "zhpgv_"),
            (4, "sspgvd_"),
            (5, "dspgvd_"),
            (6, "chpgvd_"),
            (7, "zhpgvd_"),
            (8, "sspgvx_"),
            (9, "dspgvx_"),
            (10, "chpgvx_"),
            (11, "zhpgvx_"),
            (12, "ssbgst_"),
            (13, "dsbgst_"),
            (14, "chbgst_"),
            (15, "zhbgst_"),
            (16, "ssbgv_"),
            (17, "dsbgv_"),
            (18, "chbgv_"),
            (19, "zhbgv_"),
            (20, "ssbgvd_"),
            (21, "dsbgvd_"),
            (22, "chbgvd_"),
            (23, "zhbgvd_"),
            (24, "ssbgvx_"),
            (25, "dsbgvx_"),
            (26, "chbgvx_"),
            (27, "zhbgvx_"),
        )
            previous = BLAS.lbt_find_backing_library(symbol, :ilp64)
            if previous !== nothing && realpath(previous.libname) != realpath(path)
                pointer = BLAS.lbt_get_forward(symbol, :ilp64)
                ccall(register_structured_gv, Cvoid, (Cint, Ptr{Cvoid}), id, pointer)
            end
            ccall(has_structured_gv, Cint, (Cint,), id) != 0 || error(
                "Colmat structured generalized EVD requires an existing ILP64 LAPACK backend; configure one before loading Colmat.",
            )
        end
        # Schur eigenvector helpers preserve the original driver on refusal.
        register_vectors = Libdl.dlsym(_handle[], :colmat_schur_vectors_set_fallback)
        has_vectors = Libdl.dlsym(_handle[], :colmat_schur_vectors_has_fallback)
        for (id, symbol) in enumerate((
            "strevc_",
            "dtrevc_",
            "ctrevc_",
            "ztrevc_",
            "strevc3_",
            "dtrevc3_",
            "ctrevc3_",
            "ztrevc3_",
        ))
            previous = BLAS.lbt_find_backing_library(symbol, :ilp64)
            if previous !== nothing && realpath(previous.libname) != realpath(path)
                pointer = BLAS.lbt_get_forward(symbol, :ilp64)
                ccall(register_vectors, Cvoid, (Cint, Ptr{Cvoid}), id - 1, pointer)
            end
            ccall(has_vectors, Cint, (Cint,), id - 1) != 0 || error(
                "Colmat Schur eigenvectors require an existing ILP64 LAPACK backend; configure one before loading Colmat.",
            )
        end
        register_schur_expert = Libdl.dlsym(_handle[], :colmat_schur_expert_set_fallback)
        has_schur_expert = Libdl.dlsym(_handle[], :colmat_schur_expert_has_fallback)
        for (id, symbol) in enumerate((
            "strsen_",
            "dtrsen_",
            "ctrsen_",
            "ztrsen_",
            "sgeesx_",
            "dgeesx_",
            "cgeesx_",
            "zgeesx_",
        ))
            previous = BLAS.lbt_find_backing_library(symbol, :ilp64)
            if previous !== nothing && realpath(previous.libname) != realpath(path)
                pointer = BLAS.lbt_get_forward(symbol, :ilp64)
                ccall(register_schur_expert, Cvoid, (Cint, Ptr{Cvoid}), id - 1, pointer)
            end
            ccall(has_schur_expert, Cint, (Cint,), id - 1) != 0 || error(
                "Colmat Schur expert estimates require an existing ILP64 LAPACK backend; configure one before loading Colmat.",
            )
        end
        # LBT appends the suffix to the complete Fortran name: dgemm_ + 64_.
        # Detect ILP64 first, then LP64, preserving the existing fallback table.
        for suffix in ("64_", "")
            count = BLAS.lbt_forward(path; clear=false, suffix_hint=suffix, verbose)
            count > 0 ||
                error("Could not register Colmat BLAS symbols (suffix=$(repr(suffix))).")
        end
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
