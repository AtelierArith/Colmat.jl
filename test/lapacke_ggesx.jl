using Test, Colmat, LinearAlgebra, Libdl, Random
isdefined(@__MODULE__, :lapacke288_selected_entry) || include("lapacke_calls.jl")

const GX_THRESHOLD = Ref(0.0)

for (T, R) in ((Float32, Float32), (Float64, Float64), (ComplexF32, Float32), (ComplexF64, Float64)),
    J in (Int32, Int64)
    # Literal ccalls keep the ABI evidence recorder able to trace every entry.
    if T <: Complex
        @eval begin
            gx_select_pointer(::Type{$T}, ::Type{$J}) =
                @cfunction((a, b) -> Base.unsafe_load(a) / Base.unsafe_load(b) |> x -> $J(real(x) < GX_THRESHOLD[]),
                           $J, (Ptr{$T}, Ptr{$T}))
            function gx_call(::Type{$T}, ::Type{$R}, ::Type{$J}, p, name, route, work, layout, jobvsl, jobvsr, sort, sel, sense, n,
                             a, lda, b, ldb, sdim, ar, ai, be, vsl, ldvsl, vsr, ldvsr, rce, rcv, wk, lw, rw, iw, liw, bw)
                lapacke288_selected_entry(name, $J, route) do
                    work ? ccall(p, $J, (Cint, Cchar, Cchar, Cchar, Ptr{Cvoid}, Cchar, $J, Ptr{$T}, $J, Ptr{$T}, $J, Ptr{$J},
                                         Ptr{$T}, Ptr{$T}, Ptr{$T}, $J, Ptr{$T}, $J, Ptr{$R}, Ptr{$R}, Ptr{$T}, $J, Ptr{$R},
                                         Ptr{$J}, $J, Ptr{$J}),
                                 layout, jobvsl, jobvsr, sort, sel, sense, n, a, lda, b, ldb, sdim, ar, be, vsl, ldvsl, vsr, ldvsr,
                                 rce, rcv, wk, lw, rw, iw, liw, bw) :
                           ccall(p, $J, (Cint, Cchar, Cchar, Cchar, Ptr{Cvoid}, Cchar, $J, Ptr{$T}, $J, Ptr{$T}, $J, Ptr{$J},
                                         Ptr{$T}, Ptr{$T}, Ptr{$T}, $J, Ptr{$T}, $J, Ptr{$R}, Ptr{$R}),
                                 layout, jobvsl, jobvsr, sort, sel, sense, n, a, lda, b, ldb, sdim, ar, be, vsl, ldvsl, vsr, ldvsr,
                                 rce, rcv)
                end
            end
        end
    else
        @eval begin
            gx_select_pointer(::Type{$T}, ::Type{$J}) =
                @cfunction((a, b, c) -> $J(Base.unsafe_load(a) / Base.unsafe_load(c) < GX_THRESHOLD[]),
                           $J, (Ptr{$T}, Ptr{$T}, Ptr{$T}))
            function gx_call(::Type{$T}, ::Type{$R}, ::Type{$J}, p, name, route, work, layout, jobvsl, jobvsr, sort, sel, sense, n,
                             a, lda, b, ldb, sdim, ar, ai, be, vsl, ldvsl, vsr, ldvsr, rce, rcv, wk, lw, rw, iw, liw, bw)
                lapacke288_selected_entry(name, $J, route) do
                    work ? ccall(p, $J, (Cint, Cchar, Cchar, Cchar, Ptr{Cvoid}, Cchar, $J, Ptr{$T}, $J, Ptr{$T}, $J, Ptr{$J},
                                         Ptr{$T}, Ptr{$T}, Ptr{$T}, Ptr{$T}, $J, Ptr{$T}, $J, Ptr{$T}, Ptr{$T}, Ptr{$T}, $J,
                                         Ptr{$J}, $J, Ptr{$J}),
                                 layout, jobvsl, jobvsr, sort, sel, sense, n, a, lda, b, ldb, sdim, ar, ai, be, vsl, ldvsl, vsr,
                                 ldvsr, rce, rcv, wk, lw, iw, liw, bw) :
                           ccall(p, $J, (Cint, Cchar, Cchar, Cchar, Ptr{Cvoid}, Cchar, $J, Ptr{$T}, $J, Ptr{$T}, $J, Ptr{$J},
                                         Ptr{$T}, Ptr{$T}, Ptr{$T}, Ptr{$T}, $J, Ptr{$T}, $J, Ptr{$T}, Ptr{$T}),
                                 layout, jobvsl, jobvsr, sort, sel, sense, n, a, lda, b, ldb, sdim, ar, ai, be, vsl, ldvsl, vsr,
                                 ldvsr, rce, rcv)
                end
            end
        end
    end
end

gx_row(m, ld) = (out = fill(eltype(m)(NaN), size(m, 1) * ld);
    for j in 1:size(m, 2), i in 1:size(m, 1); out[(i - 1) * ld + j] = m[i, j]; end; out)
gx_fromrow(v, m, n, ld) = [v[(i - 1) * ld + j] for i in 1:m, j in 1:n]

@testset "LAPACKE GGESX" begin
    native = Libdl.dlopen(Colmat.backend_library())
    trampoline = Libdl.dlopen(BLAS.libblastrampoline)
    for (T, R, prefix) in ((Float32, Float32, "s"), (Float64, Float64, "d"), (ComplexF32, Float32, "c"), (ComplexF64, Float64, "z")),
        J in (Int32, Int64), n in (2, 4), (srt, sense) in (('N', 'N'), ('S', 'N'), ('S', 'B')), (jobvsl, jobvsr) in (('N', 'N'), ('V', 'V')),
        work in (false, true), route in (:dll, :lbt_trampoline)

        suffix = J === Int64 ? "64_" : ""
        interface = J === Int64 ? :ilp64 : :lp64
        name = "LAPACKE_" * prefix * "ggesx" * (work ? "_work" : "")
        @test BLAS.lbt_get_forward(name, interface) == Libdl.dlsym(native, name * suffix)
        p = Libdl.dlsym(route === :dll ? native : trampoline, name * suffix)
        rng = MersenneTwister(6500 + n)
        a0 = randn(rng, T, n, n)
        b0 = a0 * a0' + T(n) * Matrix{T}(I, n, n)
        GX_THRESHOLD[] = Float64(sort(real.(eigvals(b0 \ a0)))[max(n ÷ 2, 1)]) + 1e-3
        sel = srt == 'S' ? gx_select_pointer(T, J) : C_NULL
        results = Dict()
        for layout in (102, 101)
            ld = layout == 102 ? n : n + 2
            a = layout == 102 ? vec(copy(a0)) : gx_row(a0, ld); b = layout == 102 ? vec(copy(b0)) : gx_row(b0, ld)
            vsl = fill(T(NaN), n * ld); vsr = fill(T(NaN), n * ld)
            ar = zeros(T, n); ai = zeros(R, n); be = zeros(T, n); sd = J[-1]; rce = R[NaN, NaN]; rcv = R[NaN, NaN]
            ai = T <: Complex ? ai : zeros(T, n)
            wk = fill(T(NaN), 1); rw = fill(R(NaN), 8n + 8); iw = zeros(J, 1); bw = zeros(J, n)
            if work
                @test gx_call(T, R, J, p, name, route, true, layout, jobvsl, jobvsr, srt, sel, sense, J(n), copy(a), J(ld), copy(b), J(ld),
                              copy(sd), copy(ar), copy(ai), copy(be), copy(vsl), J(ld), copy(vsr), J(ld), copy(rce), copy(rcv),
                              wk, J(-1), rw, iw, J(-1), bw) == 0
                wk = fill(T(NaN), Int(real(wk[1])) + 2n * n + 1); iw = zeros(J, max(Int(iw[1]), 1))
            end
            info = gx_call(T, R, J, p, name, route, work, layout, jobvsl, jobvsr, srt, sel, sense, J(n), a, J(ld), b, J(ld), sd, ar, ai, be,
                           vsl, J(ld), vsr, J(ld), rce, rcv, wk, J(length(wk)), rw, iw, J(length(iw)), bw)
            @test info == 0
            sdim = sd[1]
            if srt == 'N'
                @test sdim == 0
            else
                @test 0 < sdim <= n
            end
            amat = layout == 102 ? reshape(a, n, n) : gx_fromrow(a, n, n, ld)
            results[layout] = (copy(amat), copy(ar), copy(be), sdim)
        end
        tol = 5000 * eps(R) * max(n, 1)
        @test results[101][4] == results[102][4]
        @test isapprox(results[101][1], results[102][1]; rtol=tol, atol=tol)
        @test isapprox(results[101][2], results[102][2]; rtol=tol, atol=tol)
        @test isapprox(results[101][3], results[102][3]; rtol=tol, atol=tol)
    end
end

@testset "LAPACKE GGESX diagnostics" begin
    native = Libdl.dlopen(Colmat.backend_library())
    for (T, R, prefix) in ((Float64, Float64, "d"), (ComplexF64, Float64, "z")), J in (Int32, Int64)
        suffix = J === Int64 ? "64_" : ""
        name = "LAPACKE_" * prefix * "ggesx"; p = Libdl.dlsym(native, name * suffix)
        a = zeros(T, 16); b = zeros(T, 16); vsl = zeros(T, 16); vsr = zeros(T, 16)
        ar = zeros(T, 4); ai = zeros(T == Float64 ? Float64 : R, 4); be = zeros(T, 4); sd = J[0]; rce = R[0, 0]; rcv = R[0, 0]
        ai = T <: Complex ? ai : zeros(T, 4)
        g(layout, lda, ldb, ldl, ldr, aa, bb) = gx_call(T, R, J, p, name, :dll, false, layout, 'V', 'V', 'N', C_NULL, 'N', J(4), aa, J(lda), bb, J(ldb),
            sd, ar, ai, be, vsl, J(ldl), vsr, J(ldr), rce, rcv, T[], J(0), R[], J[], J(0), J[])
        @test g(301, 4, 4, 4, 4, a, b) == -1
        @test g(101, 3, 4, 4, 4, a, b) == -9
        @test g(101, 4, 3, 4, 4, a, b) == -11
        @test g(101, 4, 4, 3, 4, a, b) == -(T <: Complex ? 16 : 17)
        @test g(101, 4, 4, 4, 3, a, b) == -(T <: Complex ? 18 : 19)
        bad = copy(a); bad[1] = T(NaN)
        @test g(102, 4, 4, 4, 4, bad, b) == -8
        @test g(102, 4, 4, 4, 4, a, bad) == -10
    end
    ccall(Libdl.dlsym(native, :colmat_blas_last_error), Cint, ())
end
