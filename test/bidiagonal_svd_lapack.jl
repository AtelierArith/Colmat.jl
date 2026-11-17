using Test, Colmat, LinearAlgebra, Libdl, Random

const BDS_LIBRARY = Ref{Any}(nothing)
bds_library() = BDS_LIBRARY[] === nothing ? (BDS_LIBRARY[] = Libdl.dlopen(Colmat.backend_library())) : BDS_LIBRARY[]

for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
    prefix = T === Float32 ? "s" : T === Float64 ? "d" : T === ComplexF32 ? "c" : "z"
    R = real(T)
    suffix = J === Int64 ? "_64_" : "_"
    @eval function bdsqr_call(
        ::Type{$T}, ::Type{$J}, uplo, n, d::Vector{$R}, e::Vector{$R},
        vt::Matrix{$T}, u::Matrix{$T}, c::Matrix{$T};
        ldvt=max(1, size(vt, 1)), ldu=max(1, size(u, 1)), ldc=max(1, size(c, 1)),
        work=fill($R(NaN), max(1, 4n)),
    )
        pointer = Libdl.dlsym(bds_library(), $(prefix * "bdsqr" * suffix))
        info = Ref{$J}(0)
        ncvt, nru, ncc = size(vt, 2), size(u, 1), size(c, 2)
        if $T <: Complex
            ccall(
                pointer, Cvoid,
                (Ref{UInt8}, Ref{$J}, Ref{$J}, Ref{$J}, Ref{$J},
                 Ptr{$R}, Ptr{$R}, Ptr{$T}, Ref{$J}, Ptr{$T}, Ref{$J},
                 Ptr{$T}, Ref{$J}, Ptr{$R}, Ref{$J}),
                uplo, n, ncvt, nru, ncc, d, e, vt, ldvt, u, ldu, c, ldc,
                fill($R(NaN), max(1, 4n)), info,
            )
        else
            ccall(
                pointer, Cvoid,
                (Ref{UInt8}, Ref{$J}, Ref{$J}, Ref{$J}, Ref{$J},
                 Ptr{$R}, Ptr{$R}, Ptr{$T}, Ref{$J}, Ptr{$T}, Ref{$J},
                 Ptr{$T}, Ref{$J}, Ptr{$T}, Ref{$J}),
                uplo, n, ncvt, nru, ncc, d, e, vt, ldvt, u, ldu, c, ldc,
                work, info,
            )
        end
        (; info=info[], d=copy(d), e=copy(e), vt=copy(vt), u=copy(u), c=copy(c))
    end

    @eval function bdsdc_call(
        ::Type{$T}, ::Type{$J}, uplo, compq, n, d::Vector{$T}, e::Vector{$T};
        u=zeros($T, max(1, n), max(1, n)), vt=zeros($T, max(1, n), max(1, n)),
        q=zeros($T, 2000), iq=zeros($J, 500),
    )
        pointer = Libdl.dlsym(bds_library(), $(prefix * "bdsdc" * suffix))
        info = Ref{$J}(0)
        ldu, ldvt = max(1, size(u, 1)), max(1, size(vt, 1))
        work = fill($T(NaN), max(1, 4n))
        iwork = zeros($J, max(1, 8n))
        ccall(
            pointer, Cvoid,
            (Ref{UInt8}, Ref{UInt8}, Ref{$J}, Ptr{$T}, Ptr{$T}, Ptr{$T},
             Ref{$J}, Ptr{$T}, Ref{$J}, Ptr{$T}, Ptr{$J}, Ptr{$T},
             Ptr{$J}, Ref{$J}),
            uplo, compq, n, d, e, u, ldu, vt, ldvt, q, iq, work, iwork, info,
        )
        (; info=info[], d=copy(d), e=copy(e), u=copy(u), vt=copy(vt), q, iq)
    end

    real(T) == T && @eval function bdsvdx_call(
        ::Type{$T}, ::Type{$J}, uplo, jobz, range, n, d::Vector{$T}, e::Vector{$T};
        vl=$T(0), vu=$T(0), il=1, iu=n, ldz=max(1, 2n),
    )
        pointer = Libdl.dlsym(bds_library(), $(prefix * "bdsvdx" * suffix))
        s = fill($T(NaN), max(1, n))
        z = fill($T(NaN), max(1, ldz), max(1, n + 1))
        ns = Ref{$J}(-777)
        info = Ref{$J}(0)
        ccall(
            pointer, Cvoid,
            (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{$J}, Ptr{$T}, Ptr{$T},
             Ref{$T}, Ref{$T}, Ref{$J}, Ref{$J}, Ref{$J}, Ptr{$T}, Ptr{$T},
             Ref{$J}, Ptr{$T}, Ptr{$J}, Ref{$J}),
            uplo, jobz, range, n, d, e, vl, vu, il, iu, ns, s, z, ldz,
            fill($T(NaN), max(1, 14n)), zeros($J, max(1, 12n)), info,
        )
        (; info=info[], ns=ns[], s=s, z=z)
    end

    real(T) == T && @eval function lasq1_call(
        ::Type{$T}, ::Type{$J}, n, d::Vector{$T}, e::Vector{$T}
    )
        pointer = Libdl.dlsym(bds_library(), $(prefix * "lasq1" * suffix))
        info = Ref{$J}(0)
        ccall(
            pointer, Cvoid,
            (Ref{$J}, Ptr{$T}, Ptr{$T}, Ptr{$T}, Ref{$J}),
            n, d, e, fill($T(NaN), max(1, 4n)), info,
        )
        (; info=info[], d=copy(d), e=copy(e))
    end
end

function bidiagonal_matrix(T, d, e, upper)
    n = length(d)
    b = zeros(T, n, n)
    for i in 1:n
        b[i, i] = d[i]
        if i < n
            upper ? (b[i, i + 1] = e[i]) : (b[i + 1, i] = e[i])
        end
    end
    return b
end

@testset "BDSQR native bidiagonal SVD and factor application" begin
    for (T, R, tol) in
        ((Float32, Float32, 2f-4), (Float64, Float64, 2e-12),
         (ComplexF32, Float32, 3f-4), (ComplexF64, Float64, 3e-12)),
        J in (Int32, Int64), n in (0, 1, 4), upper in (false, true)

        d = R[1 + i for i in 1:n]
        e = R[0.5i - 0.2 for i in 1:max(0, n - 1)]
        n > 1 && (d[end] = 0)
        b = bidiagonal_matrix(T, d, e, upper)
        # Values only.
        values = bdsqr_call(
            T, J, upper ? 'U' : 'L', n, copy(d), copy(e),
            zeros(T, max(1, n), 0), zeros(T, 0, max(1, n)), zeros(T, max(1, n), 0),
        )
        @test values.info == 0
        n > 0 && @test issorted(values.d; rev=true)
        expected = svdvals(b)
        n > 0 && @test isapprox(sort(values.d; rev=true), expected; rtol=tol, atol=tol)
        # Full identities.
        if n > 0
            rng = MersenneTwister(n + (upper ? 1 : 0))
            vt_in = Matrix{T}(I, n, n)
            u_in = Matrix{T}(I, n, n)
            c_in = Matrix{T}(I, n, n)
            full = bdsqr_call(
                T, J, upper ? 'U' : 'L', n, copy(d), copy(e),
                copy(vt_in), copy(u_in), copy(c_in),
            )
            @test full.info == 0
            @test isapprox(full.u * Diagonal(full.d) * full.vt, b; rtol=20tol, atol=20tol)
            @test isapprox(full.u' * full.u, Matrix{T}(I, n, n); rtol=20tol, atol=20tol)
            @test isapprox(full.vt * full.vt', Matrix{T}(I, n, n); rtol=20tol, atol=20tol)
            @test isapprox(full.c, full.u'; rtol=20tol, atol=20tol)
            # Partial factors: VT is n-by-2, U is 3-by-n, C is n-by-2.
            ncvt, nru, ncc = 2, 3, 2
            vt_part = randn(rng, T, n, ncvt)
            u_part = randn(rng, T, nru, n)
            c_part = randn(rng, T, n, ncc)
            part = bdsqr_call(
                T, J, upper ? 'U' : 'L', n, copy(d), copy(e),
                copy(vt_part), copy(u_part), copy(c_part),
            )
            @test part.info == 0
            @test isapprox(part.vt, full.vt * vt_part; rtol=50tol, atol=50tol)
            @test isapprox(part.u, u_part * full.u; rtol=50tol, atol=50tol)
            @test isapprox(part.c, full.u' * c_part; rtol=50tol, atol=50tol)
        end
    end
end

@testset "BDSDC native full SVD and compact fallback" begin
    for (T, tol) in ((Float32, 3f-4), (Float64, 3e-12)), J in (Int32, Int64),
        n in (0, 1, 5), upper in (false, true)

        d = T[1 + i for i in 1:n]
        e = T[0.5i - 0.2 for i in 1:max(0, n - 1)]
        b = bidiagonal_matrix(T, d, e, upper)
        for compq in ('N', 'I')
            result = bdsdc_call(
                T, J, upper ? 'U' : 'L', compq, n, copy(d), copy(e)
            )
            @test result.info == 0
            if n > 0
                @test isapprox(sort(result.d; rev=true), svdvals(b); rtol=tol, atol=tol)
                if compq == 'I'
                    @test isapprox(result.u * Diagonal(result.d) * result.vt, b; rtol=30tol, atol=30tol)
                    @test isapprox(result.u' * result.u, Matrix{T}(I, n, n); rtol=30tol, atol=30tol)
                end
            end
        end
        if n > 0
            full = bdsdc_call(T, J, upper ? 'U' : 'L', 'I', n, copy(d), copy(e))
            compact = bdsdc_call(T, J, upper ? 'U' : 'L', 'P', n, copy(d), copy(e))
            @test compact.info == 0
            @test isapprox(compact.d, full.d; rtol=tol, atol=tol)
            @test any(!iszero, compact.q)
            again = bdsdc_call(T, J, upper ? 'U' : 'L', 'P', n, copy(d), copy(e))
            @test compact.q == again.q
            @test compact.iq == again.iq
        end
    end
end

@testset "BDSVDX range selection and Z vectors" begin
    for (T, tol) in ((Float32, 3f-4), (Float64, 3e-12)), J in (Int32, Int64),
        n in (0, 1, 5), upper in (false, true)

        d = T[1 + i for i in 1:n]
        e = T[0.5i - 0.2 for i in 1:max(0, n - 1)]
        b = bidiagonal_matrix(T, d, e, upper)
        all = bdsvdx_call(T, J, upper ? 'U' : 'L', 'N', 'A', n, copy(d), copy(e))
        @test all.info == 0
        @test all.ns == n
        n > 0 && @test isapprox(all.s[1:n], sort(svdvals(b)); rtol=tol, atol=tol)
        if n > 0
            vectors = bdsvdx_call(T, J, upper ? 'U' : 'L', 'V', 'A', n, copy(d), copy(e))
            @test vectors.info == 0
            @test vectors.ns == n
            u = vectors.z[1:n, 1:n]
            v = vectors.z[n+1:2n, 1:n]
            @test isapprox(u * Diagonal(vectors.s[1:n]) * v', b; rtol=40tol, atol=40tol)
            # Value range [smallest, second smallest).
            sorted = sort(svdvals(b))
            lo, hi = sorted[1], sorted[min(2, n)]
            ranged = bdsvdx_call(
                T, J, upper ? 'U' : 'L', 'N', 'V', n, copy(d), copy(e);
                vl=lo - tol, vu=hi + tol,
            )
            @test ranged.info == 0
            @test ranged.ns >= 1
            vals = ranged.s[1:Int(ranged.ns)]
            @test minimum(vals) >= lo - tol
            @test maximum(vals) <= hi + tol
            indexed = bdsvdx_call(
                T, J, upper ? 'U' : 'L', 'N', 'I', n, copy(d), copy(e);
                il=1, iu=min(2, n),
            )
            @test indexed.info == 0
            @test indexed.ns == min(2, n)
            @test isapprox(indexed.s[1:indexed.ns], sorted[1:indexed.ns]; rtol=tol, atol=tol)
        end
    end
end

@testset "LASQ1 singular values and diagnostics" begin
    for (T, tol) in ((Float32, 3f-4), (Float64, 3e-12)), J in (Int32, Int64),
        n in (0, 1, 5)

        d = T[1 + i for i in 1:n]
        e = T[0.5i - 0.2 for i in 1:max(0, n - 1)]
        b = bidiagonal_matrix(T, d, e, false)
        result = lasq1_call(T, J, n, copy(d), copy(e))
        @test result.info == 0
        n > 0 && @test isapprox(result.d, svdvals(b); rtol=tol, atol=tol)
    end
    # Diagnostics.
    for (T, J) in ((Float64, Int32), (Float64, Int64))
        d = [1.0, 2.0]
        e = [0.5]
        b = bidiagonal_matrix(T, d, e, false)
        @test bdsqr_call(T, J, '?', 2, copy(d), copy(e), zeros(T, 2, 0), zeros(T, 0, 2), zeros(T, 2, 0)).info == -1
        @test bdsqr_call(T, J, 'L', -1, copy(d), copy(e), zeros(T, 2, 0), zeros(T, 0, 2), zeros(T, 2, 0)).info == -2
        @test bdsdc_call(T, J, 'L', '?', 2, copy(d), copy(e)).info == -2
        @test bdsvdx_call(T, J, 'L', 'V', 'A', 2, copy(d), copy(e)).info == 0
        @test bdsvdx_call(T, J, 'L', '?', 'A', 2, copy(d), copy(e)).info == -2
        @test bdsvdx_call(T, J, 'L', 'N', '?', 2, copy(d), copy(e)).info == -3
        @test lasq1_call(T, J, -1, copy(d), copy(e)).info == -1
    end
end
