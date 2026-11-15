using Test, LinearAlgebra, Colmat
isdefined(@__MODULE__, :direct_structured_bandgvx!) ||
    include("structured_generalized_lbt.jl")

@testset "Structured generalized zero rank-one and repeated spectra" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        u in ('U', 'L'),
        spectrum in ((0, 0, 0, 0), (0, 0, 0, 6), (0, 0, 6, 6)),
        family in (:pgv, :pgvd, :pgvx, :bgv, :bgvd, :bgvx)

        n = 4
        a = Matrix(Diagonal(T.(collect(spectrum))))
        b = 2Matrix{T}(I, n, n)
        tol = T in (Float32, ComplexF32) ? 2e-3 : 2e-10
        for kind in (startswith(String(family), "p") ? (1:3) : (1:1))
            z = fill(T(777), n + 3, n)
            q = copy(z)
            result = if startswith(String(family), "p")
                ap = structured_gv_pack(a, u; poison=true)
                bp = structured_gv_pack(b, u; poison=true)
                if family === :pgv
                    direct_structured_gv!(ap, bp, z, J; n, kind, u)
                elseif family === :pgvd
                    direct_structured_gvd!(ap, bp, z, J; n, kind, u)
                else
                    direct_structured_gvx!(ap, bp, z, J; n, kind, u)
                end
            else
                ab = structured_band_pack(a, 0, u)
                bb = structured_band_pack(b, 0, u)
                if family === :bgv
                    direct_structured_bandgv!(ab, bb, z, J; ka=0, kb=0, u)
                elseif family === :bgvd
                    direct_structured_bandgvd!(ab, bb, z, J; ka=0, kb=0, u)
                else
                    direct_structured_bandgvx!(ab, bb, q, z, J; ka=0, kb=0, u)
                end
            end
            @test result.info == result.diagnostic == 0
            @test result.values ≈ collect(spectrum) .* (kind == 1 ? 0.5 : 2) atol=tol rtol=tol
            x = z[1:n, :]
            d = Diagonal(result.values)
            lhs = if kind == 1
                a*x
            elseif kind == 2
                a*b*x
            else
                b*a*x
            end
            rhs = kind == 1 ? b*x*d : x*d
            @test lhs ≈ rhs atol=tol rtol=tol
            @test x' * (kind == 3 ? b\x : b*x) ≈ Matrix{T}(I, n, n) atol=tol rtol=tol
            @test all(==(T(777)), z[(n + 1):end, :])
            if family === :bgvx
                basis = q[1:n, :]
                @test basis'*b*basis ≈ Matrix{T}(I, n, n) atol=tol rtol=tol
                @test basis'*a*basis ≈ d atol=tol rtol=tol
            end
        end
    end
end
