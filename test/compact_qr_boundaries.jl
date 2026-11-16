using Test, Colmat, LinearAlgebra, Libdl
for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
    @eval function compact_qr_boundary_probe(::Type{$T}, ::Type{$J}, p, m, n, lda, ldt)
        a=fill($T(777), 16)
        t=fill($T(778), 16)
        info=fill($J(-777), 3)
        ccall(
            p,
            Cvoid,
            (Ref{$J}, Ref{$J}, Ptr{$T}, Ref{$J}, Ptr{$T}, Ref{$J}, Ptr{$J}),
            m,
            n,
            a,
            lda,
            t,
            ldt,
            view(info, 2:2),
        )
        (info, a, t)
    end
end
@testset "GEQRT2/3 header order and empty-output guards" begin
    native=Libdl.dlopen(Colmat.backend_library())
    reference=Libdl.dlopen(
        only(
            filter(
                l->l.interface==:ilp64&&occursin("openblas", lowercase(l.libname)),
                BLAS.get_config().loaded_libs,
            ),
        ).libname,
    )
    cases=(
        (-1, -2, 1, 1, -2),
        (-2, -1, 1, 1, -2),
        (-1, 0, 1, 1, -1),
        (0, -1, 1, 1, -2),
        (1, 2, 1, 2, -1),
        (2, 1, 1, 1, -4),
        (2, 1, 2, 0, -6),
        (0, 0, 1, 1, 0),
        (3, 0, 3, 1, 0),
    )
    for (T, prefix) in
        ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
        J in (Int32, Int64), family in ("geqrt2", "geqrt3"),
        (m, n, lda, ldt, expected) in cases

        ownp=Libdl.dlsym(native, prefix*family*(J==Int64 ? "_64_" : "_"))
        # The isolated reproducer demonstrates reference GEQRT3 recursion at
        # N=0. GEQRT2 is the independent empty-factor oracle for those cases.
        ref_family=family=="geqrt3"&&n==0&&m>=0 ? "geqrt2" : family
        refp=Libdl.dlsym(reference, prefix*ref_family*"_64_")
        @test refp!=Libdl.dlsym(native, prefix*ref_family*"_64_")
        own=compact_qr_boundary_probe(T, J, ownp, m, n, lda, ldt)
        external=compact_qr_boundary_probe(T, Int64, refp, m, n, lda, ldt)
        @test own==external
        @test own[1]==[-777, expected, -777]
        @test own[2]==fill(T(777), 16)
        @test own[3]==fill(T(778), 16)
    end
end
