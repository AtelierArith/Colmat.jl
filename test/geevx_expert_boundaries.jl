using Test, LinearAlgebra, Libdl, Colmat
isdefined(@__MODULE__, :direct_geevx!) || include("geevx_expert_helpers.jl")
@testset "GEEVX exact invalid arguments and minimum workspace, all types and ABIs" begin
    reference=geevx_reference()
    for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
        off=T<:Real ? 1 : 0
        cases=[
            ('?', 'V', 'V', 'E', 4, 6, 6, 6, 1),
            ('N', '?', 'V', 'E', 4, 6, 6, 6, 2),
            ('N', 'V', '?', 'E', 4, 6, 6, 6, 3),
            ('N', 'V', 'V', '?', 4, 6, 6, 6, 4),
            ('N', 'N', 'V', 'E', 4, 6, 6, 6, 4),
            ('N', 'V', 'N', 'B', 4, 6, 6, 6, 4),
            ('N', 'V', 'V', 'E', -1, 6, 6, 6, 5),
            ('N', 'V', 'V', 'E', 4, 0, 6, 6, 7),
            ('N', 'V', 'V', 'E', 4, 6, 0, 6, 10+off),
            ('N', 'V', 'V', 'E', 4, 6, 6, 0, 12+off),
            ('N', 'V', 'V', '?', typemax(J), 1, 1, 1, 4),
            ('N', 'V', 'V', 'E', typemax(J), 0, 1, 1, 7),
        ]
        for (bal, left, right, sense, n, lda, ldvl, ldvr, arg) in cases,
            lwork in (-1, 100),
            (library, K) in ((Colmat.backend_library(), J), (reference, Int64))

            a=vcat(geevx_fixture(T, 4), fill(T(777), 2, 4))
            old=copy(a)
            result=direct_geevx!(
                a, K; bal, left, right, sense, n, lda, ldvl, ldvr, lwork, library
            )
            @test result.info == -arg
            @test result.diagnostic == (library==Colmat.backend_library() ? arg : 0)
            @test isequal(a, old)
            @test result.ilo==result.ihi==-777 && result.abnrm==777
            @test all(==(T(777)), result.values) &&
                all(==(T(777)), result.left) &&
                all(==(T(777)), result.right)
            @test result.workspace==777
        end
        for sense in ('N', 'E', 'V', 'B'), left in ('N', 'V'), right in ('N', 'V')
            sense in ('E', 'B') && (left!='V'||right!='V') && continue
            minimum=geevx_minimum(T, 4, left, right, sense)
            for lwork in (-2, minimum-1),
                (library, K) in ((Colmat.backend_library(), J), (reference, Int64))

                a=vcat(geevx_fixture(T, 4), fill(T(777), 2, 4))
                old=copy(a)
                result=direct_geevx!(a, K; sense, left, right, lwork, library)
                @test result.info == -(20+off)
                @test result.diagnostic == (library==Colmat.backend_library() ? 20+off : 0)
                @test isequal(a, old)
                # WORK(1) is advisory; different implementations may recommend different blocks.
                @test result.workspace>=minimum ||
                    (library==Colmat.backend_library() && sense=='N')
            end
        end
    end
end
