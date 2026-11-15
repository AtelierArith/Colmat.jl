using Test, LinearAlgebra, Libdl, Colmat
isdefined(@__MODULE__, :direct_schur_vectors!) || include("schur_vectors.jl")
@testset "TREVC3 minimal workspace collective queries and validation order" begin
    external=only(
        filter(
            l->l.interface==:ilp64&&occursin("openblas", lowercase(basename(l.libname))),
            BLAS.get_config().loaded_libs,
        ),
    )
    lib=Libdl.dlopen(external.libname)
    for (T, prefix) in
        ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
        J in (Int32, Int64), n in (0, 1, 5), side in ('L', 'R', 'B'),
        how in ('A', 'B', 'S')

        t=schur_vectors_fixture(T, n, :pairs)
        stored=vcat(t, fill(T(777), 3, n))
        old=copy(stored)
        initial=vcat(J(-777), J.(iseven.(1:n)), J(-777))
        for query in (T<:Real ? (:work,) : (:work, :real))
            selected=copy(initial)
            sr=Int64.(selected)
            vl=fill(T(777), 1, 1)
            vr=copy(vl)
            vlr=copy(vl)
            vrr=copy(vr)
            result=direct_schur_vectors!(
                stored,
                vl,
                vr,
                selected,
                J,
                Val(true);
                side,
                how,
                query,
                ldvl=max(1, n),
                ldvr=max(1, n),
                lwork=-9,
                lrwork=-9,
            )
            oracle=direct_schur_vectors!(
                stored,
                vlr,
                vrr,
                sr,
                Int64,
                Val(true);
                side,
                how,
                query,
                ldvl=max(1, n),
                ldvr=max(1, n),
                lwork=-9,
                lrwork=-9,
                pointer=Libdl.dlsym(lib, Symbol(prefix*"trevc3_64_")),
            )
            @test result.info==result.diagnostic==oracle.info==0
            @test result.count==oracle.count
            @test selected==sr
            minimum=max(1, (T<:Real ? 3 : 2)*n)
            @test real(result.work[1])==minimum
            @test real(oracle.work[1])>=minimum
            if T<:Complex
                @test result.rwork[1]==max(1, n)
                @test oracle.rwork[1]>=max(1, n)
            end
            @test isequal(stored, old)
            @test vl==vr==vlr==vrr==fill(T(777), 1, 1)
        end
        for blocked in (false, true)
            minimum=max(1, (T<:Real ? 3 : 2)*n)
            cases=[
                (; side=('?'), expected=-1),
                (; how=('?'), expected=-2),
                (; n=-1, expected=-4),
                (; ldt=max(0, n-1), expected=-6),
                (; ldvl=0, expected=-8),
                (; ldvr=0, expected=-10),
                (; mm=-1, expected=-11),
            ]
            if blocked
                push!(cases, (; lwork=minimum-1, expected=-14))
                if T<:Complex
                    push!(cases, (; lrwork=max(1, n)-1, expected=-16))
                end
                # Real checks LWORK before MM; complex checks MM first.
                push!(cases, (; mm=-1, lwork=minimum-1, expected=T<:Real ? -14 : -11))
            end
            for case in cases
                expected=case.expected
                options=Base.structdiff(case, (; expected))
                selected=copy(initial)
                sr=Int64.(selected)
                vl=fill(T(777), n+3, n+2)
                vr=copy(vl)
                vlr=copy(vl)
                vrr=copy(vr)
                # Keywords in a case override the normal valid headers.
                kwargs=merge((; side, how, mm=n+2), options)
                result=direct_schur_vectors!(
                    stored, vl, vr, selected, J, Val(blocked); kwargs...
                )
                oracle=direct_schur_vectors!(
                    stored,
                    vlr,
                    vrr,
                    sr,
                    Int64,
                    Val(blocked);
                    kwargs...,
                    pointer=Libdl.dlsym(
                        lib, Symbol(prefix*(blocked ? "trevc3_64_" : "trevc_64_"))
                    ),
                )
                @test result.info==oracle.info==expected
                @test result.diagnostic==-expected
                @test selected==sr
                @test isequal(stored, old)
                @test vl==vr==vlr==vrr==fill(T(777), n+3, n+2)
            end
        end
    end
end
