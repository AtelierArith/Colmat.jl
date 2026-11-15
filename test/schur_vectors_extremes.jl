using Test, LinearAlgebra, Libdl, Colmat
isdefined(@__MODULE__, :direct_schur_vectors!) || include("schur_vectors.jl")
@testset "Schur eigenvectors retain relative residuals at extreme scales" begin
    external=only(
        filter(
            l->l.interface==:ilp64&&occursin("openblas", lowercase(basename(l.libname))),
            BLAS.get_config().loaded_libs,
        ),
    )
    lib=Libdl.dlopen(external.libname)
    for (T, prefix) in
        ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
        J in (Int32, Int64), blocked in (false, true), side in ('L', 'R', 'B'),
        how in ('A', 'B', 'S')

        R=real(T)
        tol=R===Float32 ? 4e-3 : 5e-10
        for scale in (R===Float32 ? (R(1e-30), R(1e30)) : (R(1e-200), R(1e200)))
            n=5
            t=schur_vectors_fixture(T, n, :pairs) .* scale
            layout=schur_vectors_layout(t)
            ql=Matrix(qr(randn(T, n, n)).Q)
            qr_matrix=Matrix(qr(randn(T, n, n)).Q)
            flags=J.(iseven.(1:n))
            selected=vcat(J(-777), flags, J(-777))
            sr=Int64.(selected)
            chosen=if how=='S'
                reduce(
                vcat,
                [
                    collect(j:(j + len - 1)) for
                    (j, len) in layout if any(!iszero, flags[j:(j + len - 1)])
                ];
                init=Int[],
            )
            else
                collect(1:n)
            end
            vl=fill(T(777), side=='R' ? 1 : n+3, n+2)
            vr=fill(T(777), side=='L' ? 1 : n+3, n+2)
            if how=='B'
                side!='R'&&(vl[1:n, 1:n].=ql)
                side!='L'&&(vr[1:n, 1:n].=qr_matrix)
            end
            vlr=copy(vl)
            vrr=copy(vr)
            stored=vcat(t, fill(T(777), 3, n))
            old=copy(stored)
            result=direct_schur_vectors!(
                stored, vl, vr, selected, J, Val(blocked); side, how, mm=n+2
            )
            oracle=direct_schur_vectors!(
                stored,
                vlr,
                vrr,
                sr,
                Int64,
                Val(blocked);
                side,
                how,
                mm=n+2,
                pointer=Libdl.dlsym(
                    lib, Symbol(prefix*(blocked ? "trevc3_64_" : "trevc_64_"))
                ),
            )
            @test result.info==result.diagnostic==oracle.info==0
            @test isequal(stored, old)&&selected==sr
            check_schur_vectors(
                result, vl, vr, t, layout, chosen, side, how, ql, qr_matrix, tol
            )
            check_schur_vectors(
                oracle, vlr, vrr, t, layout, chosen, side, how, ql, qr_matrix, tol
            )
        end
    end
end
@testset "TREVC and TREVC3 actual LBT providers and arithmetic" begin
    for (T, prefix) in
        ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
        J in (Int32, Int64),
        blocked in (false, true)

        symbol=prefix*(blocked ? "trevc3_" : "trevc_")
        interface=J===Int64 ? :ilp64 : :lp64
        provider=BLAS.lbt_find_backing_library(symbol, interface)
        @test provider!==nothing&&realpath(provider.libname)==realpath(
            Colmat.backend_library()
        )
        pointer=BLAS.lbt_get_forward(symbol, interface)
        t=reshape(T[2], 1, 1)
        vl=fill(T(777), 4, 2)
        vr=copy(vl)
        selected=J[-777, 0, -777]
        result=direct_schur_vectors!(
            t, vl, vr, selected, J, Val(blocked); side=('B'), how=('A'), mm=2, pointer
        )
        @test result.info==result.diagnostic==0&&result.count==1
        @test abs(vl[1, 1])≈1&&abs(vr[1, 1])≈1
        @test all(==(T(777)), vl[2:end, :])&&all(==(T(777)), vr[2:end, :])
    end
end
