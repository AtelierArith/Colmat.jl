using Test, LinearAlgebra, Libdl, Colmat
isdefined(@__MODULE__, :direct_ordinary_trsen!) || include("ordinary_schur_selection.jl")
isdefined(@__MODULE__, :direct_ordinary_geesx!) || include("ordinary_schur_expert.jl")
@testset "All ordinary Schur reordering symbols forward through native LBT" begin
    for (T, prefix) in
        ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
        J in (Int32, Int64),
        family in (T<:Real ? (:trexc, :laexc, :trsen, :geesx) : (:trexc, :trsen, :geesx))

        symbol=prefix*String(family)*"_"
        interface=J===Int64 ? :ilp64 : :lp64
        provider=BLAS.lbt_find_backing_library(symbol, interface)
        @test provider!==nothing&&realpath(provider.libname)==realpath(
            Colmat.backend_library()
        )
        pointer=BLAS.lbt_get_forward(symbol, interface)
        initial=T[3 0.5; 0 1]
        t=vcat(initial, fill(T(777), 3, 2))
        q=fill(T(777), 5, 2)
        q[1:2, :].=Matrix{T}(I, 2, 2)
        result=if family===:trexc
            direct_ordinary_trexc!(t, q, J; first=1, last=2, pointer)
        elseif family===:laexc
            direct_ordinary_laexc!(t, q, J; first=1, n1=1, n2=1, pointer)
        elseif family===:trsen
            mask=J[-777, 0, 1, -777]
            direct_ordinary_trsen!(t, q, mask, J; lwork=2, liwork=1, pointer)
        else
            direct_ordinary_geesx!(t, q, J; lwork=T<:Real ? 6 : 4, liwork=1, pointer)
        end
        @test result.info==result.diagnostic==0
        @test ordinary_schur_values(t)≈[1, 3]
        @test q[1:2, :]*t[1:2, :]*q[1:2, :]'≈initial
        @test q[1:2, :]'*q[1:2, :]≈Matrix{T}(I, 2, 2)
        @test all(==(T(777)), t[3:end, :])&&all(==(T(777)), q[3:end, :])
        if family in (:trsen, :geesx)
            @test result.count==1
        end
    end
end
@testset "Schur eigenvalue extraction retains tiny imaginary parts next to large real parts" begin
    lib=Libdl.dlopen(
        only(
            filter(
                l->l.interface==:ilp64&&occursin(
                    "openblas", lowercase(basename(l.libname))
                ),
                BLAS.get_config().loaded_libs,
            ),
        ).libname,
    )
    for (T, prefix) in ((Float32, "s"), (Float64, "d")),
        J in (Int32, Int64),
        allselected in (false, true)

        re, beta=T===Float32 ? (T(1e20), T(1e-10)) : (T(1e200), T(1e-100))
        initial=T[re beta; -beta re]
        t=vcat(initial, fill(T(777), 3, 2))
        tr=copy(t)
        q=vcat(Matrix{T}(I, 2, 2), fill(T(777), 3, 2))
        qr=copy(q)
        oldq=copy(q)
        mask=J[-777, allselected, allselected, -777]
        mr=Int64.(mask)
        result=direct_ordinary_trsen!(t, q, mask, J; lwork=2, liwork=1)
        oracle=direct_ordinary_trsen!(
            tr,
            qr,
            mr,
            Int64;
            lwork=2,
            liwork=1,
            pointer=Libdl.dlsym(lib, Symbol(prefix*"trsen_64_")),
        )
        @test result.info==result.diagnostic==oracle.info==0
        @test result.count==oracle.count==(allselected ? 2 : 0)
        @test real(result.values[1])==real(result.values[2])==re
        @test imag(result.values[1])>0&&imag(result.values[2])<0
        @test imag(result.values[1])≈beta rtol=8eps(T) atol=0
        @test imag(result.values[2])≈-beta rtol=8eps(T) atol=0
        @test imag.(result.values)≈imag.(oracle.values) rtol=8eps(T) atol=0
        @test isequal(t[1:2, :], initial)&&isequal(tr[1:2, :], initial)&&isequal(q, oldq)&&isequal(
            qr, oldq
        )
    end
end
