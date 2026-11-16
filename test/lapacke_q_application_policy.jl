using Test, Colmat, LinearAlgebra, Libdl
isdefined(@__MODULE__, :lapacke288_qapply_call)||include("lapacke_q_application.jl")
@testset "LAPACKE Q application: diagnostics and logical NaN bounds" begin
    native=Libdl.dlopen(Colmat.backend_library())
    getter=Libdl.dlsym(native, :LAPACKE_get_nancheck)
    setter=Libdl.dlsym(native, :LAPACKE_set_nancheck)
    oldflag=ccall(getter, Cint, ())
    ccall(setter, Cvoid, (Cint,), 1)
    try
        for (T, prefix) in
            ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
            J in (Int32, Int64),
            orientation in ("qr", "lq", "ql", "rq"),
            layout in (101, 102),
            (m, n) in ((3, 5), (5, 3)),
            side in ('L', 'R')

            rows=orientation in ("lq", "rq")
            reverse=orientation in ("ql", "rq")
            order=side=='L' ? m : n
            k=1
            ar, ac=rows ? (k, order) : (order, k)
            la=layout==101 ? ac+2 : ar+2
            lc=layout==101 ? n+2 : m+2
            ai(i, j) = layout==101 ? (i-1)*la+j : i+(j-1)*la
            ci(i, j) = layout==101 ? (i-1)*lc+j : i+(j-1)*lc
            a=zeros(T, 128)
            tau=T[1, NaN, 779]
            c=ones(T, 128)
            work=fill(T(778), 16)
            family=(T<:Complex ? "unm" : "orm")*orientation
            name="LAPACKE_"*prefix*family
            suffix=J==Int64 ? "64_" : ""
            hp=Libdl.dlsym(native, name*suffix)
            wp=Libdl.dlsym(native, name*"_work"*suffix)
            minimum=max(1, side=='L' ? n : m)
            low(l, s, t, mm, nn, kk, lda, ldc, lw) = lapacke288_qapply_call(
                T,
                J,
                wp,
                false,
                l,
                s,
                t,
                mm,
                nn,
                kk,
                a,
                lda,
                tau,
                c,
                ldc,
                work,
                lw,
                name*"_work",
                :dll,
            )
            high(kk) = lapacke288_qapply_call(
                T,
                J,
                hp,
                true,
                layout,
                side,
                'N',
                m,
                n,
                kk,
                a,
                la,
                tau,
                c,
                lc,
                work,
                minimum,
                name,
                :dll,
            )
            @test low(0, side, 'N', m, n, k, la, lc, -1)==-1
            @test low(102, 'X', 'N', m, n, k, 8, 8, -1)==-2
            @test low(102, side, 'X', m, n, k, 8, 8, -1)==-3
            @test low(102, side, 'N', -1, n, 0, 8, 8, -1)==-4
            @test low(102, side, 'N', m, -1, 0, 8, 8, -1)==-5
            @test low(102, side, 'N', m, n, -1, 8, 8, -1)==-6
            @test low(102, side, 'N', m, n, order+1, 8, 8, -1)==-6
            @test low(102, side, 'N', m, n, k, 0, 8, -1)==-8
            @test low(102, side, 'N', m, n, k, 8, 0, -1)==-11
            @test low(101, 'X', 'N', m, n, k, 0, 0, -1)==-8
            @test low(101, 'X', 'N', m, n, k, 8, 0, -1)==-11
            before=copy(c)
            @test low(layout, side, 'N', m, n, k, la, lc, minimum-1)==-13
            @test c==before
            @test low(layout, lowercase(side), 'n', m, n, k, la, lc, minimum)==0
            fill!(c, 1)
            # A stored diagonal is ignored numerically; high scans logical A.
            diagonal=rows ? ai(1, reverse ? order : 1) : ai(reverse ? order : 1, 1)
            a[diagonal]=T(NaN)
            saved_a=copy(a)
            savedtau=copy(tau)
            before=copy(c)
            @test high(k)==-7
            @test isequal(a, saved_a)&&c==before&&isequal(tau, savedtau)
            @test low(layout, side, 'N', m, n, k, la, lc, minimum)==0
            @test all(isfinite, T[c[ci(i, j)] for i in 1:m, j in 1:n])
            fill!(c, 1)
            ccall(setter, Cvoid, (Cint,), 0)
            @test high(k)==0
            ccall(setter, Cvoid, (Cint,), 1)
            @test isequal(a, saved_a)&&isequal(tau, savedtau)
            # High diagnostic order is A, C, TAU; negative INFO keeps C.
            fill!(c, 1)
            c[ci(1, 1)]=T(NaN)
            tau[1]=T(NaN)
            before=copy(c)
            @test high(k)==-7
            fill!(a, 0)
            @test high(k)==-10
            @test isequal(c, before)
            fill!(c, 1)
            before=copy(c)
            @test high(k)==-9
            @test c==before
            tau[1]=one(T)
            savedtau=copy(tau)
            @test high(k)==0
            @test isequal(tau, savedtau)
            # Correct the reference M-vs-order scan using the logical extent.
            fill!(a, 0)
            fill!(c, 1)
            active=rows ? ai(1, order) : ai(order, 1)
            a[active]=T(NaN)
            before=copy(c)
            @test high(k)==-7
            @test c==before
            fill!(a, 0)
            padding=rows ? ai(1, order+1) : ai(order+1, 1)
            a[padding]=T(NaN)
            saved_a=copy(a)
            @test high(k)==0
            @test isequal(a, saved_a)
            fill!(a, T(NaN))
            fill!(tau, T(NaN))
            fill!(c, 1)
            before=copy(c)
            @test high(0)==0
            @test c==before
            if J==Int64
                @test low(102, side, 'N', -4294967293, 0, 0, 1, 1, -1)==-4
                @test low(102, 'L', 'N', 0, 4294967296, 0, 1, 1, -1)==0
                @test real(work[1])>=4294967296
                @test low(layout, side, 'N', m, n, k, la, lc, -4294967293)==-13
            end
        end
    finally
        ccall(setter, Cvoid, (Cint,), oldflag)
    end
end
