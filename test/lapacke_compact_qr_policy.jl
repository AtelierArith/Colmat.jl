using Test, Colmat, LinearAlgebra, Libdl
isdefined(@__MODULE__, :lapacke288_compact_apply)||include("lapacke_compact_qr.jl")
@testset "Compact QR C diagnostics and logical NaN bounds" begin
    native=Libdl.dlopen(Colmat.backend_library())
    getter=Libdl.dlsym(native, :LAPACKE_get_nancheck)
    setter=Libdl.dlsym(native, :LAPACKE_set_nancheck)
    old=ccall(getter, Cint, ())
    ccall(setter, Cvoid, (Cint,), 1)
    try
        for (T, prefix) in
            ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
            J in (Int32, Int64),
            layout in (101, 102)

            suffix=J==Int64 ? "64_" : ""
            v=zeros(T, 128)
            t=zeros(T, 128)
            c=ones(T, 128)
            w=fill(T(779), 32)
            name="LAPACKE_"*prefix*"gemqrt"
            hp=Libdl.dlsym(native, name*suffix)
            wp=Libdl.dlsym(native, name*"_work"*suffix)
            low(l, s, tr, m, n, k, nb, lv, lt, lc) = lapacke288_compact_apply(
                T,
                J,
                wp,
                false,
                l,
                s,
                tr,
                m,
                n,
                k,
                nb,
                v,
                lv,
                t,
                lt,
                c,
                lc,
                w,
                name*"_work",
                :dll,
            )
            @test low(0, 'L', 'N', 0, 0, 0, 1, 1, 1, 1)==-1
            @test low(102, 'X', 'N', 0, 0, 0, 1, 1, 1, 1)==-2
            @test low(102, 'L', 'X', 0, 0, 0, 1, 1, 1, 1)==-3
            @test low(102, 'L', 'N', -1, 0, 0, 1, 1, 1, 1)==-4
            @test low(102, 'L', 'N', 0, -1, 0, 1, 1, 1, 1)==-5
            @test low(102, 'L', 'N', 2, 2, 3, 1, 2, 1, 2)==-6
            @test low(102, 'L', 'N', 2, 2, 1, 0, 2, 1, 2)==-7
            @test low(102, 'L', 'N', 2, 2, 1, 2, 2, 2, 2)==-7
            @test low(102, 'L', 'N', 2, 2, 1, 1, 1, 1, 2)==-9
            @test low(102, 'L', 'N', 2, 2, 1, 1, 2, 0, 2)==-11
            @test low(102, 'L', 'N', 2, 2, 1, 1, 2, 1, 1)==-13
            @test low(101, 'X', 'X', 3, 5, 2, 1, 0, 0, 0)==-13
            @test low(101, 'X', 'X', 3, 5, 2, 1, 0, 0, 5)==-11
            @test low(101, 'X', 'X', 3, 5, 2, 1, 0, 2, 5)==-9
            @test low(101, 'X', 'X', 3, 5, 2, 1, 2, 2, 5)==-2
            if J==Int64
                @test low(102, 'L', 'N', -4294967293, 0, 0, 1, 1, 1, 1)==-4
                @test low(102, 'L', 'N', 0, 4294967296, 0, 1, 1, 1, 1)==0
            end
            for side in ('L', 'R'), (m, n) in ((3, 5), (5, 3))
                order=side=='L' ? m : n
                k=2
                nb=1
                lv=layout==101 ? k+2 : order+2
                lt=layout==101 ? k+2 : nb+2
                lc=layout==101 ? n+2 : m+2
                vi(i, j) = layout==101 ? (i-1)*lv+j : i+(j-1)*lv
                ti(i, j) = layout==101 ? (i-1)*lt+j : i+(j-1)*lt
                ci(i, j) = layout==101 ? (i-1)*lc+j : i+(j-1)*lc
                fill!(v, T(NaN))
                fill!(t, T(NaN))
                fill!(c, T(NaN))
                for i in 1:order, j in 1:k
                    v[vi(i, j)]=0
                end
                for i in 1:nb, j in 1:k
                    t[ti(i, j)]=0
                end
                for i in 1:m, j in 1:n
                    c[ci(i, j)]=1
                end
                high() = lapacke288_compact_apply(
                    T,
                    J,
                    hp,
                    true,
                    layout,
                    side,
                    'N',
                    m,
                    n,
                    k,
                    nb,
                    v,
                    lv,
                    t,
                    lt,
                    c,
                    lc,
                    w,
                    name,
                    :dll,
                )
                sv, st, sc=copy(v), copy(t), copy(c)
                @test high()==0
                @test isequal(v, sv)&&isequal(t, st)&&isequal(c, sc)
                v[vi(order, k)]=T(NaN)
                t[ti(nb, k)]=T(NaN)
                c[ci(m, n)]=T(NaN)
                poisoned=copy(c)
                @test high()==-12
                @test isequal(c, poisoned)
                c[ci(m, n)]=1
                @test high()==-10
                t[ti(nb, k)]=0
                @test high()==-8
                v[vi(order, k)]=0
                @test high()==0
            end
            for variant in 1:3
                family=("geqrt", "geqrt2", "geqrt3")[variant]
                fn="LAPACKE_"*prefix*family
                hp2=Libdl.dlsym(native, fn*suffix)
                wp2=Libdl.dlsym(native, fn*"_work"*suffix)
                call(high, l, m, n, nb, la, lt) = lapacke288_compact_factor(
                    T,
                    J,
                    high ? hp2 : wp2,
                    variant,
                    high,
                    l,
                    m,
                    n,
                    nb,
                    v,
                    la,
                    t,
                    lt,
                    w,
                    fn*(high ? "" : "_work"),
                    :dll,
                )
                @test call(false, 0, 0, 0, 1, 1, 1)==-1
                @test call(false, 102, 0, -1, 1, 1, 1)==-3
                @test call(false, 102, -1, 0, 1, 1, 1)==-2
                @test call(false, 101, -1, 3, 1, 2, 0)==(variant==1 ? -6 : -5)
                @test call(false, 101, 3, 2, 1, 2, 1)==(variant==1 ? -8 : -7)
                fill!(v, zero(T))
                v[1]=T(NaN)
                sv, st=copy(v), copy(t)
                @test call(true, 102, 1, 1, 1, 1, 1)==(variant==1 ? -5 : -4)
                @test isequal(v, sv)&&isequal(t, st)
            end
        end
    finally
        ccall(setter, Cvoid, (Cint,), old)
    end
end
