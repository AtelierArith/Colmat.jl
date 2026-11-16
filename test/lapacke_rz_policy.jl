using Test, Colmat, LinearAlgebra, Libdl
isdefined(@__MODULE__, :lapacke288_rz_apply)||include("lapacke_rz.jl")
@testset "RZ C diagnostics, sparse reads and logical NaN policy" begin
    native=Libdl.dlopen(Colmat.backend_library())
    getter=Libdl.dlsym(native, :LAPACKE_get_nancheck)
    setter=Libdl.dlsym(native, :LAPACKE_set_nancheck)
    error=Libdl.dlsym(native, :colmat_blas_last_error)
    old=ccall(getter, Cint, ())
    ccall(setter, Cvoid, (Cint,), 1)
    try
        for (T, prefix) in
            ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
            J in (Int32, Int64),
            layout in (101, 102)

            suffix=J==Int64 ? "64_" : ""
            name="LAPACKE_"*prefix*"tzrzf"
            hp=Libdl.dlsym(native, name*suffix)
            wp=Libdl.dlsym(native, name*"_work"*suffix)
            a=zeros(T, 64)
            tau=fill(T(778), 8)
            work=fill(T(779), 16)
            factor(l, m, n, la, lw) = lapacke288_rz_factor(
                T, J, wp, false, l, m, n, a, la, tau, work, lw, name*"_work", :dll
            )
            @test factor(0, 0, 0, 1, -1)==-1
            @test factor(102, -1, 0, 1, -1)==-2
            @test factor(102, 3, 2, 3, -1)==-3
            @test factor(102, 3, 4, 2, -1)==-5
            @test factor(101, -1, 4, 3, -1)==-5
            saved, st=copy(a), copy(tau)
            @test factor(layout, 3, 3, 5, 0)==-8
            @test ccall(error, Cint, ())==7
            @test work[1]==one(T)&&a==saved&&tau==st
            @test factor(layout, 3, 3, 5, 1)==0
            @test work[1]==one(T)&&a==saved&&tau[1:3]==zeros(T, 3)&&tau[4:end]==st[4:end]
            if J==Int64
                @test factor(102, 0, 4294967296, 1, -1)==0
                @test work[1]==one(T)
                @test factor(102, -4294967293, 0, 1, -1)==-2
            end
            family=T<:Complex ? "unmrz" : "ormrz"
            an="LAPACKE_"*prefix*family
            ah=Libdl.dlsym(native, an*suffix)
            aw=Libdl.dlsym(native, an*"_work"*suffix)
            c=ones(T, 64)
            fill!(tau, zero(T))
            low(l, s, tr, m, n, k, ll, la, lc, lw) = lapacke288_rz_apply(
                T,
                J,
                aw,
                false,
                l,
                s,
                tr,
                m,
                n,
                k,
                ll,
                a,
                la,
                tau,
                c,
                lc,
                work,
                lw,
                an*"_work",
                :dll,
            )
            @test low(0, 'L', 'N', 0, 0, 0, 0, 1, 1, -1)==-1
            @test low(102, 'X', 'N', 0, 0, 0, 0, 1, 1, -1)==-2
            @test low(102, 'L', 'X', 0, 0, 0, 0, 1, 1, -1)==-3
            @test low(102, 'L', 'N', -1, 0, 0, 0, 1, 1, -1)==-4
            @test low(102, 'L', 'N', 0, -1, 0, 0, 1, 1, -1)==-5
            @test low(102, 'L', 'N', 2, 3, 3, 0, 4, 4, -1)==-6
            @test low(102, 'L', 'N', 2, 3, 1, 3, 4, 4, -1)==-7
            @test low(102, 'L', 'N', 2, 3, 1, 1, 0, 4, -1)==-9
            @test low(102, 'L', 'N', 2, 3, 1, 1, 4, 1, -1)==-12
            @test low(102, 'L', 'N', 2, 3, 1, 1, 4, 4, 2)==-14
            @test low(101, 'R', 'N', 3, 5, 1, 1, 4, 7, -1)==-9
            @test low(101, 'R', 'N', 5, 3, 1, 1, 3, 2, -1)==-12
            # Invalid K must be diagnosed without dereferencing null TAU.
            @test lapacke288_rz_apply(
                T,
                J,
                ah,
                true,
                102,
                'L',
                'N',
                1,
                1,
                2,
                0,
                C_NULL,
                2,
                C_NULL,
                C_NULL,
                1,
                work,
                -1,
                an,
                :dll,
            )==-6
            # Identity reflectors never read A or C, even in row conversion.
            @test lapacke288_rz_apply(
                T,
                J,
                aw,
                false,
                layout,
                'L',
                'N',
                3,
                5,
                1,
                0,
                C_NULL,
                8,
                tau,
                C_NULL,
                8,
                work,
                5,
                an*"_work",
                :dll,
            )==0
            if J==Int64
                @test low(102, 'L', 'N', 0, 4294967296, 0, 0, 1, 1, -1)==0
                @test work[1]==one(T)
                @test low(102, 'L', 'N', -4294967293, 0, 0, 0, 1, 1, -1)==-4
            end
            for side in ('L', 'R'), (m, n) in ((3, 5), (5, 3))
                order=side=='L' ? m : n
                k=1
                ll=1
                la=layout==101 ? order+2 : k+2
                lc=layout==101 ? n+2 : m+2
                ai(i, j) = layout==101 ? (i-1)*la+j : i+(j-1)*la
                ci(i, j) = layout==101 ? (i-1)*lc+j : i+(j-1)*lc
                fill!(a, T(NaN))
                fill!(c, T(NaN))
                fill!(tau, zero(T))
                for j in 1:order
                    a[ai(1, j)]=0
                end
                for i in 1:m, j in 1:n
                    c[ci(i, j)]=1
                end
                high() = lapacke288_rz_apply(
                    T,
                    J,
                    ah,
                    true,
                    layout,
                    side,
                    'N',
                    m,
                    n,
                    k,
                    ll,
                    a,
                    la,
                    tau,
                    c,
                    lc,
                    work,
                    8,
                    an,
                    :dll,
                )
                sa, sc=copy(a), copy(c)
                @test high()==0
                @test isequal(a, sa)&&isequal(c, sc)
                a[ai(1, order)]=T(NaN)
                c[ci(m, n)]=T(NaN)
                tau[1]=T(NaN)
                @test high()==-8
                a[ai(1, order)]=0
                @test high()==-11
                c[ci(m, n)]=1
                @test high()==-10
                tau[1]=0
                @test high()==0
            end
        end
    finally
        ccall(setter, Cvoid, (Cint,), old)
    end
end
