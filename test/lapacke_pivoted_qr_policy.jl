using Test, Colmat, LinearAlgebra, Libdl
isdefined(@__MODULE__, :lapacke288_pivot_call)||include("lapacke_pivoted_qr.jl")
@testset "Pivoted QR C diagnostics, NaN bounds and workspace publication" begin
    native=Libdl.dlopen(Colmat.backend_library())
    getter=Libdl.dlsym(native, :LAPACKE_get_nancheck)
    setter=Libdl.dlsym(native, :LAPACKE_set_nancheck)
    error=Libdl.dlsym(native, :colmat_blas_last_error)
    old=ccall(getter, Cint, ())
    ccall(setter, Cvoid, (Cint,), 1)
    try
        for (T, prefix) in
            ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
            J in (Int32, Int64), blocked in (false, true),
            layout in (101, 102)

            name="LAPACKE_"*prefix*(blocked ? "geqp3" : "geqpf")
            suffix=J==Int64 ? "64_" : ""
            hp=Libdl.dlsym(native, name*suffix)
            wp=Libdl.dlsym(native, name*"_work"*suffix)
            a=zeros(T, 32)
            pivot=fill(J(0), 4)
            tau=fill(T(778), 4)
            w=fill(T(779), 32)
            rw=fill(real(T)(780), 16)
            low(l, m, n, la, lw) = lapacke288_pivot_call(
                T,
                J,
                wp,
                blocked,
                false,
                l,
                m,
                n,
                a,
                la,
                pivot,
                tau,
                w,
                lw,
                rw,
                name*"_work",
                :dll,
            )
            @test low(0, 0, 0, 1, 16)==-1
            @test low(102, -1, 0, 1, 16)==-2
            @test low(102, 0, -1, 1, 16)==-3
            @test low(102, 3, 2, 2, 16)==-5
            @test low(101, -1, 3, 2, 16)==-5
            @test low(101, 2, 3, 2, 16)==-5
            if blocked
                a0, p0, t0, r0=copy(a), copy(pivot), copy(tau), copy(rw)
                fill!(w, T(779))
                @test low(layout, 2, 2, 4, 0)==-9
                @test ccall(error, Cint, ())==8
                @test isreal(w[1])&&real(w[1])>0&&all(==(T(779)), w[2:end])
                @test a==a0&&pivot==p0&&tau==t0&&rw==r0
                if J==Int64
                    @test low(102, 0, 4294967296, 1, -1)==0
                    @test w[1]==one(T)
                    @test low(102, -4294967293, 0, 1, -1)==-2
                end
            end
            ai(i, j) = layout==101 ? (i-1)*4+j : i+(j-1)*4
            fill!(a, T(NaN))
            for i in 1:2, j in 1:2
                a[ai(i, j)]=T(i==j ? 2 : 1)
            end
            high() = lapacke288_pivot_call(
                T,
                J,
                hp,
                blocked,
                true,
                layout,
                2,
                2,
                a,
                4,
                pivot,
                tau,
                w,
                16,
                rw,
                name,
                :dll,
            )
            padding=copy(a)
            p0, t0=copy(pivot), copy(tau)
            a[ai(2, 2)]=T(NaN)
            poisoned=copy(a)
            @test high()==-4
            @test isequal(a, poisoned)&&pivot==p0&&tau==t0
            a[ai(2, 2)]=padding[ai(2, 2)]
            @test high()==0
            @test sort(pivot[1:2])==[1, 2]
            @test tau[3:4]==fill(T(778), 2)
            for pos in eachindex(a)
                i=layout==101 ? (pos-1)÷4+1 : (pos-1)%4+1
                j=layout==101 ? (pos-1)%4+1 : (pos-1)÷4+1
                if i>2||j>2
                    @test isequal(a[pos], padding[pos])
                end
            end
        end
    finally
        ccall(setter, Cvoid, (Cint,), old)
    end
end
