using Test, Colmat, LinearAlgebra, Libdl
isdefined(@__MODULE__, :lapacke288_selected_entry)||include("lapacke_calls.jl")
for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
    @eval function lapacke288_trtri(
        ::Type{$T}, ::Type{$J}, p, layout, u, d, n, a, la, name, route
    )
        lapacke288_selected_entry(name, $J, route) do
            ccall(p, $J, (Cint, Cchar, Cchar, $J, Ptr{$T}, $J), layout, u, d, n, a, la)
        end
    end
    @eval function lapacke288_trtrs(
        ::Type{$T}, ::Type{$J}, p, layout, u, t, d, n, k, a, la, b, lb, name, route
    )
        lapacke288_selected_entry(name, $J, route) do
            ccall(
                p,
                $J,
                (Cint, Cchar, Cchar, Cchar, $J, $J, Ptr{$T}, $J, Ptr{$T}, $J),
                layout,
                u,
                t,
                d,
                n,
                k,
                a,
                la,
                b,
                lb,
            )
        end
    end
end
for T in (Float32, Float64, ComplexF32, ComplexF64)
    @eval function lapacke288_tri_reference(::Type{$T}, pinv, psolve, u, t, d, n, k, a, b)
        info=Ref{Int64}(-777)
        ai=copy(a)
        rhs=copy(b)
        ld=size(a, 1)
        lb=size(b, 1)
        ccall(
            pinv,
            Cvoid,
            (
                Ref{UInt8},
                Ref{UInt8},
                Ref{Int64},
                Ptr{$T},
                Ref{Int64},
                Ptr{Int64},
                Clong,
                Clong,
            ),
            u,
            d,
            n,
            ai,
            ld,
            info,
            1,
            1,
        )
        @test info[]==0
        ccall(
            psolve,
            Cvoid,
            (
                Ref{UInt8},
                Ref{UInt8},
                Ref{UInt8},
                Ref{Int64},
                Ref{Int64},
                Ptr{$T},
                Ref{Int64},
                Ptr{$T},
                Ref{Int64},
                Ptr{Int64},
                Clong,
                Clong,
                Clong,
            ),
            u,
            t,
            d,
            n,
            k,
            a,
            ld,
            rhs,
            lb,
            info,
            1,
            1,
            1,
        )
        @test info[]==0
        return ai, rhs
    end
end
@testset "LAPACKE triangular selected storage and independent factors" begin
    native=Libdl.dlopen(Colmat.backend_library())
    trampoline=Libdl.dlopen(BLAS.libblastrampoline)
    reference=Libdl.dlopen(
        only(
            filter(
                l->l.interface==:ilp64&&occursin("openblas", lowercase(l.libname)),
                BLAS.get_config().loaded_libs,
            ),
        ).libname,
    )
    for (T, prefix) in
        ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
        J in (Int32, Int64),
        n in (0, 1, 3, 5),
        k in (0, 2),
        u in ('L', 'U'),
        d in ('N', 'U'),
        t in ('N', 'T', 'C')

        R=real(T)
        tol=R(200)*eps(R)
        ld=n+2
        original=zeros(T, n, n)
        stored=fill(T(NaN), ld, n)
        selected(i, j) = (u=='L' ? i>=j : i<=j)&&(d!='U'||i!=j)
        for i in 1:n, j in 1:n
            if u=='L' ? i>=j : i<=j
                original[i, j]=if i==j
                    T(d=='U' ? 1 : i+1)
                elseif T<:Complex
                    T(complex((i+j)/10, (i-j)/20))
                else
                    T((i+j)/10)
                end
                selected(i, j)&&(stored[i, j]=original[i, j])
            end
        end
        solution=T[(i+j)/3 for i in 1:n, j in 1:k]
        T<:Complex && (solution .+= T(im)/7)
        rhs=(
            if t=='N'
                original
            elseif t=='T'
                transpose(original)
            else
                original'
            end
        )*solution
        stored_b=fill(T(NaN), ld, k)
        stored_b[1:n, :].=rhs
        pinv=Libdl.dlsym(reference, prefix*"trtri_64_")
        psolve=Libdl.dlsym(reference, prefix*"trtrs_64_")
        @test pinv!=Libdl.dlsym(native, prefix*"trtri_64_") &&
            psolve!=Libdl.dlsym(native, prefix*"trtrs_64_")
        expected_inv, expected_b=lapacke288_tri_reference(
            T, pinv, psolve, u, t, d, n, k, stored, stored_b
        )
        for route in (:dll, :lbt_trampoline), layout in (101, 102), high in (false, true)
            suffix=J==Int64 ? "64_" : ""
            ending=high ? "" : "_work"
            names=("LAPACKE_"*prefix*"trtri"*ending, "LAPACKE_"*prefix*"trtrs"*ending)
            pointers=map(names) do name
                @test BLAS.lbt_get_forward(name, J==Int64 ? :ilp64 : :lp64)==Libdl.dlsym(
                    native, name*suffix
                )
                Libdl.dlsym(route==:dll ? native : trampoline, name*suffix)
            end
            at(i, j) = layout==101 ? (i-1)*ld+j : i+(j-1)*ld
            a=fill(T(NaN), ld*max(1, n))
            mask=falses(length(a))
            for i in 1:n, j in 1:n
                if selected(i, j)
                    a[at(i, j)]=stored[i, j]
                    mask[at(i, j)]=true
                end
            end
            saved=copy(a)
            @test lapacke288_trtri(
                T,
                J,
                pointers[1],
                layout,
                lowercase(u),
                lowercase(d),
                n,
                a,
                ld,
                names[1],
                route,
            )==0
            inverse=Matrix{T}(I, n, n)
            for i in 1:n, j in 1:n
                if selected(i, j)
                    @test a[at(i, j)]≈expected_inv[i, j] rtol=tol atol=tol
                    inverse[i, j]=a[at(i, j)]
                elseif i!=j
                    inverse[i, j]=0
                end
            end
            @test original*inverse≈Matrix{T}(I, n, n) rtol=tol atol=tol
            @test isequal(a[.!mask], saved[.!mask])
            a=copy(saved)
            lb=layout==101 ? k+2 : n+2
            bt(i, j) = layout==101 ? (i-1)*lb+j : i+(j-1)*lb
            b=fill(T(NaN), layout==101 ? max(1, n)*lb : max(1, k)*lb)
            bmask=falses(length(b))
            for i in 1:n, j in 1:k
                b[bt(i, j)]=rhs[i, j]
                bmask[bt(i, j)]=true
            end
            before=copy(b)
            @test lapacke288_trtrs(
                T,
                J,
                pointers[2],
                layout,
                lowercase(u),
                lowercase(t),
                lowercase(d),
                n,
                k,
                a,
                ld,
                b,
                lb,
                names[2],
                route,
            )==0
            @test isequal(a, saved)
            @test isequal(b[.!bmask], before[.!bmask])
            for i in 1:n, j in 1:k
                @test b[bt(i, j)]≈expected_b[i, j] rtol=tol atol=tol
                @test b[bt(i, j)]≈solution[i, j] rtol=tol atol=tol
            end
        end
    end
end
