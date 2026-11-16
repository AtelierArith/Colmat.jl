using Test, Colmat, LinearAlgebra, Libdl
isdefined(@__MODULE__, :lapacke288_selected_entry)||include("lapacke_calls.jl")
for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
    @eval function lapacke288_lauum_call(
        ::Type{$T}, ::Type{$J}, p, layout, u, n, a, ld, name, route
    )
        lapacke288_selected_entry(name, $J, route) do
            ccall(p, $J, (Cint, Cchar, $J, Ptr{$T}, $J), layout, u, n, a, ld)
        end
    end
end
for T in (Float32, Float64, ComplexF32, ComplexF64)
    @eval function lapacke288_lauum_reference(::Type{$T}, p, u, n, a)
        info=Ref{Int64}(-777)
        ccall(
            p,
            Cvoid,
            (Ref{UInt8}, Ref{Int64}, Ptr{$T}, Ref{Int64}, Ptr{Int64}, Clong),
            u,
            n,
            a,
            size(a, 1),
            info,
            1,
        )
        @test info[]==0
        a
    end
end
@testset "LAPACKE LAUUM native products and complex diagonal compatibility" begin
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
        J in (Int32, Int64), n in (0, 1, 3, 8, 80), u in ('L', 'U'),
        mode in (T<:Complex ? (:real, :nonreal, :nan) : (:real,))

        R=real(T)
        tol=R(500)*eps(R)
        ld=n+2
        original=zeros(T, n, n)
        stored=fill(T(NaN), ld, n)
        selected(i, j) = u=='L' ? i>=j : i<=j
        for i in 1:n, j in 1:n
            if selected(i, j)
                value=if i==j
                    if mode==:real
                        T(2+i/max(1, n))
                    else
                        T(complex(2+i/max(1, n), mode==:nan ? NaN : 0.75))
                    end
                else
                    T<:Complex ? T(complex(cos(i+j)/10, sin(i-j)/10)) : T(cos(i+j)/10)
                end
                original[i, j]=stored[i, j]=value
            end
        end
        refp=Libdl.dlsym(reference, prefix*"lauum_64_")
        @test refp!=Libdl.dlsym(native, prefix*"lauum_64_")
        expected_column=lapacke288_lauum_reference(T, refp, u, n, copy(stored))
        # Reference LAPACKE's row adapter uses compact column storage. Even the
        # independent BLAS may propagate used NaNs differently for other LDA.
        expected_row=lapacke288_lauum_reference(T, refp, u, n, copy(stored[1:max(1, n), :]))
        product=mode==:real ? (u=='U' ? original*original' : original'*original) : nothing
        for route in (:dll, :lbt_trampoline), layout in (101, 102), high in (false, true)
            expected=layout==101 ? expected_row : expected_column
            name="LAPACKE_"*prefix*"lauum"*(high ? "" : "_work")
            suffix=J==Int64 ? "64_" : ""
            @test BLAS.lbt_get_forward(name, J==Int64 ? :ilp64 : :lp64)==Libdl.dlsym(
                native, name*suffix
            )
            p=Libdl.dlsym(route==:dll ? native : trampoline, name*suffix)
            at(i, j) = layout==101 ? (i-1)*ld+j : i+(j-1)*ld
            a=fill(T(NaN), ld*max(1, n))
            mask=falses(length(a))
            for i in 1:n, j in 1:n
                if selected(i, j)
                    a[at(i, j)]=stored[i, j]
                    mask[at(i, j)]=true
                end
            end
            before=copy(a)
            info=lapacke288_lauum_call(T, J, p, layout, lowercase(u), n, a, ld, name, route)
            if high&&mode==:nan&&n>0
                @test info==-4 && isequal(a, before)
                continue
            end
            @test info==0
            @test isequal(a[.!mask], before[.!mask])
            for i in 1:n, j in 1:n
                if selected(i, j)
                    @test isapprox(
                        real(a[at(i, j)]),
                        real(expected[i, j]);
                        rtol=tol,
                        atol=tol,
                        nans=true,
                    )
                    @test isapprox(
                        imag(a[at(i, j)]),
                        imag(expected[i, j]);
                        rtol=tol,
                        atol=tol,
                        nans=true,
                    )
                    if mode==:real
                        @test a[at(i, j)]≈product[i, j] rtol=tol atol=tol
                    end
                end
            end
        end
    end
end
