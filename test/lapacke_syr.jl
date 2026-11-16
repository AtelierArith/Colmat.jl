using Test, Colmat, LinearAlgebra, Libdl
isdefined(@__MODULE__, :lapacke288_selected_entry) || include("lapacke_calls.jl")

for T in (ComplexF32, ComplexF64), J in (Int32, Int64)
    @eval function lapacke288_syr_call(
        ::Type{$J},
        pointer,
        layout,
        uplo,
        n,
        alpha::$T,
        x::Vector{$T},
        incx,
        a::Vector{$T},
        lda,
        name,
        route,
    )
        lapacke288_selected_entry(name, $J, route) do
            ccall(
                pointer,
                $J,
                (Cint, Cchar, $J, $T, Ptr{$T}, $J, Ptr{$T}, $J),
                layout,
                uplo,
                n,
                alpha,
                x,
                incx,
                a,
                lda,
            )
        end
    end
end
for T in (ComplexF32, ComplexF64)
    @eval function lapacke288_syr_reference(
        pointer, uplo, n, alpha::$T, x::Vector{$T}, incx, a::Matrix{$T}
    )
        ccall(
            pointer,
            Cvoid,
            (Ref{UInt8}, Ref{Int64}, Ref{$T}, Ptr{$T}, Ref{Int64}, Ptr{$T}, Ref{Int64}),
            UInt8(uplo),
            n,
            alpha,
            x,
            incx,
            a,
            max(1, n),
        )
    end
end

function lapacke288_syr(::Type{T}, ::Type{J}) where {T,J}
    prefix=T===ComplexF32 ? "c" : "z"
    native=Libdl.dlopen(Colmat.backend_library())
    trampoline=Libdl.dlopen(BLAS.libblastrampoline)
    reference=only(
        filter(
            l->l.interface==:ilp64 && occursin("openblas", lowercase(l.libname)),
            BLAS.get_config().loaded_libs,
        ),
    )
    reference_pointer=Libdl.dlsym(Libdl.dlopen(reference.libname), prefix*"syr_64_")
    @test reference_pointer!=Libdl.dlsym(native, prefix*"syr_64_")
    suffix=J===Int64 ? "64_" : ""
    interface=J===Int64 ? :ilp64 : :lp64
    for work in (false, true),
        route in (:dll, :lbt_trampoline),
        layout in (101, 102),
        uplo in ('L', 'U', 'l', 'u'),
        n in (0, 1, 3),
        incx in (-3, -1, 1, 2)

        name="LAPACKE_"*prefix*"syr"*(work ? "_work" : "")
        @test BLAS.lbt_get_forward(name, interface)==Libdl.dlsym(native, name*suffix)
        pointer=Libdl.dlsym(route===:dll ? native : trampoline, name*suffix)
        selected(i, j) = uppercase(uplo)=='L' ? i>=j : i<=j
        ld=max(1, n)+2
        index(i, j) = layout==101 ? (i-1)*ld+j : i+(j-1)*ld
        a=fill(T(NaN), ld*max(1, n)+3)
        external=fill(T(NaN), n, n)
        for i in 1:n, j in 1:n
            if selected(i, j)
                a[index(i, j)]=T((2+i+3j)/7, (i-2j)/11)
                external[i, j]=a[index(i, j)]
            end
        end
        logical=T[T((i+2)/3, (2i-1)/7) for i in 1:n]
        x=fill(T(NaN), 1+max(0, n-1)*abs(incx))
        first=incx<0 ? 1+(n-1)*(-incx) : 1
        for i in 1:n
            x[first + (i - 1) * incx]=logical[i]
        end
        alpha=T(0.7, -0.4)
        saved_a, saved_x=copy(a), copy(x)
        lapacke288_syr_reference(reference_pointer, uplo, n, alpha, x, incx, external)
        @test lapacke288_syr_call(
            J, pointer, layout, uplo, n, alpha, x, incx, a, ld, name, route
        )==0
        @test isequal(x, saved_x)
        used=Set(index(i, j) for i in 1:n for j in 1:n if selected(i, j))
        @test all(isequal(a[q], saved_a[q]) for q in eachindex(a) if q∉used)
        for i in 1:n, j in 1:n
            if selected(i, j)
                expected=saved_a[index(i, j)]+alpha*logical[i]*logical[j]
                @test isapprox(
                    a[index(i, j)], expected; rtol=20eps(real(T)), atol=20eps(real(T))
                )
                @test isapprox(
                    a[index(i, j)], external[i, j]; rtol=20eps(real(T)), atol=20eps(real(T))
                )
            end
        end
    end
end

@testset "LAPACKE complex SYR coefficient and strided triangle storage" begin
    for T in (ComplexF32, ComplexF64), J in (Int32, Int64)
        lapacke288_syr(T, J)
    end
end
