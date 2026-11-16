using Test, Colmat, LinearAlgebra, Libdl

isdefined(@__MODULE__, :lapacke288_selected_entry) || include("lapacke_calls.jl")
for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
    @eval function lapacke288_po_single(
        ::Type{$J}, pointer, layout, uplo, a::Vector{$T}, name, route
    )
        lapacke288_selected_entry(name, $J, route) do
            ccall(pointer, $J, (Cint, Cchar, $J, Ptr{$T}, $J), layout, uplo, 3, a, 5)
        end
    end
    @eval function lapacke288_po_rhs(
        ::Type{$J}, pointer, layout, uplo, a::Vector{$T}, b::Vector{$T}, name, route, nrhs=2
    )
        lapacke288_selected_entry(name, $J, route) do
            ccall(
                pointer,
                $J,
                (Cint, Cchar, $J, $J, Ptr{$T}, $J, Ptr{$T}, $J),
                layout,
                uplo,
                3,
                nrhs,
                a,
                5,
                b,
                5,
            )
        end
    end
end
for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
    @eval function lapacke288_po_fortran(::Type{$J}, pointer, uplo, a::Vector{$T})
        info=Ref{$J}(-99)
        ccall(
            pointer,
            Cvoid,
            (Ref{UInt8}, Ref{$J}, Ptr{$T}, Ref{$J}, Ref{$J}),
            UInt8(uplo),
            3,
            a,
            5,
            info,
        )
        return info[]
    end
end

for T in (Float32, Float64, ComplexF32, ComplexF64)
    @eval function lapacke288_po_reference_factor(pointer, uplo, a::Matrix{$T})
        info=Ref{Int64}(-99)
        ccall(
            pointer,
            Cvoid,
            (Ref{UInt8}, Ref{Int64}, Ptr{$T}, Ref{Int64}, Ref{Int64}),
            UInt8(uplo),
            3,
            a,
            3,
            info,
        )
        @test info[]==0
    end
    @eval function lapacke288_po_external_solve(pointer, uplo, a::Matrix{$T}, b::Matrix{$T})
        info=Ref{Int64}(-99)
        ccall(
            pointer,
            Cvoid,
            (
                Ref{UInt8},
                Ref{Int64},
                Ref{Int64},
                Ptr{$T},
                Ref{Int64},
                Ptr{$T},
                Ref{Int64},
                Ref{Int64},
            ),
            UInt8(uplo),
            3,
            2,
            a,
            3,
            b,
            3,
            info,
        )
        @test info[]==0
    end
end

function lapacke288_cholesky(::Type{T}, ::Type{J}) where {T,J}
    prefix=Dict(Float32=>"s", Float64=>"d", ComplexF32=>"c", ComplexF64=>"z")[T]
    native=Libdl.dlopen(Colmat.backend_library())
    trampoline=Libdl.dlopen(BLAS.libblastrampoline)
    reference=only(
        filter(
            l -> l.interface==:ilp64 && occursin("openblas", lowercase(l.libname)),
            BLAS.get_config().loaded_libs,
        ),
    )
    reference_handle=Libdl.dlopen(reference.libname)
    factor_pointer=Libdl.dlsym(reference_handle, prefix*"potrf_64_")
    solve_pointer=Libdl.dlsym(reference_handle, prefix*"potrs_64_")
    l=T[2 0 0; 1 3 0; 2 -1 4]
    expected=T[1 2; 3 -1; 2 4]
    if T<:Complex
        l[2, 1]+=T(0, 0.7)
        l[3, 1]+=T(0, -0.4)
        l[3, 2]+=T(0, 0.8)
        for i in 1:3, j in 1:2
            expected[i, j]+=T(0, (i+j)/5)
        end
    end
    original=zeros(T, 3, 3)
    rhs=zeros(T, 3, 2)
    for i in 1:3, j in 1:3, k in 1:3
        original[i, j]+=l[i, k]*conj(l[j, k])
    end
    for i in 1:3, j in 1:2, k in 1:3
        rhs[i, j]+=original[i, k]*expected[k, j]
    end
    suffix=J===Int64 ? "64_" : ""
    interface=J===Int64 ? :ilp64 : :lp64
    for uplo in ('L', 'U', 'l', 'u')
        selected(i, j) = uppercase(uplo)=='L' ? i>=j : i<=j
        reference_factor=fill(T(NaN), 3, 3)
        for i in 1:3, j in 1:3
            if selected(i, j)
                reference_factor[i, j]=original[i, j]
            end
        end
        lapacke288_po_reference_factor(factor_pointer, uplo, reference_factor)
        # Direct Fortran callers also own the unselected triangle and padding.
        for routine in ("potrf", "potf2", "potrf2")
            index(i, j) = i+(j-1)*5
            a=fill(T(NaN), 15)
            for i in 1:3, j in 1:3
                if selected(i, j)
                    a[index(i, j)]=original[i, j]
                end
            end
            if T<:Complex
                for i in 1:3
                    a[index(i, i)]=T(real(a[index(i, i)]), NaN)
                end
            end
            saved_a=copy(a)
            fortran_suffix=J===Int64 ? "_64_" : "_"
            pointer=Libdl.dlsym(native, prefix*routine*fortran_suffix)
            @test lapacke288_po_fortran(J, pointer, uplo, a)==0
            used=Set(index(i, j) for i in 1:3 for j in 1:3 if selected(i, j))
            @test all(isequal(a[q], saved_a[q]) for q in eachindex(a) if q∉used)
            factor=fill(T(NaN), 3, 3)
            for i in 1:3, j in 1:3
                if selected(i, j)
                    factor[i, j]=a[index(i, j)]
                end
            end
            second_rhs=copy(rhs)
            lapacke288_po_external_solve(solve_pointer, uplo, factor, second_rhs)
            @test isapprox(second_rhs, expected; rtol=200eps(real(T)), atol=200eps(real(T)))
        end
        for routine in ("potrf", "potrf2", "potri", "potrs", "posv"),
            work in (false, true), route in (:dll, :lbt_trampoline),
            layout in (101, 102)

            name="LAPACKE_"*prefix*routine*(work ? "_work" : "")
            @test BLAS.lbt_get_forward(name, interface)==Libdl.dlsym(native, name*suffix)
            pointer=Libdl.dlsym(route===:dll ? native : trampoline, name*suffix)
            index(i, j) = layout==101 ? (i-1)*5+j : i+(j-1)*5
            a=fill(T(NaN), 15)
            b=fill(T(NaN), 15)
            input=routine in ("potri", "potrs") ? reference_factor : original
            for i in 1:3, j in 1:3
                if selected(i, j)
                    a[index(i, j)]=input[i, j]
                end
            end
            for i in 1:3, j in 1:2
                b[index(i, j)]=rhs[i, j]
            end
            if work && T<:Complex && routine in ("potrf", "potrf2", "posv")
                for i in 1:3
                    a[index(i, i)]=T(real(a[index(i, i)]), NaN)
                end
            end
            saved_a, saved_b=copy(a), copy(b)
            info=if routine in ("potrs", "posv")
                lapacke288_po_rhs(J, pointer, layout, uplo, a, b, name, route)
            else
                lapacke288_po_single(J, pointer, layout, uplo, a, name, route)
            end
            @test info==0
            selected_indices=Set(index(i, j) for i in 1:3 for j in 1:3 if selected(i, j))
            @test all(isequal(a[q], saved_a[q]) for q in eachindex(a) if q∉selected_indices)
            used_rhs=Set(index(i, j) for i in 1:3 for j in 1:2)
            @test all(isequal(b[q], saved_b[q]) for q in eachindex(b) if q∉used_rhs)
            if routine in ("potrs", "posv")
                for i in 1:3, j in 1:2
                    @test isapprox(
                        b[index(i, j)],
                        expected[i, j];
                        rtol=200eps(real(T)),
                        atol=200eps(real(T)),
                    )
                end
            end
            if routine=="potrs"
                @test isequal(a, saved_a)
            elseif routine=="potri"
                inverse(i, j) = selected(i, j) ? a[index(i, j)] : conj(a[index(j, i)])
                for i in 1:3, j in 1:3
                    left, right=zero(T), zero(T)
                    for k in 1:3
                        left+=original[i, k]*inverse(k, j)
                        right+=inverse(i, k)*original[k, j]
                    end
                    @test isapprox(
                        left, T(i==j); rtol=200eps(real(T)), atol=200eps(real(T))
                    )
                    @test isapprox(
                        right, T(i==j); rtol=200eps(real(T)), atol=200eps(real(T))
                    )
                end
            else
                factor(i, j) = selected(i, j) ? a[index(i, j)] : zero(T)
                for i in 1:3, j in 1:3
                    value=zero(T)
                    for k in 1:3
                        value+=if uppercase(uplo)=='L'
                            factor(i, k)*conj(factor(j, k))
                        else
                            conj(factor(k, i))*factor(k, j)
                        end
                    end
                    @test isapprox(
                        value, original[i, j]; rtol=200eps(real(T)), atol=200eps(real(T))
                    )
                end
                # External standard POTRS consumes each produced factor.
                outgoing=fill(T(NaN), 3, 3)
                for i in 1:3, j in 1:3
                    if selected(i, j)
                        outgoing[i, j]=a[index(i, j)]
                    end
                end
                second_rhs=copy(rhs)
                lapacke288_po_external_solve(solve_pointer, uplo, outgoing, second_rhs)
                @test isapprox(
                    second_rhs, expected; rtol=200eps(real(T)), atol=200eps(real(T))
                )
            end
            if routine=="posv"
                zero_a=copy(saved_a)
                @test lapacke288_po_rhs(
                    J, pointer, layout, uplo, zero_a, T[], name, route, 0
                )==0
                @test isequal(zero_a, a)
                bad=copy(saved_a)
                bad[index(1, 1)]=T(-1)
                failed_rhs=copy(saved_b)
                @test lapacke288_po_rhs(
                    J, pointer, layout, uplo, bad, failed_rhs, name, route
                )==1
                @test isequal(failed_rhs, saved_b)
            elseif routine in ("potrf", "potrf2")
                bad=copy(saved_a)
                bad[index(1, 1)]=T(-1)
                @test lapacke288_po_single(J, pointer, layout, uplo, bad, name, route)==1
            elseif routine=="potri"
                bad=copy(saved_a)
                bad[index(2, 2)]=zero(T)
                @test lapacke288_po_single(J, pointer, layout, uplo, bad, name, route)==2
                @test isequal(
                    bad,
                    begin
                        x=copy(saved_a)
                        x[index(2, 2)]=zero(T)
                        x
                    end,
                )
            end
        end
    end
end

@testset "LAPACKE Cholesky triangle storage and interoperability" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
        lapacke288_cholesky(T, J)
    end
end
