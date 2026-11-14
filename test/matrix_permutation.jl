using Test, LinearAlgebra, Libdl, Colmat

for (T, p) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    J in (Int32, Int64)

    suffix = J === Int64 ? "64_" : ""
    @eval function direct_laswp!(
        a::Matrix{$T}, piv::Vector{$J}; n, k1, k2, incx, library=Colmat.backend_library()
    )
        f=Libdl.dlsym(Libdl.dlopen(library), Symbol($p * "laswp_" * $suffix))
        ccall(
            f,
            Cvoid,
            (Ref{$J}, Ptr{$T}, Ref{$J}, Ref{$J}, Ref{$J}, Ptr{$J}, Ref{$J}),
            n,
            a,
            size(a, 1),
            k1,
            k2,
            piv,
            incx,
        )
    end
end

@testset "LASWP pivot spacing and application order" begin
    reference = only(
        filter(
            l->l.interface==:ilp64 && occursin("openblas", lowercase(basename(l.libname))),
            BLAS.get_config().loaded_libs,
        ),
    ).libname
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        n in (0, 1, 3, 31, 32, 33),
        inc in (-3, -2, -1, 0, 1, 2, 3),
        first in (1, 2, 3)

        last=5
        lda=8
        a=T[T <: Complex ? T(i+10j, i-j) : T(i+10j) for i in 1:lda, j in 1:(n + 1)]
        a[1, 1]=T(-0.0)
        a[2, 1]=T(NaN)
        original=copy(a)
        expected=copy(a)
        piv=fill(J(-777), max(1, first+(last-first)*abs(inc)))
        if inc!=0
            for row in first:last
                piv[first + (row - first) * abs(inc)]=J(7-row)
            end
            for row in (inc>0 ? (first:last) : (last:-1:first))
                target=Int(piv[first + (row - first) * abs(inc)])
                for j in 1:n
                    expected[row, j], expected[target, j]=expected[target, j],
                    expected[row, j]
                end
            end
        end
        before=copy(piv)
        direct_laswp!(a, piv; n, k1=first, k2=last, incx=inc)
        @test isequal(a, expected)
        @test piv==before
        if J===Int64
            external=copy(original)
            direct_laswp!(external, piv; n, k1=first, k2=last, incx=inc, library=reference)
            @test isequal(a, external)
        end
        if inc!=0
            direct_laswp!(a, piv; n, k1=first, k2=last, incx=(-inc))
            @test isequal(a, original)
        end
    end
end

for (T, p) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    J in (Int32, Int64)

    suffix = J === Int64 ? "64_" : ""
    @eval function direct_lapm!(
        family,
        a::Matrix{$T},
        k::Vector{$J};
        m,
        n,
        forward,
        library=Colmat.backend_library(),
    )
        f=Libdl.dlsym(Libdl.dlopen(library), Symbol($p*family*"_"*$suffix))
        ccall(
            f,
            Cvoid,
            (Ref{$J}, Ref{$J}, Ref{$J}, Ptr{$T}, Ref{$J}, Ptr{$J}),
            forward,
            m,
            n,
            a,
            size(a, 1),
            k,
        )
    end
end

@testset "LAPMR/LAPMT cycles, direction, and restored permutations" begin
    reference=only(
        filter(
            l->l.interface==:ilp64 && occursin("openblas", lowercase(basename(l.libname))),
            BLAS.get_config().loaded_libs,
        ),
    ).libname
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        m in 0:6,
        n in 0:6,
        family in ("lapmr", "lapmt"),
        forward in (false, true),
        pattern in 1:3

        rows=family=="lapmr"
        count=rows ? m : n
        k=J.(
            if pattern==1
                collect(1:count)
            elseif pattern==2
                reverse(1:count)
            else
                circshift(collect(1:count), 2)
            end,
        )
        a=T[T <: Complex ? T(i+10j, i-j) : T(i+10j) for i in 1:(m + 2), j in 1:(n + 1)]
        a[1, 1]=T(-0.0)
        if m>=2 && n>=1
            a[2, 1]=T(NaN)
        end
        before=copy(a)
        saved_k=copy(k)
        expected=copy(a)
        order=forward ? Int.(k) : invperm(Int.(k))
        expected[1:m, 1:n]=rows ? before[order, 1:n] : before[1:m, order]
        direct_lapm!(family, a, k; m, n, forward=J(forward))
        @test isequal(a, expected)
        @test k==saved_k
        if J===Int64
            external=copy(before)
            direct_lapm!(family, external, k; m, n, forward=J(forward), library=reference)
            @test isequal(a, external)
            @test k==saved_k
        end
        direct_lapm!(family, a, k; m, n, forward=J(!forward))
        @test isequal(a, before)
    end
end
