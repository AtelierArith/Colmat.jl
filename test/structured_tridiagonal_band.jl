using Test, LinearAlgebra, Random, Libdl, Colmat
isdefined(@__MODULE__, :packed_tridiagonal_diagnostic) ||
    include("structured_tridiagonal_packed.jl")
function tridiagonal_band_store(a, kd, u)
    T=eltype(a)
    n=size(a, 2)
    ab=fill(T(NaN), kd+3, n)
    ab[(kd + 2):end, :].=777
    for j in 1:n, i in (u=='L' ? (j:min(n, j + kd)) : (max(1, j - kd):j))
        row=u=='L' ? 1+i-j : kd+1+i-j
        ab[row, j]=T<:Complex&&i==j ? T(real(a[i, j]), NaN) : a[i, j]
    end
    return ab
end
function tridiagonal_band_ignored(ab, old, kd, u)
    n=size(ab, 2)
    all(axes(ab, 2)) do j
        all(axes(ab, 1)) do row
            i=u=='L' ? j+row-1 : j+row-kd-1
            selected=row<=kd+1&&1<=i<=n&&(u=='L' ? i>=j : i<=j)
            return selected||isequal(ab[row, j], old[row, j])
        end
    end
end
for (T, prefix) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    J in (Int32, Int64)

    R=real(T)
    suffix=J===Int64 ? "64_" : ""
    symbol=Symbol(prefix*(T<:Real ? "sbtrd_" : "hbtrd_")*suffix)
    @eval function direct_band_trd!(
        ab::Matrix{$T}, q::Matrix{$T}, ::Type{$J}; kd, u, vect=('V'), pointer=nothing
    )
        n=size(ab, 2)
        f=if isnothing(pointer)
            Libdl.dlsym(Libdl.dlopen(Colmat.backend_library()), $(QuoteNode(symbol)))
        else
            pointer
        end
        d=fill($R(777), n+2)
        e=fill($R(777), max(0, n-1)+2)
        work=fill($T(777), n+2)
        info=fill($J(-777), 3)
        ccall(
            f,
            Cvoid,
            (
                Ref{UInt8},
                Ref{UInt8},
                Ref{$J},
                Ref{$J},
                Ptr{$T},
                Ref{$J},
                Ptr{$R},
                Ptr{$R},
                Ptr{$T},
                Ref{$J},
                Ptr{$T},
                Ptr{$J},
                Clong,
                Clong,
            ),
            vect,
            u,
            n,
            kd,
            ab,
            size(ab, 1),
            view(d, 2:(length(d) - 1)),
            view(e, 2:(length(e) - 1)),
            q,
            size(q, 1),
            view(work, 2:(length(work) - 1)),
            view(info, 2:2),
            1,
            1,
        )
        @test info[[1, 3]]==[$J(-777), $J(-777)]
        @test d[[1, end]]==[$R(777), $R(777)]&&e[[1, end]]==[$R(777), $R(777)]&&work[[
            1, end
        ]]==[$T(777), $T(777)]
        (;
            values=d[2:(end - 1)],
            offdiag=e[2:(end - 1)],
            info=info[2],
            diagnostic=packed_tridiagonal_diagnostic(),
        )
    end
end
@testset "Band tridiagonal native reduction and existing Q updates" begin
    Random.seed!(3091)
    external=only(
        filter(
            l->l.interface==:ilp64&&occursin("openblas", lowercase(basename(l.libname))),
            BLAS.get_config().loaded_libs,
        ),
    )
    lib=Libdl.dlopen(external.libname)
    for (T, prefix) in
        ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
        J in (Int32, Int64),
        u in ('U', 'L'),
        n in (0, 1, 2, 5, 9),
        kd in unique((0, 1, 2, n+1)),
        form in (:dense, :zero, :rankone, :repeated)

        tol=T in (Float32, ComplexF32) ? 3e-3 : 3e-10
        raw=randn(T, n, n)/max(1, n)
        a=if form===:dense
            raw+raw'
        elseif form===:zero
            zeros(T, n, n)
        elseif form===:rankone
            Matrix(Diagonal(T.([i==1 ? 3 : 0 for i in 1:n])))
        else
            Matrix(Diagonal(T.(mod.(1:n, 2))))
        end
        for j in 1:n, i in 1:n
            abs(i-j)>kd&&(a[i, j]=0)
        end
        original=tridiagonal_band_store(a, kd, u)
        ab=copy(original)
        q=fill(T(777), n+3, n)
        result=direct_band_trd!(ab, q, J; kd, u)
        @test result.info==result.diagnostic==0
        @test tridiagonal_band_ignored(ab, original, kd, u)
        basis=q[1:n, :]
        tri=Matrix(SymTridiagonal(result.values, result.offdiag))
        @test basis*tri*basis'≈a atol=tol rtol=tol
        @test basis'*basis≈Matrix{T}(I, n, n) atol=tol rtol=tol
        @test all(==(T(777)), q[(n + 1):end, :])
        for j in 1:n
            @test ab[u=='L' ? 1 : kd+1, j]≈result.values[j] atol=tol rtol=tol
            if j<n&&kd>0
                @test ab[u=='L' ? 2 : kd, u=='L' ? j : j+1]≈result.offdiag[j] atol=tol rtol=tol
            end
        end
        if T<:Complex
            @test all(>=(0), result.offdiag)
        elseif kd<=1
            @test basis==Matrix{T}(I, n, n)
        end
        qr=fill(T(777), n+3, n)
        ar=copy(original)
        reference=direct_band_trd!(
            ar,
            qr,
            Int64;
            kd,
            u,
            pointer=Libdl.dlsym(lib, Symbol(prefix*(T<:Real ? "sbtrd_64_" : "hbtrd_64_"))),
        )
        @test reference.info==0
        reference_basis=qr[1:n, :]
        @test reference_basis*Matrix(SymTridiagonal(reference.values, reference.offdiag))*reference_basis'≈a atol=tol rtol=tol
        if n == 0
            # Julia's SymTridiagonal eigvals passes LDZ=0 to STEV for an
            # empty matrix. Compare the empty spectra without that unrelated
            # invalid LAPACK call (and its sticky diagnostic).
            @test isempty(result.values) && isempty(reference.values)
        else
            @test eigvals(SymTridiagonal(result.values, result.offdiag)) ≈
                eigvals(SymTridiagonal(reference.values, reference.offdiag)) atol=tol rtol=tol
        end
        for vect in ('N', 'U')
            ab=copy(original)
            x=randn(T, n, n)
            output=vect=='N' ? fill(T(777), 1, 1) : vcat(x, fill(T(777), 3, n))
            before=copy(output)
            changed=direct_band_trd!(ab, output, J; kd, u, vect)
            @test changed.info==changed.diagnostic==0
            @test changed.values≈result.values atol=tol rtol=tol
            @test changed.offdiag≈result.offdiag atol=tol rtol=tol
            @test tridiagonal_band_ignored(ab, original, kd, u)
            if vect=='N'
                @test output==before
            else
                @test output[1:n, :]≈x*basis atol=tol rtol=tol
                @test output[(n + 1):end, :]==before[(n + 1):end, :]
            end
        end
    end
end
