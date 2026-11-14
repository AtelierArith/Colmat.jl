using Test, LinearAlgebra, Random, Libdl, Colmat
for (T, p) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    J in (Int32, Int64)

    R=typeof(real(zero(T)))
    suffix=J===Int64 ? "64_" : ""
    blocked=Symbol(p*"pstrf_"*suffix)
    unblocked=Symbol(p*"pstf2_"*suffix)
    @eval function direct_pivoted_cholesky!(
        a::Matrix{$T},
        ::Type{$J};
        uplo=('L'),
        tol=-1,
        block=true,
        n_override=nothing,
        library=Colmat.backend_library(),
    )
        n=size(a, 2)
        piv=fill($J(777), n+2)
        rank=Ref{$J}(777)
        work=zeros($R, 2n)
        info=Ref{$J}(0)
        f=Libdl.dlsym(
            Libdl.dlopen(library), block ? $(QuoteNode(blocked)) : $(QuoteNode(unblocked))
        )
        ccall(
            f,
            Cvoid,
            (
                Ref{UInt8},
                Ref{$J},
                Ptr{$T},
                Ref{$J},
                Ptr{$J},
                Ref{$J},
                Ref{$R},
                Ptr{$R},
                Ref{$J},
            ),
            uplo,
            isnothing(n_override) ? n : n_override,
            a,
            max(1, size(a, 1)),
            view(piv, 2:(n + 1)),
            rank,
            $R(tol),
            work,
            info,
        )
        return piv[2:(n + 1)], rank[], info[], piv[[1, end]]
    end
end
@testset "Pivoted Cholesky direct ABIs and partial factors" begin
    Random.seed!(265)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64), uplo in ('L', 'U'), block in (false, true),
        (n, k) in ((0, 0), (1, 0), (1, 1), (7, 3), (7, 7))

        original=zeros(T, n, n)
        if k>0
            x=randn(T, k, k)
            original[1:k, 1:k]=x*x'+I
            permutation=randperm(n)
            original=original[permutation, permutation]
        end
        a=fill(T(987), n+2, n)
        a[1:n, :]=original
        for j in 1:n, i in 1:n
            if (uplo=='L'&&i<j)||(uplo=='U'&&i>j)
                a[i, j]=T(NaN)
            end
        end
        before=copy(a)
        piv, rank, info, guards=direct_pivoted_cholesky!(a, J; uplo, block)
        @test rank==k
        @test info==(k<n ? 1 : 0)
        @test sort(piv)==collect(1:n)
        @test guards==fill(J(777), 2)
        @test all(==(T(987)), a[(n + 1):end, :])
        factor=uplo=='L' ? tril(a[1:n, :])[:, 1:rank] : triu(a[1:n, :])[1:rank, :]'
        @test factor*factor' ≈ original[piv, piv]
        for j in 1:n, i in 1:n
            if (uplo=='L'&&i<j)||(uplo=='U'&&i>j)
                @test isequal(a[i, j], before[i, j])
            end
        end
    end
end
@testset "Pivoted Cholesky tolerance and stop boundaries" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64), uplo in ('L', 'U'),
        block in (false, true)

        R=typeof(real(zero(T)))
        for (d, tol, expected) in (
            (R[2, 1], 100, 1),
            (R[1, 1.5eps(R)], -1, 2),
            (R[1, eps(R)], -1, 1),
            (R[1, 2.0 ^ -10, 2.0 ^ -20], 2.0^-10, 1),
            (R[1, 2.0 ^ -10, 2.0 ^ -20], 0, 3),
            (R[1, -1], 0, 1),
            (R[-1, -2], -1, 0),
            (R[floatmax(R) / 4, nextfloat(zero(R))], 0, 2),
        )
            original=Matrix(Diagonal(T.(d)))
            a=copy(original)
            piv, rank, info, _=direct_pivoted_cholesky!(a, J; uplo, block, tol)
            @test rank==expected
            @test info==(expected<length(d) ? 1 : 0)
            if expected==length(d)
                @test real.(diag(a)) .^ 2 ≈ d[piv]
            end
        end
        a=ones(T, 1, 1)
        _, rank, info, _=direct_pivoted_cholesky!(a, J; uplo=('?'))
        @test info == -1 && rank==777
        _, rank, info, _=direct_pivoted_cholesky!(a, J; n_override=-1)
        @test info == -2 && rank==777
    end
end

@testset "Pivoted Cholesky independent OpenBLAS factors and stop state" begin
    library=only(
        filter(
            l->l.interface==:ilp64 && occursin("openblas", lowercase(basename(l.libname))),
            BLAS.get_config().loaded_libs,
        ),
    ).libname
    Random.seed!(2651)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64), uplo in ('L', 'U'),
        block in (false, true)

        x=randn(T, 5, 5)/10
        original=x*x'+Diagonal(T.(1:5))
        a=copy(original)
        reference=copy(original)
        piv, rank, info, _=direct_pivoted_cholesky!(a, J; uplo, block)
        rp, rr, ri, _=direct_pivoted_cholesky!(reference, Int64; uplo, block, library)
        @test rank==rr==5 && info==ri==0
        @test piv==rp
        @test (uplo=='L' ? tril(a) : triu(a)) ≈
            (uplo=='L' ? tril(reference) : triu(reference))
        for (original, tol) in
            ((T[4 2 0; 2 1 0; 0 0 0], -1), (Matrix(Diagonal(T[4, 1, 0])), 1))
            a=copy(original)
            reference=copy(original)
            piv, rank, info, _=direct_pivoted_cholesky!(a, J; uplo, block=false, tol)
            rp, rr, ri, _=direct_pivoted_cholesky!(
                reference, Int64; uplo, block=false, tol, library
            )
            @test rank==rr && info==ri
            @test piv==rp
            @test (uplo=='L' ? tril(a) : triu(a)) ≈
                (uplo=='L' ? tril(reference) : triu(reference))
        end
        prefix=if T===Float32
            "s"
        elseif T===Float64
            "d"
        elseif T===ComplexF32
            "c"
        else
            "z"
        end
        for family in ("pstrf", "pstf2")
            backing=BLAS.lbt_find_backing_library(
                prefix*family*"_", J===Int32 ? :lp64 : :ilp64
            )
            @test backing!==nothing &&
                realpath(backing.libname)==realpath(Colmat.backend_library())
        end
    end
end

@testset "Pivoted Cholesky Julia wrappers and ignored diagonal phases" begin
    Random.seed!(2652)
    for T in (Float32, Float64, ComplexF32, ComplexF64), uplo in ('L', 'U')
        x=randn(T, 3, 3)
        original=zeros(T, 6, 6)
        original[1:3, 1:3]=x*x'+I
        permutation=randperm(6)
        original=original[permutation, permutation]
        result=cholesky(
            Hermitian(original, Symbol(uplo)), RowMaximum(); tol=-1, check=false
        )
        @test result.rank==3
        l=Matrix(result.L)[:, 1:result.rank]
        @test l*l' ≈ original[result.p, result.p]
        y=randn(T, 6, 3)
        gram=y*y'
        approximate=cholesky(
            Hermitian(gram, Symbol(uplo)), RowMaximum(); tol=-1, check=false
        )
        @test approximate.rank>=3
        columns=Matrix(approximate.L)[:, 1:approximate.rank]
        @test columns*columns' ≈ gram[approximate.p, approximate.p]
        if T<:Complex
            original=Matrix(Diagonal(T[3, 2, 1]))
            a=copy(original)
            for i in 1:3
                a[i, i]=T(real(a[i, i]), NaN)
            end
            piv, rank, info, _=direct_pivoted_cholesky!(a, Int64; uplo)
            @test rank==3 && info==0
            @test all(iszero, imag.(diag(a)))
            @test real.(diag(a)) .^ 2 ≈ real.(diag(original))[piv]
        end
    end
end
