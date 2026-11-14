using Test, LinearAlgebra, Random, Libdl, Colmat

for (T, prefix) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    J in (Int32, Int64)

    suffix=J===Int64 ? "64_" : ""
    solve=Symbol(prefix*"trtrs_"*suffix)
    symbols=Dict(
        f=>Symbol(prefix*String(f)*"_"*suffix) for f in (:trtri, :trti2, :lauum, :lauu2)
    )
    @eval function direct_triangular_solve!(
        a::Matrix{$T},
        b::Matrix{$T},
        ::Type{$J};
        uplo=('U'),
        trans=('N'),
        diag=('N'),
        n=size(a, 2),
        nrhs=size(b, 2),
        lda=max(1, size(a, 1)),
        ldb=max(1, size(b, 1)),
        library=Colmat.backend_library(),
    )
        info=Ref{$J}(-777)
        f=Libdl.dlsym(Libdl.dlopen(library), $(QuoteNode(solve)))
        ccall(
            f,
            Cvoid,
            (
                Ref{UInt8},
                Ref{UInt8},
                Ref{UInt8},
                Ref{$J},
                Ref{$J},
                Ptr{$T},
                Ref{$J},
                Ptr{$T},
                Ref{$J},
                Ref{$J},
            ),
            uplo,
            trans,
            diag,
            n,
            nrhs,
            a,
            lda,
            b,
            ldb,
            info,
        )
        info[]
    end
    @eval function direct_triangular_matrix!(
        a::Matrix{$T},
        ::Type{$J};
        family=:trtri,
        uplo=('U'),
        diag=('N'),
        n=size(a, 2),
        lda=max(1, size(a, 1)),
        library=Colmat.backend_library(),
    )
        info=Ref{$J}(-777)
        f=Libdl.dlsym(Libdl.dlopen(library), $symbols[family])
        if family in (:trtri, :trti2)
            ccall(
                f,
                Cvoid,
                (Ref{UInt8}, Ref{UInt8}, Ref{$J}, Ptr{$T}, Ref{$J}, Ref{$J}),
                uplo,
                diag,
                n,
                a,
                lda,
                info,
            )
        else
            ccall(
                f,
                Cvoid,
                (Ref{UInt8}, Ref{$J}, Ptr{$T}, Ref{$J}, Ref{$J}),
                uplo,
                n,
                a,
                lda,
                info,
            )
        end
        info[]
    end
end
function triangular_fixture(T, n, uplo, diag)
    raw=randn(T, n, n)/max(1, n)
    for i in 1:n
        raw[i, i]=T(2+i/max(1, n))
    end
    matrix=uplo=='U' ? triu(raw) : tril(raw)
    if diag=='U'
        for i in 1:n
            matrix[i, i]=one(T)
        end
    end
    stored=fill(T(NaN), n+3, n)
    for j in 1:n, i in 1:n
        if (uplo=='U'&&i<=j)||(uplo=='L'&&i>=j)
            stored[i, j]=i==j&&diag=='U' ? T(NaN) : matrix[i, j]
        end
    end
    return stored, matrix
end
function triangular_materialize(a, n, uplo, diag)
    result=uplo=='U' ? triu(a[1:n, 1:n]) : tril(a[1:n, 1:n])
    if diag=='U'
        for i in 1:n
            result[i, i]=one(eltype(a))
        end
    end
    return result
end
function triangular_untouched(a, before, n, uplo, diag)
    return all(
        isequal(a[i, j], before[i, j]) for j in axes(a, 2), i in axes(a, 1) if
        i>n||(uplo=='U'&&i>j)||(uplo=='L'&&i<j)||(diag=='U'&&i==j)
    )
end
@testset "Triangular native solve/inverse with ignored storage" begin
    Random.seed!(270)
    reference=only(
        filter(
            l->l.interface==:ilp64&&occursin("openblas", lowercase(basename(l.libname))),
            BLAS.get_config().loaded_libs,
        ),
    ).libname
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64), uplo in ('U', 'L'), diag in ('N', 'U'),
        n in (0, 1, 3, 80)

        R=typeof(real(zero(T)))
        tolerance=R===Float32 ? 8e-5 : 8e-13
        a, matrix=triangular_fixture(T, n, uplo, diag)
        before=copy(a)
        for trans in ('N', 'T', 'C'), nrhs in (0, 3)
            rhs=randn(T, n, nrhs)
            b=fill(T(777), n+2, nrhs)
            b[1:n, :]=rhs
            @test direct_triangular_solve!(a, b, J; uplo, trans, diag)==0
            op=if trans=='N'
                matrix
            elseif trans=='T'
                transpose(matrix)
            else
                matrix'
            end
            @test op*b[1:n, :] ≈ rhs atol=tolerance
            @test all(==(T(777)), b[(n + 1):end, :])
            @test isequal(a, before)
        end
        for family in (:trtri, :trti2)
            inverse=copy(a)
            @test direct_triangular_matrix!(inverse, J; family, uplo, diag)==0
            actual=triangular_materialize(inverse, n, uplo, diag)
            @test matrix*actual ≈ Matrix{T}(I, n, n) atol=tolerance
            @test actual*matrix ≈ Matrix{T}(I, n, n) atol=tolerance
            @test triangular_untouched(inverse, before, n, uplo, diag)
            expected=copy(a)
            @test direct_triangular_matrix!(
                expected, Int64; family, uplo, diag, library=reference
            )==0
            @test actual ≈ triangular_materialize(expected, n, uplo, diag) atol=tolerance
            # An externally generated inverse is valid input to native TRTRS.
            rhs=randn(T, n, 2)
            b=fill(T(777), n+2, 2)
            b[1:n, :]=rhs
            @test direct_triangular_solve!(expected, b, J; uplo, diag)==0
            @test b[1:n, :] ≈ matrix*rhs atol=tolerance
        end
    end
end
@testset "Triangular products, complex diagonal and selected output" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64), uplo in ('U', 'L'), n in (0, 1, 3, 80),
        family in (:lauum, :lauu2)

        R=typeof(real(zero(T)))
        tolerance=R===Float32 ? 8e-5 : 8e-13
        a, matrix=triangular_fixture(T, n, uplo, 'N')
        if T<:Complex
            for i in 1:n
                a[i, i]+=T(0.0, 0.75)
                matrix[i, i]=a[i, i]
            end
        end
        before=copy(a)
        expected=uplo=='U' ? matrix*matrix' : matrix'*matrix
        @test direct_triangular_matrix!(a, J; family, uplo)==0
        @test triangular_untouched(a, before, n, uplo, 'N')
        actual=uplo=='U' ? triu(a[1:n, 1:n]) : tril(a[1:n, 1:n])
        selected=uplo=='U' ? triu(expected) : tril(expected)
        @test actual ≈ selected atol=tolerance
        @test all(iszero, imag.(diag(a[1:n, 1:n])))
    end
end
@testset "Triangular singularity and argument contracts" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        uplo in ('U', 'L')

        a, _=triangular_fixture(T, 3, uplo, 'N')
        a[2, 2]=zero(T)
        before=copy(a)
        for nrhs in (0, 2)
            b=fill(T(777), 5, nrhs)
            prior=copy(b)
            @test direct_triangular_solve!(a, b, J; uplo)==2
            @test isequal(b, prior)
        end
        @test direct_triangular_matrix!(a, J; uplo)==2
        @test isequal(a, before)
        @test direct_triangular_matrix!(copy(a), J; family=:trti2, uplo)==0
        b=ones(T, 5, 2)
        for (options, expected) in (
            ((; uplo=('?')), -1),
            ((; trans=('?')), -2),
            ((; diag=('?')), -3),
            ((; n=-1), -4),
            ((; nrhs=-1), -5),
            ((; lda=2), -7),
            ((; ldb=2), -9),
        )
            @test direct_triangular_solve!(a, b, J; options...)==expected
        end
        for family in (:trtri, :trti2, :lauum, :lauu2)
            @test direct_triangular_matrix!(a, J; family, uplo=('?'))==-1
            @test direct_triangular_matrix!(a, J; family, n=-1)==(
                family in (:trtri, :trti2) ? -3 : -2
            )
            @test direct_triangular_matrix!(a, J; family, lda=2)==(
                family in (:trtri, :trti2) ? -5 : -4
            )
        end
        if J===Int64
            @test direct_triangular_solve!(a, b, J; n=(-(Int64(1)<<32)))==-4
            @test direct_triangular_matrix!(a, J; n=(-(Int64(1)<<32)))==-3
        end
    end
end
@testset "Triangular LAPACK forwards and Julia API" begin
    for prefix in ("s", "d", "c", "z"),
        family in (:trtrs, :trtri, :trti2, :lauum, :lauu2),
        interface in (:lp64, :ilp64)

        library=BLAS.lbt_find_backing_library(prefix*String(family)*"_", interface)
        @test library!==nothing
        @test realpath(library.libname)==realpath(Colmat.backend_library())
    end
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        a=triu(randn(T, 4, 4))
        for i in 1:4
            a[i, i]+=T(3)
        end
        inverse=inv(UpperTriangular(a))
        @test a*inverse ≈ Matrix{T}(I, 4, 4)
        rhs=randn(T, 4, 2)
        solution=LAPACK.trtrs!('U', 'N', 'N', copy(a), copy(rhs))
        @test a*solution ≈ rhs
    end
end
