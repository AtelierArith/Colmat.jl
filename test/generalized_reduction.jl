using Test, LinearAlgebra, Random, Libdl, Colmat

for (T, prefix, families) in (
        (Float32, "s", (:sygst, :sygs2)),
        (Float64, "d", (:sygst, :sygs2)),
        (ComplexF32, "c", (:hegst, :hegs2)),
        (ComplexF64, "z", (:hegst, :hegs2)),
    ),
    J in (Int32, Int64)

    symbols=Dict(f=>Symbol(prefix*String(f)*"_"*(J===Int64 ? "64_" : "")) for f in families)
    @eval function direct_generalized_reduction!(
        a::Matrix{$T},
        b::Matrix{$T},
        ::Type{$J};
        family=($(QuoteNode(first(families)))),
        itype=1,
        uplo=('L'),
        n=size(a, 2),
        lda=max(1, size(a, 1)),
        ldb=max(1, size(b, 1)),
        library=Colmat.backend_library(),
    )
        info=Ref{$J}(-777)
        f=Libdl.dlsym(Libdl.dlopen(library), $symbols[family])
        ccall(
            f,
            Cvoid,
            (Ref{$J}, Ref{UInt8}, Ref{$J}, Ptr{$T}, Ref{$J}, Ptr{$T}, Ref{$J}, Ref{$J}),
            itype,
            uplo,
            n,
            a,
            lda,
            b,
            ldb,
            info,
        )
        info[]
    end
end
function congruence_storage(matrix, uplo, rows; poison_diagonal=false)
    T=eltype(matrix)
    n=size(matrix, 2)
    result=fill(T(NaN), rows, n)
    for j in 1:n, i in 1:n
        if (uplo=='U'&&i<=j)||(uplo=='L'&&i>=j)
            result[i, j]=matrix[i, j]
            if poison_diagonal&&T<:Complex&&i==j
                result[i, j]=T(real(matrix[i, j]), NaN)
            end
        end
    end
    return result
end
congruence_materialize(a, n, uplo) = Matrix(Hermitian(a[1:n, 1:n], uplo=='U' ? :U : :L))
function congruence_untouched(a, prior, n, uplo)
    return all(
        isequal(a[i, j], prior[i, j]) for
        j in axes(a, 2), i in axes(a, 1) if i>n||(uplo=='U'&&i>j)||(uplo=='L'&&i<j)
    )
end
@testset "Generalized standard-form congruences and external factors" begin
    Random.seed!(275)
    reference=only(
        filter(
            l->l.interface==:ilp64&&occursin("openblas", lowercase(basename(l.libname))),
            BLAS.get_config().loaded_libs,
        ),
    ).libname
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64), uplo in ('L', 'U'), itype in 1:3, n in (0, 1, 3, 40, 80),
        family in (T<:Real ? (:sygst, :sygs2) : (:hegst, :hegs2))

        R=typeof(real(zero(T)))
        tolerance=R===Float32 ? 1e-4 : 1e-12
        raw=randn(T, n, n)/max(1, n)
        matrix=Matrix(Hermitian(raw+raw'))
        lower=tril(randn(T, n, n)/max(1, n))
        for i in 1:n
            lower[i, i]=T(1+i/max(1, n))
        end
        factor=uplo=='L' ? lower : Matrix(lower')
        a=congruence_storage(matrix, uplo, n+3)
        b=congruence_storage(factor, uplo, n+4)
        prior=copy(a)
        bprior=copy(b)
        external=copy(a)
        @test direct_generalized_reduction!(a, b, J; family, itype, uplo)==0
        @test direct_generalized_reduction!(
            external, b, Int64; family, itype, uplo, library=reference
        )==0
        actual=congruence_materialize(a, n, uplo)
        expected=itype==1 ? (lower\matrix)/lower' : lower'*matrix*lower
        @test actual ≈ expected atol=tolerance
        @test actual ≈ congruence_materialize(external, n, uplo) atol=tolerance
        @test isequal(b, bprior)
        @test congruence_untouched(a, prior, n, uplo)
        @test all(iszero, imag.(diag(a[1:n, 1:n])))
    end
end
@testset "Generalized original equations and ignored diagonal components" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64), uplo in ('L', 'U'),
        itype in 1:3

        R=typeof(real(zero(T)))
        tolerance=R===Float32 ? 2e-4 : 2e-12
        n=5
        raw=randn(T, n, n)
        matrix=Matrix(Hermitian(raw+raw'))
        lower=tril(randn(T, n, n)/n)
        for i in 1:n
            lower[i, i]=T(1+i/n)
        end
        positive=lower*lower'
        factor=uplo=='L' ? lower : Matrix(lower')
        a=congruence_storage(matrix, uplo, n+3; poison_diagonal=true)
        b=congruence_storage(factor, uplo, n+4; poison_diagonal=true)
        bprior=copy(b)
        @test direct_generalized_reduction!(a, b, J; itype, uplo)==0
        reduced=congruence_materialize(a, n, uplo)
        decomposition=eigen(Hermitian(reduced))
        values, vectors=decomposition.values, decomposition.vectors
        original=itype==3 ? lower*vectors : lower'\vectors
        if itype==1
            @test matrix*original ≈ positive*original*Diagonal(values) atol=tolerance
        elseif itype==2
            @test matrix*positive*original ≈ original*Diagonal(values) atol=tolerance
        else
            @test positive*matrix*original ≈ original*Diagonal(values) atol=tolerance
        end
        if itype==3
            @test original'*(positive\original) ≈ Matrix{T}(I, n, n) atol=tolerance
        else
            @test original'*positive*original ≈ Matrix{T}(I, n, n) atol=tolerance
        end
        @test isequal(b, bprior)
    end
end
@testset "Generalized reduction argument contracts and integer width" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        family in (T<:Real ? (:sygst, :sygs2) : (:hegst, :hegs2))

        a=Matrix{T}(I, 2, 2)
        b=copy(a)
        for (options, status) in (
            ((; itype=0), -1),
            ((; itype=4), -1),
            ((; uplo=('?')), -2),
            ((; n=-1), -3),
            ((; lda=1), -5),
            ((; ldb=1), -7),
        )
            prior=(copy(a), copy(b))
            @test direct_generalized_reduction!(a, b, J; family, options...)==status
            @test isequal(prior, (a, b))
        end
        if J===Int64
            @test direct_generalized_reduction!(a, b, J; family, n=(-(Int64(1)<<32)))==-3
            @test direct_generalized_reduction!(a, b, J; family, itype=(Int64(1)<<32)+1)==-1
        end
    end
end
@testset "Generalized reduction LBT and ordinary arrays" begin
    for (prefix, families) in (
            ("s", (:sygst, :sygs2)),
            ("d", (:sygst, :sygs2)),
            ("c", (:hegst, :hegs2)),
            ("z", (:hegst, :hegs2)),
        ),
        interface in (:lp64, :ilp64),
        family in families

        library=BLAS.lbt_find_backing_library(prefix*String(family)*"_", interface)
        @test library!==nothing
        @test realpath(library.libname)==realpath(Colmat.backend_library())
    end
    for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
        a=Matrix{T}(I, 3, 3)
        b=Matrix(Diagonal(T[2, 3, 4]))
        @test direct_generalized_reduction!(a, b, J; library=BLAS.libblastrampoline)==0
        @test a ≈ Diagonal(T[1 / 4, 1 / 9, 1 / 16])
    end
end

for (T, prefix) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z"))
    symbol=Symbol(prefix*"potrf_64_")
    @eval function external_congruence_factor!(b::Matrix{$T}, uplo, library)
        info=Ref{Int64}(-777)
        f=Libdl.dlsym(Libdl.dlopen(library), $(QuoteNode(symbol)))
        ccall(
            f,
            Cvoid,
            (Ref{UInt8}, Ref{Int64}, Ptr{$T}, Ref{Int64}, Ref{Int64}),
            uplo,
            size(b, 2),
            b,
            max(1, size(b, 1)),
            info,
        )
        info[]
    end
end
@testset "Generalized reduction with external Cholesky and wide scales" begin
    reference=only(
        filter(
            l->l.interface==:ilp64&&occursin("openblas", lowercase(basename(l.libname))),
            BLAS.get_config().loaded_libs,
        ),
    ).libname
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64), uplo in ('L', 'U'),
        itype in 1:3

        R=typeof(real(zero(T)))
        tolerance=R===Float32 ? 2e-4 : 2e-12
        matrix=T[2 0.25 -0.125; 0.25 3 0.5; -0.125 0.5 -1]
        lower=T[1 0 0; 0.125 2 0; -0.25 0.25 3]
        if T<:Complex
            lower[2, 1]+=T(0, 0.1)
            matrix[2, 1]+=T(0, 0.2)
            matrix[1, 2]=conj(matrix[2, 1])
        end
        positive=lower*lower'
        b=congruence_storage(positive, uplo, 7)
        @test external_congruence_factor!(b, uplo, reference)==0
        before=copy(b)
        a=congruence_storage(matrix, uplo, 6)
        @test direct_generalized_reduction!(a, b, J; itype, uplo)==0
        expected=itype==1 ? (lower\matrix)/lower' : lower'*matrix*lower
        @test congruence_materialize(a, 3, uplo) ≈ expected rtol=tolerance
        @test isequal(b, before)
        for scale in
            (R===Float32 ? (R(1e-10), one(R), R(1e10)) : (R(1e-100), one(R), R(1e100)))
            scaled=scale*lower
            factor=uplo=='L' ? scaled : Matrix(scaled')
            b=congruence_storage(factor, uplo, 7)
            a=congruence_storage(matrix, uplo, 6)
            @test direct_generalized_reduction!(a, b, J; itype, uplo)==0
            expected=itype==1 ? (scaled\matrix)/scaled' : scaled'*matrix*scaled
            @test congruence_materialize(a, 3, uplo) ≈ expected rtol=tolerance
        end
    end
end
