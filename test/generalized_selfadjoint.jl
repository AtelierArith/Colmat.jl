using Test, LinearAlgebra, Random, Libdl, Colmat

function selfadjoint_workspace(T, n, job, family)
    if family==:gv
        return (max(1, (T<:Real ? 3 : 2)*n-1), max(1, 3n-2), 1)
    elseif family==:gvx
        return (max(1, (T<:Real ? 8 : 2)*n), max(1, 7n), max(1, 5n))
    elseif n<=1
        return (1, 1, 1)
    elseif T<:Real
        return job=='V' ? (1+6n+2n*n, 1, 3+5n) : (2n+1, 1, 1)
    end
    return job=='V' ? (2n+n*n, 1+5n+2n*n, 3+5n) : (n+1, n, 1)
end
for (T, prefix) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    J in (Int32, Int64)

    R=typeof(real(zero(T)))
    symbols=Dict(
        f=>Symbol(prefix*(T<:Real ? "sy" : "he")*String(f)*"_"*(J===Int64 ? "64_" : "")) for
        f in (:gv, :gvd, :gvx)
    )
    @eval function direct_selfadjoint!(
        a::Matrix{$T},
        b::Matrix{$T},
        ::Type{$J};
        family=:gv,
        itype=1,
        jobz=('V'),
        uplo=('L'),
        range=('A'),
        vl=($R(NaN)),
        vu=($R(NaN)),
        il=-777,
        iu=-777,
        abstol=zero($R),
        n=size(a, 2),
        lda=max(1, size(a, 1)),
        ldb=max(1, size(b, 1)),
        ldz=n+3,
        lwork=nothing,
        lrwork=nothing,
        liwork=nothing,
        library=Colmat.backend_library(),
    )
        order=size(a, 2)
        sizes=selfadjoint_workspace($T, order, jobz, family)
        lw=isnothing(lwork) ? sizes[1] : lwork
        lr=isnothing(lrwork) ? sizes[2] : lrwork
        li=isnothing(liwork) ? sizes[3] : liwork
        work=fill($T(777), max(1, lw)+2)
        rwork=fill($R(777), max(1, lr)+2)
        iwork=fill($J(777), max(1, li)+2)
        values=fill($R(777), order+2)
        failed=fill($J(777), max(1, order)+2)
        z=fill($T(777), max(order+3, 1), max(1, order))
        count=Ref{$J}(-777)
        info=Ref{$J}(-777)
        wp=view(work, 2:(length(work) - 1))
        rp=view(rwork, 2:(length(rwork) - 1))
        ip=view(iwork, 2:(length(iwork) - 1))
        vp=view(values, 2:(length(values) - 1))
        fp=view(failed, 2:(length(failed) - 1))
        f=Libdl.dlsym(Libdl.dlopen(library), $symbols[family])
        if family==:gv
            if $T<:Real
                ccall(
                    f,
                    Cvoid,
                    (
                        Ref{$J},
                        Ref{UInt8},
                        Ref{UInt8},
                        Ref{$J},
                        Ptr{$T},
                        Ref{$J},
                        Ptr{$T},
                        Ref{$J},
                        Ptr{$R},
                        Ptr{$T},
                        Ref{$J},
                        Ref{$J},
                    ),
                    itype,
                    jobz,
                    uplo,
                    n,
                    a,
                    lda,
                    b,
                    ldb,
                    vp,
                    wp,
                    lw,
                    info,
                )
            else
                ccall(
                    f,
                    Cvoid,
                    (
                        Ref{$J},
                        Ref{UInt8},
                        Ref{UInt8},
                        Ref{$J},
                        Ptr{$T},
                        Ref{$J},
                        Ptr{$T},
                        Ref{$J},
                        Ptr{$R},
                        Ptr{$T},
                        Ref{$J},
                        Ptr{$R},
                        Ref{$J},
                    ),
                    itype,
                    jobz,
                    uplo,
                    n,
                    a,
                    lda,
                    b,
                    ldb,
                    vp,
                    wp,
                    lw,
                    rp,
                    info,
                )
            end
        elseif family==:gvd
            if $T<:Real
                ccall(
                    f,
                    Cvoid,
                    (
                        Ref{$J},
                        Ref{UInt8},
                        Ref{UInt8},
                        Ref{$J},
                        Ptr{$T},
                        Ref{$J},
                        Ptr{$T},
                        Ref{$J},
                        Ptr{$R},
                        Ptr{$T},
                        Ref{$J},
                        Ptr{$J},
                        Ref{$J},
                        Ref{$J},
                    ),
                    itype,
                    jobz,
                    uplo,
                    n,
                    a,
                    lda,
                    b,
                    ldb,
                    vp,
                    wp,
                    lw,
                    ip,
                    li,
                    info,
                )
            else
                ccall(
                    f,
                    Cvoid,
                    (
                        Ref{$J},
                        Ref{UInt8},
                        Ref{UInt8},
                        Ref{$J},
                        Ptr{$T},
                        Ref{$J},
                        Ptr{$T},
                        Ref{$J},
                        Ptr{$R},
                        Ptr{$T},
                        Ref{$J},
                        Ptr{$R},
                        Ref{$J},
                        Ptr{$J},
                        Ref{$J},
                        Ref{$J},
                    ),
                    itype,
                    jobz,
                    uplo,
                    n,
                    a,
                    lda,
                    b,
                    ldb,
                    vp,
                    wp,
                    lw,
                    rp,
                    lr,
                    ip,
                    li,
                    info,
                )
            end
        else
            if $T<:Real
                ccall(
                    f,
                    Cvoid,
                    (
                        Ref{$J},
                        Ref{UInt8},
                        Ref{UInt8},
                        Ref{UInt8},
                        Ref{$J},
                        Ptr{$T},
                        Ref{$J},
                        Ptr{$T},
                        Ref{$J},
                        Ref{$R},
                        Ref{$R},
                        Ref{$J},
                        Ref{$J},
                        Ref{$R},
                        Ref{$J},
                        Ptr{$R},
                        Ptr{$T},
                        Ref{$J},
                        Ptr{$T},
                        Ref{$J},
                        Ptr{$J},
                        Ptr{$J},
                        Ref{$J},
                    ),
                    itype,
                    jobz,
                    range,
                    uplo,
                    n,
                    a,
                    lda,
                    b,
                    ldb,
                    vl,
                    vu,
                    il,
                    iu,
                    abstol,
                    count,
                    vp,
                    z,
                    ldz,
                    wp,
                    lw,
                    ip,
                    fp,
                    info,
                )
            else
                ccall(
                    f,
                    Cvoid,
                    (
                        Ref{$J},
                        Ref{UInt8},
                        Ref{UInt8},
                        Ref{UInt8},
                        Ref{$J},
                        Ptr{$T},
                        Ref{$J},
                        Ptr{$T},
                        Ref{$J},
                        Ref{$R},
                        Ref{$R},
                        Ref{$J},
                        Ref{$J},
                        Ref{$R},
                        Ref{$J},
                        Ptr{$R},
                        Ptr{$T},
                        Ref{$J},
                        Ptr{$T},
                        Ref{$J},
                        Ptr{$R},
                        Ptr{$J},
                        Ptr{$J},
                        Ref{$J},
                    ),
                    itype,
                    jobz,
                    range,
                    uplo,
                    n,
                    a,
                    lda,
                    b,
                    ldb,
                    vl,
                    vu,
                    il,
                    iu,
                    abstol,
                    count,
                    vp,
                    z,
                    ldz,
                    wp,
                    lw,
                    rp,
                    ip,
                    fp,
                    info,
                )
            end
        end
        (; info=info[], m=count[], values, work, rwork, iwork, failed, z)
    end
end
function selfadjoint_storage(matrix, uplo; rows=size(matrix, 1)+3, poison=false)
    T=eltype(matrix)
    n=size(matrix, 2)
    result=fill(T(NaN), rows, n)
    for j in 1:n, i in 1:n
        if (uplo=='U'&&i<=j)||(uplo=='L'&&i>=j)
            result[i, j]=if poison&&T<:Complex&&i==j
                T(real(matrix[i, j]), NaN)
            else
                matrix[i, j]
            end
        end
    end
    return result
end
function selfadjoint_untouched(a, prior, n, uplo)
    return all(
        isequal(a[i, j], prior[i, j]) for
        j in axes(a, 2), i in axes(a, 1) if i>n||(uplo=='U'&&i>j)||(uplo=='L'&&i<j)
    )
end
function selfadjoint_equations(matrix, positive, values, vectors, kind, tolerance)
    if kind==1
        @test matrix*vectors ≈ positive*vectors*Diagonal(values) atol=tolerance
    elseif kind==2
        @test matrix*positive*vectors ≈ vectors*Diagonal(values) atol=tolerance
    else
        @test positive*matrix*vectors ≈ vectors*Diagonal(values) atol=tolerance
    end
    n=size(vectors, 2)
    gram=kind==3 ? vectors'*(positive\vectors) : vectors'*positive*vectors
    @test gram ≈ Matrix{eltype(vectors)}(I, n, n) atol=tolerance
end
@testset "Generalized self-adjoint full drivers and normalization" begin
    Random.seed!(292)
    reference=only(
        filter(
            l->l.interface==:ilp64&&occursin("openblas", lowercase(basename(l.libname))),
            BLAS.get_config().loaded_libs,
        ),
    ).libname
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        uplo in ('L', 'U'),
        itype in 1:3,
        jobz in ('N', 'V'),
        n in (0, 1, 4, 12),
        family in (:gv, :gvd)

        R=typeof(real(zero(T)))
        tolerance=R===Float32 ? 4e-4 : 4e-12
        raw=randn(T, n, n)/max(1, n)
        matrix=Matrix(Hermitian(raw+raw'))
        lower=tril(randn(T, n, n)/max(1, n))
        for i in 1:n
            lower[i, i]=T(1+i/max(1, n))
        end
        positive=lower*lower'
        a=selfadjoint_storage(matrix, uplo)
        b=selfadjoint_storage(positive, uplo; rows=n+4)
        priora, priorb=copy(a), copy(b)
        ra, rb=copy(a), copy(b)
        result=direct_selfadjoint!(a, b, J; family, itype, jobz, uplo)
        external=direct_selfadjoint!(
            ra, rb, Int64; family, itype, jobz, uplo, library=reference
        )
        @test result.info==external.info==0
        @test result.values ≈ external.values rtol=tolerance atol=tolerance
        @test result.values[[1, end]]==R[777, 777]
        @test result.work[[1, end]]==T[777, 777]
        @test result.rwork[[1, end]]==R[777, 777]
        @test result.iwork[[1, end]]==J[777, 777]
        @test selfadjoint_untouched(b, priorb, n, uplo)
        @test isequal(a[(n + 1):end, :], priora[(n + 1):end, :])
        factor=uplo=='L' ? tril(b[1:n, 1:n]) : triu(b[1:n, 1:n])
        @test (uplo=='L' ? factor*factor' : factor'*factor) ≈ positive atol=tolerance
        if jobz=='V'
            selfadjoint_equations(
                matrix, positive, result.values[2:(n + 1)], a[1:n, 1:n], itype, tolerance
            )
        else
            @test selfadjoint_untouched(a, priora, n, uplo)
        end
    end
end
@testset "Generalized self-adjoint range endpoints and multiplicities" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64), uplo in ('L', 'U'), itype in 1:3, jobz in ('N', 'V'),
        range in ('A', 'I', 'V')

        R=typeof(real(zero(T)))
        tolerance=R===Float32 ? 4e-4 : 4e-12
        values=R[-1, 0, 0, 2, 2, 4]
        n=length(values)
        factor=R[1, 2, 4, 1, 2, 4]
        positive=Matrix(Diagonal(T.(factor .^ 2)))
        matrix=Matrix(
            Diagonal(T.(itype==1 ? values .* factor .^ 2 : values ./ factor .^ 2))
        )
        a=selfadjoint_storage(matrix, uplo)
        b=selfadjoint_storage(positive, uplo; rows=n+4)
        priora=copy(a)
        priorb=copy(b)
        result=direct_selfadjoint!(
            a, b, J; family=:gvx, itype, jobz, uplo, range, vl=R(0), vu=R(2), il=2, iu=4
        )
        chosen=if range=='A'
            collect(1:n)
        elseif range=='I'
            collect(2:4)
        else
            [4, 5]
        end
        @test result.info==0
        @test result.m==length(chosen)
        m=Int(result.m)
        @test result.values[2:(m + 1)] ≈ values[chosen] atol=tolerance
        @test all(==(R(777)), result.values[(m + 2):end])
        @test selfadjoint_untouched(a, priora, n, uplo)
        @test selfadjoint_untouched(b, priorb, n, uplo)
        @test all(==(T(777)), result.z[(n + 1):end, :])
        @test all(==(T(777)), result.z[:, (m + 1):end])
        if jobz=='V'
            selfadjoint_equations(
                matrix,
                positive,
                result.values[2:(m + 1)],
                result.z[1:n, 1:m],
                itype,
                tolerance,
            )
            @test all(iszero, result.failed[2:(m + 1)])
        else
            @test all(==(T(777)), result.z)
            @test all(==(J(777)), result.failed)
        end
    end
end

@testset "Generalized self-adjoint queries and minimum work" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64), family in (:gv, :gvd, :gvx), jobz in ('N', 'V'),
        n in (0, 1, 4)

        R=typeof(real(zero(T)))
        a=selfadjoint_storage(Matrix{T}(I, n, n), 'L')
        b=copy(a)
        before=(copy(a), copy(b))
        sizes=selfadjoint_workspace(T, n, jobz, family)
        result=direct_selfadjoint!(a, b, J; family, jobz, lwork=-1, lrwork=-1, liwork=-1)
        @test result.info==0
        @test isequal(before, (a, b))
        @test all(==(R(777)), result.values)
        @test all(==(J(777)), result.failed)
        @test all(==(T(777)), result.z)
        @test result.m==-777
        @test real(result.work[2])>=sizes[1]
        if family==:gvd
            @test result.iwork[2]>=sizes[3]
            if T<:Complex
                @test result.rwork[2]>=sizes[2]
            end
        end
        @test direct_selfadjoint!(a, b, J; family, jobz, lwork=sizes[1]-1).info==(
            family==:gvx ? -20 : -11
        )
        if family==:gvd
            @test direct_selfadjoint!(a, b, J; family, jobz, liwork=sizes[3]-1).info==(
                T<:Real ? -13 : -15
            )
            if T<:Complex
                @test direct_selfadjoint!(a, b, J; family, jobz, lrwork=sizes[2]-1).info==-13
            end
        end
        result=direct_selfadjoint!(a, b, J; family, jobz)
        @test result.info==0
        @test result.work[[1, end]]==T[777, 777]
        @test result.rwork[[1, end]]==R[777, 777]
        @test result.iwork[[1, end]]==J[777, 777]
    end
end
@testset "Generalized self-adjoint argument and positive-definite failure contracts" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        family in (:gv, :gvd, :gvx)

        a=Matrix{T}(I, 4, 4)
        b=copy(a)
        for (options, expected) in (
            ((; itype=0), -1),
            ((; jobz=('?')), -2),
            ((; uplo=('?')), family==:gvx ? -4 : -3),
            ((; n=-1), family==:gvx ? -5 : -4),
            ((; lda=3), family==:gvx ? -7 : -6),
            ((; ldb=3), family==:gvx ? -9 : -8),
        )
            before=(copy(a), copy(b))
            @test direct_selfadjoint!(a, b, J; family, options...).info==expected
            @test isequal(before, (a, b))
        end
        if family==:gvx
            @test direct_selfadjoint!(a, b, J; family, range=('?')).info==-3
            @test direct_selfadjoint!(a, b, J; family, range=('V'), vl=1, vu=1).info==-11
            @test direct_selfadjoint!(a, b, J; family, range=('I'), il=0, iu=1).info==-12
            @test direct_selfadjoint!(a, b, J; family, range=('I'), il=1, iu=5).info==-13
            @test direct_selfadjoint!(a, b, J; family, ldz=3).info==-18
        end
        if J===Int64
            @test direct_selfadjoint!(a, b, J; family, n=(-(Int64(1)<<32))).info==(
                family==:gvx ? -5 : -4
            )
            @test direct_selfadjoint!(a, b, J; family, itype=(Int64(1)<<32)+1).info==-1
        end
        for uplo in ('L', 'U'), itype in 1:3
            a=selfadjoint_storage(Matrix{T}(I, 4, 4), uplo; poison=true)
            b=selfadjoint_storage(Matrix(Diagonal(T[2, 2, -1, 2])), uplo; poison=true)
            priora=copy(a)
            priorb=copy(b)
            result=direct_selfadjoint!(a, b, J; family, itype, uplo)
            @test result.info==7
            @test isequal(a, priora)
            @test selfadjoint_untouched(b, priorb, 4, uplo)
            @test all(==(typeof(real(zero(T)))(777)), result.values)
            @test all(==(T(777)), result.z)
            @test all(==(J(777)), result.failed)
            if family==:gvx
                @test result.m==0
            end
        end
    end
end
@testset "Generalized self-adjoint expert and failure fallback" begin
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
        matrix=T[2 0.25 0; 0.25 3 -0.125; 0 -0.125 4]
        positive=T[2 0.125 0; 0.125 3 0.25; 0 0.25 4]
        a=selfadjoint_storage(matrix, uplo)
        b=selfadjoint_storage(positive, uplo)
        ra, rb=copy(a), copy(b)
        result=direct_selfadjoint!(
            a, b, J; family=:gvx, itype, uplo, range=('I'), il=1, iu=2, abstol=eps(R)/2
        )
        expected=direct_selfadjoint!(
            ra,
            rb,
            Int64;
            family=:gvx,
            itype,
            uplo,
            range=('I'),
            il=1,
            iu=2,
            abstol=eps(R)/2,
            library=reference,
        )
        @test result.info==expected.info==0
        @test result.m==expected.m==2
        @test isequal(result.values, expected.values)
        @test isequal(result.z, expected.z)
        @test result.failed==expected.failed
        @test isequal(a, ra)
        @test isequal(b, rb)
    end
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        family in (:gv, :gvd)

        matrix=Matrix{T}(I, 4, 4)
        matrix[2, 2]=T(NaN)
        positive=Matrix(Diagonal(T[2, 3, 4, 5]))
        a=selfadjoint_storage(matrix, 'L')
        b=selfadjoint_storage(positive, 'L')
        ra, rb=copy(a), copy(b)
        result=direct_selfadjoint!(a, b, J; family, jobz=('N'))
        expected=direct_selfadjoint!(ra, rb, Int64; family, jobz=('N'), library=reference)
        @test result.info==expected.info
        @test isequal(result.values, expected.values)
        @test isequal(a, ra)
        @test isequal(b, rb)
    end
end

@testset "Generalized self-adjoint ignored components and LBT arrays" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64), family in (:gv, :gvd, :gvx), uplo in ('L', 'U'),
        itype in 1:3

        R=typeof(real(zero(T)))
        tolerance=R===Float32 ? 4e-4 : 4e-12
        raw=randn(T, 4, 4)/4
        matrix=Matrix(Hermitian(raw+raw'))
        lower=tril(randn(T, 4, 4)/4)
        for i in 1:4
            lower[i, i]=T(1+i/4)
        end
        positive=lower*lower'
        a=selfadjoint_storage(matrix, uplo; poison=true)
        b=selfadjoint_storage(positive, uplo; poison=true)
        priorb=copy(b)
        result=direct_selfadjoint!(a, b, J; family, itype, uplo)
        @test result.info==0
        selfadjoint_equations(
            matrix,
            positive,
            result.values[2:5],
            family==:gvx ? result.z[1:4, 1:4] : a[1:4, 1:4],
            itype,
            tolerance,
        )
        @test selfadjoint_untouched(b, priorb, 4, uplo)
        @test all(iszero, imag.(diag(b[1:4, 1:4])))
    end
    for (T, prefix) in
        ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
        J in (Int32, Int64),
        family in (:gv, :gvd, :gvx)

        symbol=prefix*(T<:Real ? "sy" : "he")*String(family)*"_"
        library=BLAS.lbt_find_backing_library(symbol, J===Int64 ? :ilp64 : :lp64)
        @test library!==nothing
        @test realpath(library.libname)==realpath(Colmat.backend_library())
        a=fill(T(2), 1, 1)
        b=fill(T(4), 1, 1)
        result=direct_selfadjoint!(a, b, J; family, library=BLAS.libblastrampoline)
        @test result.info==0
        @test result.values[2]≈0.5
    end
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        matrix=T[2 0.125 0; 0.125 3 0.25; 0 0.25 4]
        positive=T[2 0.25 0; 0.25 3 0.125; 0 0.125 4]
        result=eigen(Hermitian(matrix), Hermitian(positive))
        selfadjoint_equations(
            matrix,
            positive,
            result.values,
            result.vectors,
            1,
            T<:Union{Float32,ComplexF32} ? 4e-4 : 4e-12,
        )
    end
end

@testset "Isolated generalized self-adjoint fallback configuration" begin
    worker=joinpath(@__DIR__, "generalized_selfadjoint_fallback_worker.jl")
    project=dirname(Base.active_project())
    @test success(`$(Base.julia_cmd()) --startup-file=no --project=$project $worker`)
end

@testset "Generalized self-adjoint large D&C and empty selections" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        uplo in ('L', 'U')

        R=typeof(real(zero(T)))
        n=160
        matrix=Matrix(Diagonal(T.(1:n)))
        for i in 1:(n - 1)
            value=T<:Complex ? T(0.125, 0.0625) : T(0.125)
            matrix[i + 1, i]=value
            matrix[i, i + 1]=conj(value)
        end
        lower=Matrix(Diagonal(T.(sqrt.(1 .+ (1:n) ./ n))))
        for i in 1:(n - 1)
            lower[i + 1, i]=T<:Complex ? T(0.03125, 0.015625) : T(0.03125)
        end
        positive=lower*lower'
        a=selfadjoint_storage(matrix, uplo)
        b=selfadjoint_storage(positive, uplo)
        priorb=copy(b)
        result=direct_selfadjoint!(a, b, J; family=:gvd, uplo)
        vectors=a[1:n, 1:n]
        @test result.info==0
        @test selfadjoint_untouched(b, priorb, n, uplo)
        @test all(iszero, imag.(diag(b[1:n, 1:n])))
        @test matrix*vectors ≈ positive*vectors*Diagonal(result.values[2:(n + 1)]) atol=(
            R===Float32 ? 1e-2 : 2e-9
        )
        @test vectors'*positive*vectors ≈ Matrix{T}(I, n, n) atol=(
            R===Float32 ? 3e-4 : 3e-11
        )
        for range in ('I', 'V')
            a=Matrix{T}(undef, 3, 0)
            b=copy(a)
            result=direct_selfadjoint!(
                a, b, J; family=:gvx, range, il=1, iu=0, vl=R(2), vu=R(1)
            )
            @test result.info==0
            @test result.m==0
            @test all(==(T(777)), result.z)
        end
        result=direct_selfadjoint!(
            fill(T(2), 1, 1),
            fill(T(4), 1, 1),
            J;
            family=:gvx,
            range=('V'),
            vl=R(0.5),
            vu=R(1),
        )
        @test result.info==0
        @test result.m==0
        @test all(==(R(777)), result.values)
        @test all(==(T(777)), result.z)
    end
end
