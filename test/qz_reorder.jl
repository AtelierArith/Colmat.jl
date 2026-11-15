using Test, LinearAlgebra, Random, Libdl, Colmat

for (T, prefix) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    J in (Int32, Int64)

    R=typeof(real(zero(T)))
    eigtypes=T<:Real ? [:(Ptr{$T}), :(Ptr{$T}), :(Ptr{$T})] : [:(Ptr{$T}), :(Ptr{$T})]
    eigargs=T<:Real ? [:ar, :ai, :beta] : [:ar, :beta]
    signature=Expr(
        :tuple,
        :(Ref{$J}),
        :(Ref{$J}),
        :(Ref{$J}),
        :(Ptr{$J}),
        :(Ref{$J}),
        :(Ptr{$T}),
        :(Ref{$J}),
        :(Ptr{$T}),
        :(Ref{$J}),
        eigtypes...,
        :(Ptr{$T}),
        :(Ref{$J}),
        :(Ptr{$T}),
        :(Ref{$J}),
        :(Ref{$J}),
        :(Ptr{$R}),
        :(Ptr{$R}),
        :(Ptr{$R}),
        :(Ptr{$T}),
        :(Ref{$J}),
        :(Ptr{$J}),
        :(Ref{$J}),
        :(Ref{$J}),
    )
    symbol=Symbol(prefix*"tgsen_"*(J===Int64 ? "64_" : ""))
    @eval function direct_tgsen!(
        a::Matrix{$T},
        b::Matrix{$T},
        q::Matrix{$T},
        z::Matrix{$T},
        mask,
        ::Type{$J};
        job=0,
        wantq=true,
        wantz=true,
        query=false,
        pointer=nothing,
    )
        n=size(a, 2)
        ar=fill($T(777), n)
        ai=copy(ar)
        beta=copy(ar)
        selected=$J.(mask)
        m=Ref{$J}(-777)
        info=Ref{$J}(-777)
        pl=fill($R(777), 1)
        pr=copy(pl)
        dif=fill($R(777), 2)
        work=fill($T(777), max(1, 4n*n+4n+16))
        iw=fill($J(-777), max(1, 2n*n+n+6))
        f=if isnothing(pointer)
            Libdl.dlsym(Libdl.dlopen(Colmat.backend_library()), $(QuoteNode(symbol)))
        else
            pointer
        end
        ccall(
            f,
            Cvoid,
            $signature,
            job,
            wantq,
            wantz,
            selected,
            n,
            a,
            max(1, size(a, 1)),
            b,
            max(1, size(b, 1)),
            $(eigargs...),
            q,
            max(1, size(q, 1)),
            z,
            max(1, size(z, 1)),
            m,
            pl,
            pr,
            dif,
            work,
            query ? -1 : length(work),
            iw,
            length(iw),
            info,
        )
        return (;
            alpha=($T<:Real ? complex.(ar, ai) : ar),
            beta,
            m=m[],
            info=info[],
            pl=pl[1],
            pr=pr[1],
            dif,
            work,
            iw,
            diagnostic=ccall(
                Libdl.dlsym(
                    Libdl.dlopen(Colmat.backend_library()), :colmat_blas_last_error
                ),
                Cint,
                (),
            ),
        )
    end
end

@testset "TGSEN native reordering and expert estimates" begin
    Random.seed!(307)
    external=only(
        filter(
            l->l.interface==:ilp64 && occursin("openblas", lowercase(basename(l.libname))),
            BLAS.get_config().loaded_libs,
        ),
    )
    reference=Libdl.dlopen(external.libname)
    for (T, prefix) in
        ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
        J in (Int32, Int64), job in 0:5, n in (0, 1, 3, 6),
        want in (false, true)

        R=typeof(real(zero(T)))
        tol=R===Float32 ? 5e-4 : 5e-11
        a=zeros(T, n+2, n)
        b=copy(a)
        for j in 1:n, i in 1:n
            a[i, j]=i==j ? T(i) : (i<j ? T(0.03*(i+j)) : zero(T))
            b[i, j]=i==j ? one(T) : (i<j ? T(0.01*(i+j)) : zero(T))
        end
        if T<:Real && n>=3
            a[1, 1]=a[2, 2]=0
            a[2, 1]=1
            a[1, 2]=-1
            b[1, 2]=0
        end
        a[(n + 1):end, :].=777
        b[(n + 1):end, :].=777
        mask=[iseven(i) for i in 1:n]
        q0=Matrix(qr(randn(T, n, n)).Q)
        z0=Matrix(qr(randn(T, n, n)).Q)
        q=copy(q0)
        z=copy(z0)
        ar=copy(a)
        br=copy(b)
        qref=copy(q0)
        zr=copy(z0)
        olda=copy(a[1:n, :])
        oldb=copy(b[1:n, :])
        result=direct_tgsen!(a, b, q, z, mask, J; job, wantq=want, wantz=want)
        oracle=direct_tgsen!(
            ar,
            br,
            qref,
            zr,
            mask,
            Int64;
            job,
            wantq=want,
            wantz=want,
            pointer=Libdl.dlsym(reference, Symbol(prefix*"tgsen_64_")),
        )
        @test result.info==oracle.info==0
        @test result.diagnostic==0
        @test result.m==oracle.m
        @test all(==(T(777)), a[(n + 1):end, :]) && all(==(T(777)), b[(n + 1):end, :])
        @test sort(result.alpha ./ result.beta; by=x->(real(x), imag(x))) ≈
            sort(oracle.alpha ./ oracle.beta; by=x->(real(x), imag(x))) rtol=tol atol=tol
        if want
            left=q0'*q
            right=z0'*z
            @test left*a[1:n, :]*right' ≈ olda rtol=tol atol=tol
            @test left*b[1:n, :]*right' ≈ oldb rtol=tol atol=tol
            @test q'*q ≈ Matrix{T}(I, n, n) rtol=tol atol=tol
            @test z'*z ≈ Matrix{T}(I, n, n) rtol=tol atol=tol
        else
            @test q==q0 && z==z0
        end
        if job in (1, 4, 5)
            @test result.pl ≈ oracle.pl rtol=tol atol=tol
            @test result.pr ≈ oracle.pr rtol=tol atol=tol
        else
            @test result.pl==result.pr==R(777)
        end
        if job in (2, 3, 4, 5)
            @test result.dif ≈ oracle.dif rtol=tol atol=tol
        else
            @test result.dif==fill(R(777), 2)
        end
        @test result.work[1]==oracle.work[1]
        @test result.iw[1]==oracle.iw[1]
    end
end

for (T, prefix) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    J in (Int32, Int64)

    worktypes=T<:Real ? [:(Ptr{$T}), :(Ref{$J})] : []
    workargs=T<:Real ? [:work, :(query ? -1 : length(work))] : []
    signature=Expr(
        :tuple,
        :(Ref{$J}),
        :(Ref{$J}),
        :(Ref{$J}),
        :(Ptr{$T}),
        :(Ref{$J}),
        :(Ptr{$T}),
        :(Ref{$J}),
        :(Ptr{$T}),
        :(Ref{$J}),
        :(Ptr{$T}),
        :(Ref{$J}),
        :(Ref{$J}),
        :(Ref{$J}),
        worktypes...,
        :(Ref{$J}),
    )
    symbol=Symbol(prefix*"tgexc_"*(J===Int64 ? "64_" : ""))
    @eval function direct_tgexc!(
        a::Matrix{$T},
        b::Matrix{$T},
        q::Matrix{$T},
        z::Matrix{$T},
        ::Type{$J};
        from=1,
        to=size(a, 2),
        wantq=true,
        wantz=true,
        query=false,
        pointer=nothing,
    )
        n=size(a, 2)
        first=Ref{$J}(from)
        last=Ref{$J}(to)
        info=Ref{$J}(-777)
        work=fill($T(777), max(1, 4n+16))
        f=if isnothing(pointer)
            Libdl.dlsym(Libdl.dlopen(Colmat.backend_library()), $(QuoteNode(symbol)))
        else
            pointer
        end
        ccall(
            f,
            Cvoid,
            $signature,
            wantq,
            wantz,
            n,
            a,
            max(1, size(a, 1)),
            b,
            max(1, size(b, 1)),
            q,
            max(1, size(q, 1)),
            z,
            max(1, size(z, 1)),
            first,
            last,
            $(workargs...),
            info,
        )
        diagnostic=ccall(
            Libdl.dlsym(Libdl.dlopen(Colmat.backend_library()), :colmat_blas_last_error),
            Cint,
            (),
        )
        return (; first=first[], last=last[], info=info[], work, diagnostic)
    end
end
@testset "TGEXC direct ABIs preserve pencils and ignored factors" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        (from, to) in ((1, 3), (3, 1), (2, 2)),
        wantq in (false, true),
        wantz in (false, true)

        R=typeof(real(zero(T)))
        tol=R===Float32 ? 5e-4 : 5e-11
        n=3
        a=zeros(T, 5, n)
        b=copy(a)
        for i in 1:n
            a[i, i]=T(i)
            b[i, i]=one(T)
        end
        a[1, 2]=T(0.25)
        a[2, 3]=T(-0.125)
        a[4:5, :].=777
        b[4:5, :].=777
        olda=copy(a[1:n, :])
        oldb=copy(b[1:n, :])
        q=wantq ? Matrix{T}(I, n, n) : fill(T(777), 1, 1)
        z=wantz ? Matrix{T}(I, n, n) : fill(T(777), 1, 1)
        result=direct_tgexc!(a, b, q, z, J; from, to, wantq, wantz)
        @test result.info==result.diagnostic==0
        @test result.first==from && result.last==to
        @test a[to, to]/b[to, to] ≈ T(from) rtol=tol atol=tol
        @test all(==(T(777)), a[4:5, :]) && all(==(T(777)), b[4:5, :])
        wantq || (@test q==fill(T(777), 1, 1))
        wantz || (@test z==fill(T(777), 1, 1))
        if wantq && wantz
            @test q*a[1:n, :]*z' ≈ olda rtol=tol atol=tol
            @test q*b[1:n, :]*z' ≈ oldb rtol=tol atol=tol
        end
    end
end

for (T, prefix) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    J in (Int32, Int64)

    sizetypes=T<:Real ? [:(Ref{$J}), :(Ref{$J}), :(Ptr{$T}), :(Ref{$J})] : []
    sizeargs=if T<:Real
        [:n1, :n2, :work, :(isnothing(lwork_override) ? length(work) : lwork_override)]
    else
        []
    end
    signature=Expr(
        :tuple,
        :(Ref{$J}),
        :(Ref{$J}),
        :(Ref{$J}),
        :(Ptr{$T}),
        :(Ref{$J}),
        :(Ptr{$T}),
        :(Ref{$J}),
        :(Ptr{$T}),
        :(Ref{$J}),
        :(Ptr{$T}),
        :(Ref{$J}),
        :(Ref{$J}),
        sizetypes...,
        :(Ref{$J}),
    )
    symbol=Symbol(prefix*"tgex2_"*(J===Int64 ? "64_" : ""))
    @eval function direct_tgex2!(
        a::Matrix{$T},
        b::Matrix{$T},
        q::Matrix{$T},
        z::Matrix{$T},
        ::Type{$J};
        j1=1,
        n1=1,
        n2=1,
        wantq=true,
        wantz=true,
        lwork_override=nothing,
        pointer=nothing,
    )
        n=size(a, 2)
        info=Ref{$J}(-777)
        work=fill($T(777), max(1, n*(n1+n2), 2*(n1+n2)^2))
        f=if isnothing(pointer)
            Libdl.dlsym(Libdl.dlopen(Colmat.backend_library()), $(QuoteNode(symbol)))
        else
            pointer
        end
        ccall(
            f,
            Cvoid,
            $signature,
            wantq,
            wantz,
            n,
            a,
            max(1, size(a, 1)),
            b,
            max(1, size(b, 1)),
            q,
            max(1, size(q, 1)),
            z,
            max(1, size(z, 1)),
            j1,
            $(sizeargs...),
            info,
        )
        diagnostic=ccall(
            Libdl.dlsym(Libdl.dlopen(Colmat.backend_library()), :colmat_blas_last_error),
            Cint,
            (),
        )
        return (; info=info[], work, diagnostic)
    end
end
@testset "TGEX2 adjacent scalar and conjugate blocks" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        (n1, n2) in (T<:Real ? ((1, 1), (1, 2), (2, 1), (2, 2)) : ((1, 1),)),
        want in (false, true)

        R=typeof(real(zero(T)))
        tol=R===Float32 ? 1e-3 : 1e-10
        n=n1+n2+2
        j1=2
        a=zeros(T, n+2, n)
        b=copy(a)
        for i in 1:n
            a[i, i]=T(10+i)
            b[i, i]=one(T)
        end
        if n1==2
            a[2, 2]=a[3, 3]=0
            a[3, 2]=1
            a[2, 3]=-1
        end
        k=j1+n1
        if n2==2
            a[k, k]=a[k + 1, k + 1]=2
            a[k + 1, k]=2
            a[k, k + 1]=-2
        end
        a[(n + 1):end, :].=777
        b[(n + 1):end, :].=777
        olda=copy(a[1:n, :])
        oldb=copy(b[1:n, :])
        target=sort(
            eigvals(
                olda[k:(k + n2 - 1), k:(k + n2 - 1)], oldb[k:(k + n2 - 1), k:(k + n2 - 1)]
            );
            by=x->imag(x),
        )
        q=want ? Matrix{T}(I, n, n) : fill(T(777), 1, 1)
        z=want ? Matrix{T}(I, n, n) : fill(T(777), 1, 1)
        result=direct_tgex2!(a, b, q, z, J; j1, n1, n2, wantq=want, wantz=want)
        @test result.info==result.diagnostic==0
        actual=sort(
            eigvals(
                a[j1:(j1 + n2 - 1), j1:(j1 + n2 - 1)], b[j1:(j1 + n2 - 1), j1:(j1 + n2 - 1)]
            );
            by=x->imag(x),
        )
        @test actual ≈ target rtol=tol atol=tol
        @test all(==(T(777)), a[(n + 1):end, :]) && all(==(T(777)), b[(n + 1):end, :])
        if want
            @test q*a[1:n, :]*z' ≈ olda rtol=tol atol=tol
            @test q*b[1:n, :]*z' ≈ oldb rtol=tol atol=tol
            @test q'*q ≈ Matrix{T}(I, n, n) rtol=tol atol=tol
            @test z'*z ≈ Matrix{T}(I, n, n) rtol=tol atol=tol
        else
            @test q==z==fill(T(777), 1, 1)
        end
    end
end
@testset "QZ reorder workspace contracts" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
        a=Matrix{T}(I, 3, 3)
        b=copy(a)
        q=copy(a)
        z=copy(a)
        result=direct_tgsen!(a, b, q, z, [false, true, false], J; query=true)
        @test result.info==result.diagnostic==result.m==0
        @test result.work[1]==(T<:Real ? 28 : 1)
        @test result.iw[1]==1
        @test a==b==q==z==Matrix{T}(I, 3, 3)
        @test all(==(T(777)), result.beta)
        if T<:Real
            result=direct_tgex2!(a, b, q, z, J; lwork_override=-1)
            @test result.info == -16 && result.diagnostic==16
            @test result.work[1]==8
            @test a==b==q==z==Matrix{T}(I, 3, 3)
            result=direct_tgexc!(a, b, q, z, J; query=true)
            @test result.info==result.diagnostic==0 && result.work[1]==28
            @test a==b==q==z==Matrix{T}(I, 3, 3)
        end
    end
end

@testset "QZ reorder LBT providers and forwarded calls" begin
    for (T, prefix) in
        ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
        J in (Int32, Int64),
        family in ("tgex2", "tgexc", "tgsen")

        interface=J===Int64 ? :ilp64 : :lp64
        symbol=prefix*family*"_"
        provider=BLAS.lbt_find_backing_library(symbol, interface)
        @test provider!==nothing
        @test realpath(provider.libname)==realpath(Colmat.backend_library())
        pointer=BLAS.lbt_get_forward(symbol, interface)
        a=Matrix(Diagonal(T[1, 3]))
        b=Matrix{T}(I, 2, 2)
        q=copy(b)
        z=copy(b)
        result=if family=="tgsen"
            direct_tgsen!(a, b, q, z, [false, true], J; pointer)
        elseif family=="tgexc"
            direct_tgexc!(a, b, q, z, J; from=2, to=1, pointer)
        else
            direct_tgex2!(a, b, q, z, J; pointer)
        end
        @test result.info==result.diagnostic==0
        @test a[1, 1]/b[1, 1]≈T(3)
    end
end
