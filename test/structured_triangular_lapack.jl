using Test, LinearAlgebra, Random, Libdl, Colmat
if !isdefined(@__MODULE__, :direct_storage_conversion!)
    include("storage_conversions.jl")
end
function st_diagnostic()
    return ccall(
        Libdl.dlsym(Libdl.dlopen(Colmat.backend_library()), :colmat_blas_last_error),
        Cint,
        (),
    )
end
for (T, prefix) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    J in (Int32, Int64)

    suffix=J===Int64 ? "64_" : ""
    @eval function direct_st_inverse!(
        storage::Vector{$T},
        ::Type{$J};
        rfp=false,
        trr=('N'),
        u=('L'),
        diag=('N'),
        n,
        library=Colmat.backend_library(),
    )
        fn=Libdl.dlsym(
            Libdl.dlopen(library), Symbol($prefix*(rfp ? "tftri_" : "tptri_")*$suffix)
        )
        ap=view(storage, 2:(length(storage) - 1))
        info=fill($J(-777), 3)
        ip=view(info, 2:2)
        if rfp
            ccall(
                fn,
                Cvoid,
                (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{$J}, Ptr{$T}, Ptr{$J}),
                trr,
                u,
                diag,
                n,
                ap,
                ip,
            )
        else
            ccall(
                fn,
                Cvoid,
                (Ref{UInt8}, Ref{UInt8}, Ref{$J}, Ptr{$T}, Ptr{$J}),
                u,
                diag,
                n,
                ap,
                ip,
            )
        end
        (; info=info[2], guard=info[[1, 3]], diagnostic=st_diagnostic())
    end
    @eval function direct_st_solve!(
        storage::Array{$T},
        b::Matrix{$T},
        ::Type{$J};
        band=false,
        u=('L'),
        trans=('N'),
        diag=('N'),
        n,
        kd=0,
        nrhs=size(b, 2),
        lda=band ? size(storage, 1) : 1,
        ldb=size(b, 1),
        library=Colmat.backend_library(),
    )
        fn=Libdl.dlsym(
            Libdl.dlopen(library), Symbol($prefix*(band ? "tbtrs_" : "tptrs_")*$suffix)
        )
        ap=band ? storage : view(storage, 2:(length(storage) - 1))
        info=fill($J(-777), 3)
        ip=view(info, 2:2)
        if band
            ccall(
                fn,
                Cvoid,
                (
                    Ref{UInt8},
                    Ref{UInt8},
                    Ref{UInt8},
                    Ref{$J},
                    Ref{$J},
                    Ref{$J},
                    Ptr{$T},
                    Ref{$J},
                    Ptr{$T},
                    Ref{$J},
                    Ptr{$J},
                ),
                u,
                trans,
                diag,
                n,
                kd,
                nrhs,
                ap,
                lda,
                b,
                ldb,
                ip,
            )
        else
            ccall(
                fn,
                Cvoid,
                (
                    Ref{UInt8},
                    Ref{UInt8},
                    Ref{UInt8},
                    Ref{$J},
                    Ref{$J},
                    Ptr{$T},
                    Ptr{$T},
                    Ref{$J},
                    Ptr{$J},
                ),
                u,
                trans,
                diag,
                n,
                nrhs,
                ap,
                b,
                ldb,
                ip,
            )
        end
        (; info=info[2], guard=info[[1, 3]], diagnostic=st_diagnostic())
    end
    @eval function direct_st_rfp_solve!(
        storage::Vector{$T},
        b::Matrix{$T},
        ::Type{$J};
        trr=('N'),
        side=('L'),
        u=('L'),
        trans=('N'),
        diag=('N'),
        m=size(b, 1),
        n=size(b, 2),
        alpha=one($T),
        ldb=size(b, 1),
        library=Colmat.backend_library(),
    )
        fn=Libdl.dlsym(Libdl.dlopen(library), Symbol($prefix*"tfsm_"*$suffix))
        ap=view(storage, 2:(length(storage) - 1))
        ccall(
            fn,
            Cvoid,
            (
                Ref{UInt8},
                Ref{UInt8},
                Ref{UInt8},
                Ref{UInt8},
                Ref{UInt8},
                Ref{$J},
                Ref{$J},
                Ref{$T},
                Ptr{$T},
                Ptr{$T},
                Ref{$J},
            ),
            trr,
            side,
            u,
            trans,
            diag,
            m,
            n,
            alpha,
            ap,
            b,
            ldb,
        )
        st_diagnostic()
    end
end
function st_pack(a, u, trr, rfp)
    n=size(a, 1)
    T=eltype(a)
    storage=fill(T(777), n*(n+1)÷2+2)
    @test direct_storage_conversion!(
        rfp ? "trttf" : "trttp",
        a,
        storage,
        Int64;
        n,
        uplo=u,
        transr=trr,
        lda=max(1, n),
        library=conversion_reference(),
    )==0
    return storage
end
function st_unpack(storage, n, u, trr, rfp)
    a=fill(eltype(storage)(NaN), max(1, n), n)
    @test direct_storage_conversion!(
        rfp ? "tfttr" : "tpttr",
        storage,
        a,
        Int64;
        n,
        uplo=u,
        transr=trr,
        library=conversion_reference(),
    )==0
    return u=='L' ? Matrix(LowerTriangular(a[1:n, :])) : Matrix(UpperTriangular(a[1:n, :]))
end
function st_input(T, n, u, diag, rng; k=n)
    a=zeros(T, n, n)
    for j in 1:n, i in 1:n
        if (u=='L' ? i>=j : i<=j) && abs(i-j)<=k
            a[i, j]=if i==j
                (diag=='U' ? one(T) : T(2+i)+(T<:Complex ? T(im/4) : zero(T)))
            else
                randn(rng, T)/5
            end
        end
    end
    return a
end
@testset "Packed and RFP triangular inverse" begin
    rng=MersenneTwister(298)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        n in (0, 1, 2, 5, 6),
        u in ('L', 'U'),
        diag in ('N', 'U'),
        rfp in (false, true),
        trr in (rfp ? (T<:Complex ? ('N', 'C') : ('N', 'T')) : ('N',))

        a=st_input(T, n, u, diag, rng)
        source=copy(a)
        diag=='U'&&foreach(i->source[i, i]=T(NaN), 1:n)
        storage=st_pack(source, u, trr, rfp)
        before=copy(storage)
        result=direct_st_inverse!(storage, J; rfp, trr, u, diag, n)
        @test result.info==result.diagnostic==0
        @test result.guard==fill(J(-777), 2)
        @test storage[[1, end]]==fill(T(777), 2)
        inva=st_unpack(storage, n, u, trr, rfp)
        if diag=='U'
            @test all(isnan, inva[diagind(inva)])
            foreach(i->inva[i, i]=one(T), 1:n)
        end
        @test a*inva≈Matrix{T}(I, n, n) atol=500eps(real(T)) rtol=500eps(real(T))
        external=copy(before)
        r=direct_st_inverse!(
            external, Int64; rfp, trr, u, diag, n, library=conversion_reference()
        )
        @test r.info==0
        reference=st_unpack(external, n, u, trr, rfp)
        diag=='U'&&foreach(i->reference[i, i]=one(T), 1:n)
        @test inva≈reference atol=500eps(real(T)) rtol=500eps(real(T))
    end
end
@testset "Band and packed triangular solves" begin
    rng=MersenneTwister(2298)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        n in (0, 1, 2, 6),
        u in ('L', 'U'),
        diag in ('N', 'U'),
        trans in ('N', 'T', 'C'),
        band in (false, true),
        kd in (band ? unique((0, min(2, max(0, n-1)), n+2)) : (n,))

        a=st_input(T, n, u, diag, rng; k=kd)
        source=copy(a)
        diag=='U'&&foreach(i->source[i, i]=T(NaN), 1:n)
        storage=if band
            ab=fill(T(NaN), kd+3, n)
            for j in 1:n, i in 1:n
                if (u=='L' ? i>=j : i<=j)&&abs(i-j)<=kd
                    ab[(u=='L' ? i-j+1 : kd+1+i-j), j]=source[i, j]
                end
            end
            ab
        else
            st_pack(source, u, 'N', false)
        end
        for nrhs in (0, 1, 3)
            rhs=randn(rng, T, n, nrhs)
            b=fill(T(777), n+2, nrhs)
            b[1:n, :]=rhs
            r=direct_st_solve!(storage, b, J; band, u, diag, trans, n, kd, nrhs)
            @test r.info==r.diagnostic==0
            @test r.guard==fill(J(-777), 2)
            op=if trans=='N'
                a
            elseif trans=='T'
                transpose(a)
            else
                a'
            end
            @test op*b[1:n, :]≈rhs atol=500eps(real(T)) rtol=500eps(real(T))
            @test all(==(T(777)), b[(n + 1):end, :])
        end
    end
end
@testset "RFP triangular solve from left and right" begin
    rng=MersenneTwister(4298)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        order in (0, 1, 2, 5, 6),
        side in ('L', 'R'),
        u in ('L', 'U'),
        diag in ('N', 'U'),
        trans in ('N', 'T', 'C'),
        trr in (T<:Complex ? ('N', 'C') : ('N', 'T'))

        a=st_input(T, order, u, diag, rng)
        source=copy(a)
        diag=='U'&&foreach(i->source[i, i]=T(NaN), 1:order)
        storage=st_pack(source, u, trr, true)
        m, n=side=='L' ? (order, 3) : (3, order)
        alpha=T(1.3)+(T<:Complex ? T(im/5) : zero(T))
        rhs=randn(rng, T, m, n)
        b=fill(T(777), m+2, n)
        b[1:m, :]=rhs
        diagnostic=direct_st_rfp_solve!(
            storage, b, J; trr, side, u, diag, trans, m, n, alpha
        )
        @test diagnostic==0
        op=if trans=='N'
            a
        elseif trans=='T'
            transpose(a)
        else
            a'
        end
        @test (side=='L' ? op*b[1:m, :] : b[1:m, :]*op)≈alpha*rhs atol=1000eps(real(T)) rtol=1000eps(
            real(T)
        )
        @test all(==(T(777)), b[(m + 1):end, :])
        @test storage[[1, end]]==fill(T(777), 2)
    end
end

@testset "Structured triangular diagnostics, singularity and unused inputs" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
        for rfp in (false, true)
            cases=if rfp
                (((; trr=('?')), 1), ((; u=('?')), 2), ((; diag=('?')), 3), ((; n=-1), 4))
            else
                (((; u=('?')), 1), ((; diag=('?')), 2), ((; n=-1), 3))
            end
            for (kw, arg) in cases
                a=fill(T(777), 8)
                r=direct_st_inverse!(a, J; rfp, n=3, kw...)
                @test r.info == -arg
                @test r.diagnostic == arg
                @test r.guard==fill(J(-777), 2)
                @test all(==(T(777)), a)
            end
        end
        for band in (false, true)
            cases=if band
                (
                ((; u=('?')), 1),
                ((; trans=('?')), 2),
                ((; diag=('?')), 3),
                ((; n=-1), 4),
                ((; kd=-1), 5),
                ((; nrhs=-1), 6),
                ((; lda=2), 8),
                ((; ldb=2), 10),
            )
            else
                (
                ((; u=('?')), 1),
                ((; trans=('?')), 2),
                ((; diag=('?')), 3),
                ((; n=-1), 4),
                ((; nrhs=-1), 5),
                ((; ldb=2), 8),
            )
            end
            for (kw, arg) in cases
                a=band ? fill(T(777), 5, 3) : fill(T(777), 8)
                b=fill(T(777), 5, 2)
                r=direct_st_solve!(a, b, J; band, n=3, kd=2, kw...)
                @test r.info == -arg
                @test r.diagnostic == arg
                @test r.guard==fill(J(-777), 2)
                @test all(==(T(777)), a)
                @test all(==(T(777)), b)
            end
        end
        for (kw, arg) in (
            ((; trr=('?')), 1),
            ((; side=('?')), 2),
            ((; u=('?')), 3),
            ((; trans=('?')), 4),
            ((; diag=('?')), 5),
            ((; m=-1), 6),
            ((; n=-1), 7),
            ((; ldb=2), 11),
        )
            a=fill(T(777), 8)
            b=fill(T(777), 5, 2)
            @test direct_st_rfp_solve!(a, b, J; m=3, n=2, kw...)==arg
            @test all(==(T(777)), a)
            @test all(==(T(777)), b)
        end
        # ALPHA=0 must avoid reading either A or old B.
        b=fill(T(NaN), 5, 2)
        @test direct_st_rfp_solve!(fill(T(777), 2), b, J; m=3, n=2, alpha=zero(T))==0
        @test iszero(b[1:3, :])
        @test all(isnan, b[4:end, :])
        for u in ('L', 'U'), zeroindex in (1, 3), nrhs in (0, 2)
            original=Matrix{T}(I, 4, 4)
            original[zeroindex, zeroindex]=zero(T)
            for rfp in (false, true)
                a=st_pack(original, u, 'N', rfp)
                before=copy(a)
                r=direct_st_inverse!(a, J; rfp, u, n=4)
                @test r.info==zeroindex
                @test r.diagnostic==0
                @test isequal(a, before)
            end
            a=st_pack(original, u, 'N', false)
            b=fill(T(777), 6, nrhs)
            before=copy(b)
            r=direct_st_solve!(a, b, J; u, n=4, nrhs)
            @test r.info==zeroindex
            @test r.diagnostic==0
            @test isequal(b, before)
        end
    end
end
@testset "All structured triangular symbols through LBT" begin
    rng=MersenneTwister(6298)
    for (T, prefix) in
        ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
        J in (Int32, Int64)

        for family in ("tbtrs", "tptrs", "tptri", "tftri", "tfsm")
            provider=BLAS.lbt_find_backing_library(
                prefix*family*"_", J===Int32 ? :lp64 : :ilp64
            )
            @test provider!==nothing
            @test realpath(provider.libname)==realpath(Colmat.backend_library())
        end
        for u in ('L', 'U')
            a=st_input(T, 5, u, 'N', rng)
            b=ones(T, 7, 2)
            packed=st_pack(a, u, 'N', false)
            r=direct_st_solve!(packed, b, J; u, n=5, library=BLAS.libblastrampoline)
            @test r.info==r.diagnostic==0
            @test a*b[1:5, :]≈ones(T, 5, 2)
            for rfp in (false, true)
                packed=st_pack(a, u, 'N', rfp)
                r=direct_st_inverse!(packed, J; rfp, u, n=5, library=BLAS.libblastrampoline)
                @test r.info==r.diagnostic==0
                @test a*st_unpack(packed, 5, u, 'N', rfp)≈Matrix{T}(I, 5, 5) atol=500eps(
                    real(T)
                ) rtol=500eps(real(T))
            end
            packed=st_pack(a, u, 'N', true)
            b=ones(T, 7, 2)
            @test direct_st_rfp_solve!(
                packed, b, J; u, m=5, n=2, library=BLAS.libblastrampoline
            )==0
            @test a*b[1:5, :]≈ones(T, 5, 2)
            ab=fill(T(NaN), 7, 5)
            for j in 1:5, i in 1:5
                if u=='L' ? i>=j : i<=j
                    ab[u=='L' ? i-j+1 : 5+i-j, j]=a[i, j]
                end
            end
            b=ones(T, 7, 2)
            r=direct_st_solve!(
                ab, b, J; band=true, u, n=5, kd=4, library=BLAS.libblastrampoline
            )
            @test r.info==r.diagnostic==0
            @test a*b[1:5, :]≈ones(T, 5, 2)
        end
    end
end
