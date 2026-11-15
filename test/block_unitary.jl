using Test, LinearAlgebra, Random, Libdl, Colmat

for (T, prefix, family) in (
        (Float32, "s", "orm22"),
        (Float64, "d", "orm22"),
        (ComplexF32, "c", "unm22"),
        (ComplexF64, "z", "unm22"),
    ),
    J in (Int32, Int64)

    symbol=Symbol(prefix*family*"_"*(J===Int64 ? "64_" : ""))
    @eval function direct_block_unitary!(
        q::Matrix{$T},
        c::Matrix{$T},
        ::Type{$J};
        m,
        n,
        n1,
        n2,
        side=('L'),
        trans=('N'),
        ldq=size(q, 1),
        ldc=size(c, 1),
        lwork=nothing,
        library=Colmat.backend_library(),
    )
        nq=side=='L' ? m : n
        minimum=n1==0||n2==0 ? 1 : nq
        lw=isnothing(lwork) ? minimum : lwork
        work=fill($T(777), max(1, lw)+2)
        info=fill($J(-777), 3)
        wp=view(work, 2:(length(work) - 1))
        ip=view(info, 2:2)
        f=Libdl.dlsym(Libdl.dlopen(library), $(QuoteNode(symbol)))
        ccall(
            f,
            Cvoid,
            (
                Ref{UInt8},
                Ref{UInt8},
                Ref{$J},
                Ref{$J},
                Ref{$J},
                Ref{$J},
                Ptr{$T},
                Ref{$J},
                Ptr{$T},
                Ref{$J},
                Ptr{$T},
                Ref{$J},
                Ptr{$J},
            ),
            side,
            trans,
            m,
            n,
            n1,
            n2,
            q,
            ldq,
            c,
            ldc,
            wp,
            lw,
            ip,
        )
        diagnostic=ccall(
            Libdl.dlsym(Libdl.dlopen(Colmat.backend_library()), :colmat_blas_last_error),
            Cint,
            (),
        )
        return (; info=info[2], guard=info[[1, 3]], work, diagnostic)
    end
end

function block_unitary_matrices(T, n1, n2, rng; unitary=false)
    nq=n1+n2
    dense=unitary ? zeros(T, nq, nq) : randn(rng, T, nq, nq)
    if unitary
        for i in 1:n1
            dense[i, n2 + i]=T<:Complex ? T(cis(rand(rng))) : T(rand(rng, Bool) ? 1 : -1)
        end
        for i in 1:n2
            dense[n1 + i, i]=T<:Complex ? T(cis(rand(rng))) : T(rand(rng, Bool) ? 1 : -1)
        end
    end
    stored=fill(T(777), max(1, nq+2), max(1, nq))
    for j in 1:nq, i in 1:nq
        unused=(i<=n1&&j>n2&&i<j-n2)||(i>n1&&j<=n2&&i-n1>j)
        if unused
            dense[i, j]=zero(T)
            stored[i, j]=T(NaN)
        else
            stored[i, j]=dense[i, j]
        end
    end
    return dense, stored
end

function block_scalar_product(a, b)
    return [
        sum(
            (a[i, k]*b[k, j] for k in axes(a, 2));
            init=zero(promote_type(eltype(a), eltype(b))),
        ) for i in axes(a, 1), j in axes(b, 2)
    ]
end

@testset "ORM22 UNM22 four types both integer ABIs and independent products" begin
    reference=only(
        filter(
            l->l.interface==:ilp64&&occursin("openblas", lowercase(basename(l.libname))),
            BLAS.get_config().loaded_libs,
        ),
    ).libname
    rng=MersenneTwister(287)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        (m, n) in ((0, 4), (4, 0), (1, 1), (3, 7), (7, 3), (5, 5)),
        side in ('L', 'R'),
        unitary in (false, true)

        nq=side=='L' ? m : n
        R=real(T)
        tolerance=R===Float32 ? 5e-5 : 1e-12
        for n1 in unique((0, nq÷2, nq)), trans in (T<:Complex ? ('N', 'C') : ('N', 'T'))
            n2=nq-n1
            dense, q=block_unitary_matrices(T, n1, n2, rng; unitary)
            before_q=copy(q)
            original=randn(rng, T, m, n)
            opq=trans=='N' ? dense : adjoint(dense)
            expected=if side=='L'
                block_scalar_product(opq, original)
            else
                block_scalar_product(original, opq)
            end
            minimum=n1==0||n2==0 ? 1 : nq
            for lwork in unique((minimum, max(minimum, m*n), max(minimum, 3m*n+7)))
                c=fill(T(777), max(1, m+2), max(1, n))
                c[1:m, 1:n]=original
                refc=copy(c)
                r=direct_block_unitary!(q, c, J; m, n, n1, n2, side, trans, lwork)
                external=direct_block_unitary!(
                    q, refc, Int64; m, n, n1, n2, side, trans, lwork, library=reference
                )
                @test r.info==external.info==0
                @test r.diagnostic==0
                @test c[1:m, 1:n]≈expected atol=tolerance rtol=tolerance
                @test c[1:m, 1:n]≈refc[1:m, 1:n] atol=tolerance rtol=tolerance
                @test all(==(T(777)), c[(m + 1):end, :])
                n==0&&(@test all(==(T(777)), c))
                @test isequal(q, before_q)
                @test r.guard==J[-777, -777]
                @test r.work[[1, end]]==T[777, 777]
                @test real(r.work[2])==(m==0||n==0||n1==0||n2==0 ? 1 : m*n)
                unitary&&(@test norm(c[1:m, 1:n])≈norm(original) atol=tolerance rtol=tolerance)
            end
            c=fill(T(NaN), max(1, m+2), max(1, n))
            before=copy(c)
            r=direct_block_unitary!(q, c, J; m, n, n1, n2, side, trans, lwork=-1)
            @test r.info==0
            @test r.diagnostic==0
            @test real(r.work[2])>=m*n
            @test isequal(c, before)
            @test isequal(q, before_q)
        end
    end
end

@testset "ORM22 UNM22 invalid INFO precedence and workspace guards" begin
    rng=MersenneTwister(1287)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        side in ('L', 'R')

        m=4
        n=3
        nq=side=='L' ? m : n
        n1=1
        n2=nq-1
        _, q=block_unitary_matrices(T, n1, n2, rng)
        c=fill(T(NaN), m+2, n)
        before_q=copy(q)
        for (options, status) in (
            ((; side=('?')), -1),
            ((; trans=('?')), -2),
            ((; trans=T<:Complex ? 'T' : 'C'), -2),
            ((; m=-1), -3),
            ((; n=-1), -4),
            ((; n1=-1), -5),
            ((; n1=0), -5),
            ((; n1=nq+1, n2=-1), -6),
            ((; ldq=nq-1), -8),
            ((; ldc=m-1), -10),
            ((; lwork=nq-1), -12),
        )
            opts=(; m, n, n1, n2, side, options...)
            r=direct_block_unitary!(q, c, J; opts...)
            @test r.info==status
            @test r.diagnostic == -status
            @test all(isnan, c)
            @test isequal(q, before_q)
            @test r.guard==J[-777, -777]
            @test all(==(T(777)), r.work)
        end
    end
end

@testset "All four block application symbols execute through both LBT interfaces" begin
    rng=MersenneTwister(2287)
    for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
        prefix=if T===Float32
            "s"
        elseif T===Float64
            "d"
        elseif T===ComplexF32
            "c"
        else
            "z"
        end
        symbol=prefix*(T<:Complex ? "unm22_" : "orm22_")
        provider=BLAS.lbt_find_backing_library(symbol, J===Int32 ? :lp64 : :ilp64)
        @test provider!==nothing
        @test realpath(provider.libname)==realpath(Colmat.backend_library())
        dense, q=block_unitary_matrices(T, 2, 3, rng; unitary=true)
        c=randn(rng, T, 5, 2)
        original=copy(c)
        r=direct_block_unitary!(
            q, c, J; m=5, n=2, n1=2, n2=3, library=BLAS.libblastrampoline
        )
        @test r.info==0
        @test r.diagnostic==0
        @test c≈block_scalar_product(dense, original)
    end
end
