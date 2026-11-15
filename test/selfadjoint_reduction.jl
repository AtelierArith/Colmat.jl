using Test, LinearAlgebra, Random, Libdl, Colmat

function tridiagonal_diagnostic()
    return ccall(
        Libdl.dlsym(Libdl.dlopen(Colmat.backend_library()), :colmat_blas_last_error),
        Cint,
        (),
    )
end
for (T, prefix) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    J in (Int32, Int64)

    R=real(T)
    suffix=J===Int64 ? "64_" : ""
    for blocked in (false, true)
        family=T<:Complex ? (blocked ? "hetrd" : "hetd2") : (blocked ? "sytrd" : "sytd2")
        symbol=Symbol(prefix*family*"_"*suffix)
        signature=Any[Ref{UInt8}, Ref{J}, Ptr{T}, Ref{J}, Ptr{R}, Ptr{R}, Ptr{T}]
        args=Any[:uplo, :n, :a, :lda, :d, :e, :tau]
        if blocked
            append!(signature, [Ptr{T}, Ref{J}])
            append!(args, [:work, :lw])
        end
        push!(signature, Ptr{J})
        push!(args, :ip)
        @eval function direct_tridiagonal!(
            a::Matrix{$T},
            ::Type{$J},
            ::Val{$blocked};
            uplo=('L'),
            n=size(a, 2),
            lda=size(a, 1),
            lwork=1,
            library=Colmat.backend_library(),
        )
            d=fill($R(777), max(1, n))
            e=copy(d)
            tau=fill($T(777), max(1, n))
            work=fill($T(777), max(1, lwork))
            lw=lwork
            info=fill($J(-777), 3)
            ip=view(info, 2:2)
            f=Libdl.dlsym(Libdl.dlopen(library), $(QuoteNode(symbol)))
            ccall(f, Cvoid, $(Expr(:tuple, signature...)), $(args...))
            (;
                d,
                e,
                tau,
                work,
                info=info[2],
                guard=info[[1, 3]],
                diagnostic=tridiagonal_diagnostic(),
            )
        end
    end
    symbol=Symbol(prefix*(T<:Complex ? "ungtr_" : "orgtr_")*suffix)
    @eval function direct_tridiagonal_generate!(
        a::Matrix{$T},
        tau::AbstractVector{$T},
        ::Type{$J};
        uplo=('L'),
        n=size(a, 2),
        lda=size(a, 1),
        lwork=max(1, n-1),
        library=Colmat.backend_library(),
    )
        work=fill($T(777), max(1, lwork))
        info=fill($J(-777), 3)
        ip=view(info, 2:2)
        f=Libdl.dlsym(Libdl.dlopen(library), $(QuoteNode(symbol)))
        ccall(
            f,
            Cvoid,
            (Ref{UInt8}, Ref{$J}, Ptr{$T}, Ref{$J}, Ptr{$T}, Ptr{$T}, Ref{$J}, Ptr{$J}),
            uplo,
            n,
            a,
            lda,
            tau,
            work,
            lwork,
            ip,
        )
        (; work, info=info[2], guard=info[[1, 3]], diagnostic=tridiagonal_diagnostic())
    end
    symbol=Symbol(prefix*(T<:Complex ? "unmtr_" : "ormtr_")*suffix)
    @eval function direct_tridiagonal_apply!(
        a::Matrix{$T},
        tau::AbstractVector{$T},
        c::Matrix{$T},
        ::Type{$J};
        side=('L'),
        uplo=('L'),
        trans=('N'),
        m=size(c, 1),
        n=size(c, 2),
        lda=size(a, 1),
        ldc=size(c, 1),
        lwork=max(1, side=='L' ? n : m),
        library=Colmat.backend_library(),
    )
        work=fill($T(777), max(1, lwork))
        info=fill($J(-777), 3)
        ip=view(info, 2:2)
        f=Libdl.dlsym(Libdl.dlopen(library), $(QuoteNode(symbol)))
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
                Ptr{$T},
                Ref{$J},
                Ptr{$T},
                Ref{$J},
                Ptr{$J},
            ),
            side,
            uplo,
            trans,
            m,
            n,
            a,
            lda,
            tau,
            c,
            ldc,
            work,
            lwork,
            ip,
        )
        (; work, info=info[2], guard=info[[1, 3]], diagnostic=tridiagonal_diagnostic())
    end
end

@testset "Self-adjoint tridiagonal reduction and Q" begin
    rng=MersenneTwister(296)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64), uplo in ('L', 'U'), blocked in (false, true),
        n in (0, 1, 2, 5, 8)

        R=real(T)
        tol=300eps(R)
        input=randn(rng, T, n, n)
        original=Matrix(Hermitian(input, uplo=='L' ? :L : :U))
        a=fill(T(777), n+2, n)
        for j in 1:n, i in 1:n
            selected=uplo=='L' ? i>=j : i<=j
            a[i, j]=selected ? original[i, j] : T(NaN)
            if T<:Complex && i==j
                a[i, j]=complex(real(original[i, j]), R(NaN))
            end
        end
        initial=copy(a)
        result=direct_tridiagonal!(a, J, Val(blocked); uplo, n)
        @test result.info==0
        @test result.diagnostic==0
        @test result.guard==fill(J(-777), 2)
        @test a[(n + 1):end, :]==initial[(n + 1):end, :]
        for j in 1:n, i in 1:n
            if uplo=='L' ? i<j : i>j
                @test isequal(a[i, j], initial[i, j])
            end
        end
        q=copy(a)
        generated=direct_tridiagonal_generate!(q, result.tau, J; uplo, n)
        @test generated.info==0
        @test generated.diagnostic==0
        @test generated.guard==fill(J(-777), 2)
        @test q[(n + 1):end, :]==initial[(n + 1):end, :]
        Q=q[1:n, :]
        tri=SymTridiagonal(result.d[1:n], result.e[1:max(0, n - 1)])
        @test Q'*Q≈Matrix{T}(I, n, n) atol=tol rtol=tol
        @test Q*tri*Q'≈original atol=tol rtol=tol
        for side in ('L', 'R'), trans in ('N', T<:Complex ? 'C' : 'T')
            m, k=side=='L' ? (n, 3) : (3, n)
            c=fill(T(777), m+2, k)
            c[1:m, :]=randn(rng, T, m, k)
            before=copy(c)
            op=trans=='N' ? Q : Q'
            want=side=='L' ? op*before[1:m, :] : before[1:m, :]*op
            applied=direct_tridiagonal_apply!(
                a, result.tau, c, J; side, uplo, trans, m, n=k
            )
            @test applied.info==0
            @test applied.diagnostic==0
            @test applied.guard==fill(J(-777), 2)
            @test c[1:m, :]≈want atol=tol rtol=tol
            @test c[(m + 1):end, :]==before[(m + 1):end, :]
        end
    end
end

function tridiagonal_reference()
    return only(
        filter(
            l ->
                l.interface == :ilp64 &&
                occursin("openblas", lowercase(basename(l.libname))),
            BLAS.get_config().loaded_libs,
        ),
    ).libname
end
@testset "External self-adjoint factors and Q interoperability" begin
    reference=tridiagonal_reference()
    rng=MersenneTwister(6296)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        uplo in ('L', 'U'),
        blocked in (false, true),
        mode in (:dense, :zero, :diagonal),
        n in (1, 2, 7)

        R=real(T)
        tol=500eps(R)
        original=if mode==:zero
            zeros(T, n, n)
        elseif mode==:diagonal
            Matrix(Diagonal(T.(1:n)))
        else
            Matrix(Hermitian(randn(rng, T, n, n)))
        end
        own_a=copy(original)
        ext_a=copy(original)
        own=direct_tridiagonal!(own_a, J, Val(blocked); uplo, n)
        ext=direct_tridiagonal!(
            ext_a, Int64, Val(blocked); uplo, n, lwork=max(1, n), library=reference
        )
        @test own.info==ext.info==0
        @test own.diagnostic==0
        for (a, factor) in ((own_a, own), (ext_a, ext))
            for library in (Colmat.backend_library(), reference)
                q=copy(a)
                result=direct_tridiagonal_generate!(
                    q, factor.tau, library==reference ? Int64 : J; uplo, n, library
                )
                @test result.info==0
                @test result.diagnostic==0
                tri=SymTridiagonal(factor.d[1:n], factor.e[1:(n - 1)])
                @test q*tri*q'≈original atol=tol rtol=tol
                for side in ('L', 'R'), trans in ('N', T<:Complex ? 'C' : 'T')
                    m, k=side=='L' ? (n, 3) : (3, n)
                    c=randn(rng, T, m, k)
                    before=copy(c)
                    op=trans=='N' ? q : q'
                    result=direct_tridiagonal_apply!(
                        a,
                        factor.tau,
                        c,
                        library==reference ? Int64 : J;
                        side,
                        uplo,
                        trans,
                        m,
                        n=k,
                        library,
                    )
                    @test result.info==0
                    @test result.diagnostic==0
                    @test c≈(side=='L' ? op*before : before*op) atol=tol rtol=tol
                end
            end
        end
    end
end

@testset "Tridiagonal workspace queries and argument diagnostics" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
        a=fill(T(NaN), 5, 3)
        tau=zeros(T, 2)
        c=fill(T(NaN), 3, 2)
        for (kw, arg) in
            (((; uplo=('?')), 1), ((; n=-1), 2), ((; lda=2), 4), ((; lwork=0), 9))
            result=direct_tridiagonal!(a, J, Val(true); kw...)
            @test result.info == -arg
            @test result.diagnostic == arg
            @test result.guard==fill(J(-777), 2)
        end
        result=direct_tridiagonal!(a, J, Val(true); lwork=-1)
        @test result.info==result.diagnostic==0
        @test real(result.work[1])>=1
        @test all(isnan, a)
        for (kw, arg) in
            (((; uplo=('?')), 1), ((; n=-1), 2), ((; lda=2), 4), ((; lwork=1), 7))
            result=direct_tridiagonal_generate!(a, tau, J; kw...)
            @test result.info == -arg
            @test result.diagnostic == arg
            @test result.guard==fill(J(-777), 2)
        end
        result=direct_tridiagonal_generate!(a, tau, J; lwork=-1)
        @test result.info==result.diagnostic==0
        @test real(result.work[1])>=2
        @test all(isnan, a)
        for (kw, arg) in (
            ((; side=('?')), 1),
            ((; uplo=('?')), 2),
            ((; trans=('?')), 3),
            ((; m=-1), 4),
            ((; n=-1), 5),
            ((; lda=2), 7),
            ((; ldc=2), 10),
            ((; lwork=1), 12),
        )
            result=direct_tridiagonal_apply!(a, tau, c, J; kw...)
            @test result.info == -arg
            @test result.diagnostic == arg
            @test result.guard==fill(J(-777), 2)
        end
        result=direct_tridiagonal_apply!(a, tau, c, J; lwork=-1)
        @test result.info==result.diagnostic==0
        @test all(isnan, c)
        result=direct_tridiagonal_apply!(a, tau, c, J)
        @test result.info==result.diagnostic==0
        @test all(isnan, c)
    end
end

for (T, prefix) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    J in (Int32, Int64)

    R=real(T)
    symbol=Symbol(prefix*"latrd_"*(J===Int64 ? "64_" : ""))
    @eval function direct_tridiagonal_panel!(
        a::Matrix{$T},
        ::Type{$J};
        uplo=('L'),
        n=size(a, 2),
        nb,
        lda=size(a, 1),
        ldw=max(1, n)+2,
        library=Colmat.backend_library(),
    )
        e=fill($R(777), max(1, n))
        tau=fill($T(777), max(1, n))
        w=fill($T(777), ldw, max(1, nb))
        f=Libdl.dlsym(Libdl.dlopen(library), $(QuoteNode(symbol)))
        ccall(
            f,
            Cvoid,
            (
                Ref{UInt8},
                Ref{$J},
                Ref{$J},
                Ptr{$T},
                Ref{$J},
                Ptr{$R},
                Ptr{$T},
                Ptr{$T},
                Ref{$J},
            ),
            uplo,
            n,
            nb,
            a,
            lda,
            e,
            tau,
            w,
            ldw,
        )
        (; e, tau, w, diagnostic=tridiagonal_diagnostic())
    end
end
@testset "LATRD partial panels and deferred updates" begin
    reference=tridiagonal_reference()
    rng=MersenneTwister(1296)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64), uplo in ('L', 'U'), n in (0, 1, 2, 6),
        nb in unique((0, min(1, n), min(3, n), n))

        R=real(T)
        tol=500eps(R)
        original=Matrix(Hermitian(randn(rng, T, n, n)))
        a=fill(T(777), n+2, n)
        for j in 1:n, i in 1:n
            a[i, j]=(uplo=='L' ? i>=j : i<=j) ? original[i, j] : T(NaN)
            if T<:Complex && i==j
                a[i, j]=complex(real(original[i, j]), R(NaN))
            end
        end
        initial=copy(a)
        external_a=copy(a)
        own=direct_tridiagonal_panel!(a, J; uplo, n, nb)
        external=direct_tridiagonal_panel!(
            external_a, Int64; uplo, n, nb, library=reference
        )
        @test own.diagnostic==0
        @test a[(n + 1):end, :]==initial[(n + 1):end, :]
        @test all(==(T(777)), own.w[(n + 1):end, :])
        for stage in 0:(min(nb, n - 1) - 1)
            col=uplo=='L' ? stage+1 : n-stage
            index=uplo=='L' ? stage+1 : n-stage-1
            wc=uplo=='L' ? stage+1 : nb-stage
            rows=uplo=='L' ? ((col + 1):n) : (1:(col - 1))
            @test own.e[index]≈external.e[index] atol=tol rtol=tol
            @test own.tau[index]≈external.tau[index] atol=tol rtol=tol
            @test a[rows, col]≈external_a[rows, col] atol=tol rtol=tol
            @test own.w[rows, wc]≈external.w[rows, wc] atol=tol rtol=tol
        end
        if nb==0
            @test isequal(a, initial)
            @test all(==(T(777)), own.w)
            continue
        end
        remaining=uplo=='L' ? ((nb + 1):n) : (1:(n - nb))
        for j in remaining, i in remaining
            if uplo=='L' ? i>=j : i<=j
                @test isequal(a[i, j], initial[i, j])
            end
        end
        if 0<nb<n
            # The unreduced principal submatrix remains in A; V and W
            # describe exactly the deferred Hermitian rank-2k update.
            V=uplo=='L' ? a[remaining, 1:nb] : a[remaining, (n - nb + 1):n]
            W=own.w[remaining, 1:nb]
            reduced=original[remaining, remaining]-V*W'-W*V'
            extV=if uplo=='L'
                external_a[remaining, 1:nb]
            else
                external_a[remaining, (n - nb + 1):n]
            end
            extW=external.w[remaining, 1:nb]
            expected=original[remaining, remaining]-extV*extW'-extW*extV'
            @test reduced≈expected atol=tol rtol=tol
        end
    end
end

@testset "LBT tridiagonal entry points and unused leading dimensions" begin
    rng=MersenneTwister(3296)
    for (T, prefix) in
        ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
        J in (Int32, Int64)

        families=if T<:Complex
            ("hetrd", "hetd2", "latrd", "ungtr", "unmtr")
        else
            ("sytrd", "sytd2", "latrd", "orgtr", "ormtr")
        end
        for family in families
            provider=BLAS.lbt_find_backing_library(
                prefix*family*"_", J===Int32 ? :lp64 : :ilp64
            )
            @test provider!==nothing
            @test realpath(provider.libname)==realpath(Colmat.backend_library())
        end
        for uplo in ('L', 'U')
            original=Matrix(Hermitian(randn(rng, T, 5, 5)))
            a=copy(original)
            factor=direct_tridiagonal!(
                a, J, Val(true); uplo, library=BLAS.libblastrampoline
            )
            @test factor.info==factor.diagnostic==0
            q=copy(a)
            r=direct_tridiagonal_generate!(
                q, factor.tau, J; uplo, library=BLAS.libblastrampoline
            )
            @test r.info==r.diagnostic==0
            @test q*SymTridiagonal(factor.d[1:5], factor.e[1:4])*q'≈original rtol=500eps(
                real(T)
            ) atol=500eps(real(T))
            c=Matrix{T}(I, 5, 5)
            r=direct_tridiagonal_apply!(
                a, factor.tau, c, J; uplo, library=BLAS.libblastrampoline
            )
            @test r.info==r.diagnostic==0
            @test c≈q rtol=500eps(real(T)) atol=500eps(real(T))
            for large in (Int(typemax(J)),)
                single=fill(T(2), 1, 1)
                f=direct_tridiagonal!(single, J, Val(false); uplo, lda=large)
                @test f.info==f.diagnostic==0
                @test f.d[1]==2
                r=direct_tridiagonal_generate!(single, T[], J; uplo, lda=large)
                @test r.info==r.diagnostic==0
                @test single[1]==1
            end
        end
    end
end
