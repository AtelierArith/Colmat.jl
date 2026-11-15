using Test, LinearAlgebra, Random, Libdl, Colmat
function pentagonal_diagnostic()
    return ccall(
        Libdl.dlsym(Libdl.dlopen(Colmat.backend_library()), :colmat_blas_last_error),
        Cint,
        (),
    )
end
function pentagonal_reference()
    return only(
        filter(
            l->l.interface==:ilp64&&occursin("openblas", lowercase(basename(l.libname))),
            BLAS.get_config().loaded_libs,
        ),
    ).libname
end
for (T, prefix) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    J in (Int32, Int64)

    suffix=J===Int64 ? "64_" : ""
    for lq in (false, true), blocked in (false, true)
        symbol=Symbol(prefix*(lq ? "tplqt" : "tpqrt")*(blocked ? "" : "2")*"_"*suffix)
        sig=Any[Ref{J}, Ref{J}, Ref{J}]
        args=Any[:m, :n, :l]
        blocked&&(push!(sig, Ref{J}); push!(args, :nb))
        append!(sig, [Ptr{T}, Ref{J}, Ptr{T}, Ref{J}, Ptr{T}, Ref{J}])
        append!(args, [:a, :lda, :b, :ldb, :t, :ldt])
        blocked&&(push!(sig, Ptr{T}); push!(args, :wp))
        push!(sig, Ptr{J})
        push!(args, :ip)
        @eval function direct_tp_factor!(
            a::Matrix{$T},
            b::Matrix{$T},
            ::Type{$J},
            ::Val{$lq},
            ::Val{$blocked};
            m,
            n,
            l,
            nb=1,
            lda=size(a, 1),
            ldb=size(b, 1),
            ldt=($blocked ? nb : max(1, $lq ? m : n))+2,
            library=Colmat.backend_library(),
        )
            k=$lq ? m : n
            t=fill($T(777), ldt, max(1, k))
            work=fill($T(777), max(1, nb*k)+2)
            wp=view(work, 2:(length(work) - 1))
            info=fill($J(-777), 3)
            ip=view(info, 2:2)
            f=Libdl.dlsym(Libdl.dlopen(library), $(QuoteNode(symbol)))
            ccall(f, Cvoid, $(Expr(:tuple, sig...)), $(args...))
            (;
                t,
                work,
                info=info[2],
                guard=info[[1, 3]],
                diagnostic=pentagonal_diagnostic(),
            )
        end
    end
    for lq in (false, true)
        symbol=Symbol(prefix*(lq ? "tpmlqt_" : "tpmqrt_")*suffix)
        @eval function direct_tp_apply!(
            v::Matrix{$T},
            t::Matrix{$T},
            a::Matrix{$T},
            b::Matrix{$T},
            ::Type{$J},
            ::Val{$lq};
            side=('L'),
            trans=('N'),
            m,
            n,
            k,
            l,
            nb,
            ldv=size(v, 1),
            ldt=size(t, 1),
            lda=size(a, 1),
            ldb=size(b, 1),
            library=Colmat.backend_library(),
        )
            work=fill($T(777), max(1, nb*max(m, n))+2)
            wp=view(work, 2:(length(work) - 1))
            info=fill($J(-777), 3)
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
                    Ref{$J},
                    Ptr{$T},
                    Ref{$J},
                    Ptr{$T},
                    Ref{$J},
                    Ptr{$T},
                    Ref{$J},
                    Ptr{$T},
                    Ref{$J},
                    Ptr{$T},
                    Ptr{$J},
                ),
                side,
                trans,
                m,
                n,
                k,
                l,
                nb,
                v,
                ldv,
                t,
                ldt,
                a,
                lda,
                b,
                ldb,
                wp,
                ip,
            )
            (; work, info=info[2], guard=info[[1, 3]], diagnostic=pentagonal_diagnostic())
        end
    end
    symbol=Symbol(prefix*"tprfb_"*suffix)
    @eval function direct_tprfb!(
        v::Matrix{$T},
        t::Matrix{$T},
        a::Matrix{$T},
        b::Matrix{$T},
        ::Type{$J};
        side=('L'),
        trans=('N'),
        direct=('F'),
        store=('C'),
        m,
        n,
        k,
        l,
        ldv=size(v, 1),
        ldt=size(t, 1),
        lda=size(a, 1),
        ldb=size(b, 1),
        ldwork=max(1, side=='L' ? k : m),
        library=Colmat.backend_library(),
    )
        work=fill($T(777), max(1, ldwork)*max(1, side=='L' ? n : k)+2)
        wp=view(work, 2:(length(work) - 1))
        f=Libdl.dlsym(Libdl.dlopen(library), $(QuoteNode(symbol)))
        ccall(
            f,
            Cvoid,
            (
                Ref{UInt8},
                Ref{UInt8},
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
                Ptr{$T},
                Ref{$J},
                Ptr{$T},
                Ref{$J},
            ),
            side,
            trans,
            direct,
            store,
            m,
            n,
            k,
            l,
            v,
            ldv,
            t,
            ldt,
            a,
            lda,
            b,
            ldb,
            wp,
            ldwork,
        )
        (; work, diagnostic=pentagonal_diagnostic())
    end
end
function tp_fixture(T, k, tail, l, lq, rng; zero=false)
    m, n=lq ? (k, tail) : (tail, k)
    a=fill(T(777), k+2, k)
    a[1:k, :].=T(NaN)
    b=fill(T(777), m+2, n)
    b[1:m, :].=T(NaN)
    logical_a=zeros(T, k, k)
    logical_b=zeros(T, m, n)
    for j in 1:k, i in 1:k
        if lq ? i>=j : i<=j
            a[i, j]=logical_a[i, j]=zero ? Base.zero(T) : randn(rng, T)
        end
    end
    for j in 1:n, i in 1:m
        row, col=lq ? (j, i) : (i, j)
        if row<=tail-l||row-(tail-l)<=col
            b[i, j]=logical_b[i, j]=zero ? Base.zero(T) : randn(rng, T)
        end
    end
    return a, b, logical_a, logical_b
end
function tp_full_q(v, t, J, lq, k, tail, l, nb; library=Colmat.backend_library())
    T=eltype(v)
    order=k+tail
    q=Matrix{T}(I, order, order)
    a=fill(T(777), order+2, k)
    b=fill(T(777), order+2, tail)
    a[1:order, :]=q[:, 1:k]
    b[1:order, :]=q[:, (k + 1):end]
    r=direct_tp_apply!(
        v, t, a, b, J, Val(lq); side=('R'), m=order, n=tail, k, l, nb, library
    )
    @test r.info==r.diagnostic==0
    @test r.guard==fill(J(-777), 2)
    @test all(==(T(777)), a[(order + 1):end, :])
    @test all(==(T(777)), b[(order + 1):end, :])
    return hcat(a[1:order, :], b[1:order, :])
end
@testset "Triangular-pentagonal factors and external block interoperability" begin
    reference=pentagonal_reference()
    rng=MersenneTwister(305)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        lq in (false, true),
        k in (0, 1, 3, 5),
        tail in (0, 1, 4),
        l in unique((0, min(k, tail))),
        blocked in (false, true),
        nb in (blocked ? unique((1, max(1, min(2, k)), max(1, k))) : (max(1, k),)),
        zero in (false, true)

        input_a, input_b, logical_a, logical_b=tp_fixture(T, k, tail, l, lq, rng; zero)
        m, n=lq ? (k, tail) : (tail, k)
        order=k+tail
        tol=3000eps(real(T))
        for (library, K) in ((Colmat.backend_library(), J), (reference, Int64))
            a=copy(input_a)
            b=copy(input_b)
            factor=direct_tp_factor!(a, b, K, Val(lq), Val(blocked); m, n, l, nb, library)
            @test factor.info==factor.diagnostic==0
            @test factor.guard==fill(K(-777), 2)
            @test factor.work[[1, end]]==fill(T(777), 2)
            @test all(==(T(777)), a[(k + 1):end, :])
            @test all(==(T(777)), b[(m + 1):end, :])
            for j in 1:k, i in 1:k
                if lq ? i<j : i>j
                    @test isequal(a[i, j], input_a[i, j])
                end
            end
            for j in 1:n, i in 1:m
                row, col=lq ? (j, i) : (i, j)
                if !(row<=tail-l||row-(tail-l)<=col)
                    @test isequal(b[i, j], input_b[i, j])
                end
            end
            triangular=if lq
                Matrix(LowerTriangular(a[1:k, :]))
            else
                Matrix(UpperTriangular(a[1:k, :]))
            end
            for (qlibrary, L) in ((Colmat.backend_library(), J), (reference, Int64))
                q=tp_full_q(b, factor.t, L, lq, k, tail, l, nb; library=qlibrary)
                @test q'*q≈Matrix{T}(I, order, order) atol=tol rtol=tol
                @test (lq ? triangular*q[1:k, :] : q[:, 1:k]*triangular)≈(
                    lq ? hcat(logical_a, logical_b) : vcat(logical_a, logical_b)
                ) atol=tol rtol=tol
                for side in ('L', 'R'), trans in ('N', T<:Complex ? 'C' : 'T')
                    rows, cols=side=='L' ? (order, 3) : (3, order)
                    c=randn(rng, T, rows, cols)
                    op=trans=='N' ? q : q'
                    expected=side=='L' ? op*c : c*op
                    ca=fill(T(777), (side=='L' ? k : rows)+2, side=='L' ? cols : k)
                    cb=fill(T(777), (side=='L' ? tail : rows)+2, side=='L' ? cols : tail)
                    if side=='L'
                        ca[1:k, :]=c[1:k, :]
                        cb[1:tail, :]=c[(k + 1):end, :]
                    else
                        ca[1:rows, :]=c[:, 1:k]
                        cb[1:rows, :]=c[:, (k + 1):end]
                    end
                    # Native multi-RHS left application is checked against
                    # Q formed using the independent right-application path.
                    # Reference row-forward-left TPRFB has a stride defect;
                    # its standalone reproducer is kept under reproducers/.
                    r=direct_tp_apply!(
                        b,
                        factor.t,
                        ca,
                        cb,
                        J,
                        Val(lq);
                        side,
                        trans,
                        m=side=='L' ? tail : rows,
                        n=side=='L' ? cols : tail,
                        k,
                        l,
                        nb,
                    )
                    @test r.info==r.diagnostic==0
                    @test r.guard==fill(J(-777), 2)
                    ar=side=='L' ? k : rows
                    br=side=='L' ? tail : rows
                    actual=if side=='L'
                        vcat(ca[1:ar, :], cb[1:br, :])
                    else
                        hcat(ca[1:ar, :], cb[1:br, :])
                    end
                    @test actual≈expected atol=tol rtol=tol
                    @test all(==(T(777)), ca[(ar + 1):end, :])
                    @test all(==(T(777)), cb[(br + 1):end, :])
                end
            end
        end
    end
end

# Build H from elementary reflectors, independently of the compact-block kernel.
function tp_block_fixture(T, k, tail, l, direct, store, rng)
    forward=direct=='F'
    extra=zeros(T, tail, k)
    for j in 1:k, r in 1:tail
        active=forward ? (r<=tail-l || r-(tail-l)<=j) : (r>l || j<=k-l+r)
        active && (extra[r, j]=randn(rng, T)/3)
    end
    w=forward ? vcat(Matrix{T}(I, k, k), extra) : vcat(extra, Matrix{T}(I, k, k))
    tau=T[(T<:Complex ? 1-cis(0.3+j/7) : 2)/sum(abs2, w[:, j]) for j in 1:k]
    h=Matrix{T}(I, k+tail, k+tail)
    indices=forward ? (1:k) : (k:-1:1)
    compact=zeros(T, k, k)
    for j in indices
        hj=Matrix{T}(I, k+tail, k+tail)-tau[j]*w[:, j]*w[:, j]'
        h=h*hj
        previous=forward ? (1:(j - 1)) : ((j + 1):k)
        compact[previous, j]=compact[previous, previous]*(-tau[j]*(w[:, previous]'*w[:, j]))
        compact[j, j]=tau[j]
    end
    raw=store=='C' ? extra : Matrix(extra')
    v=fill(T(777), size(raw, 1)+2, size(raw, 2))
    v[1:size(raw, 1), :]=raw
    for j in 1:k, r in 1:tail
        active=forward ? (r<=tail-l || r-(tail-l)<=j) : (r>l || j<=k-l+r)
        if !active
            store=='C' ? (v[r, j]=T(NaN)) : (v[j, r]=T(NaN))
        end
    end
    t=fill(T(777), k+2, k)
    for j in 1:k, i in 1:k
        t[i, j]=(forward ? i<=j : i>=j) ? compact[i, j] : T(NaN)
    end
    return v, t, h
end

@testset "TPRFB elementary-reflector oracle and all storage directions" begin
    rng=MersenneTwister(30517)
    reference=pentagonal_reference()
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        k in (1, 3, 5),
        tail in (1, 4),
        l in unique((0, min(k, tail))),
        direct in ('F', 'B'),
        store in ('C', 'R')

        v, t, h=tp_block_fixture(T, k, tail, l, direct, store, rng)
        tol=3000eps(real(T))
        @test h'*h≈Matrix{T}(I, k+tail, k+tail) atol=tol rtol=tol
        original_v=copy(v)
        original_t=copy(t)
        for side in ('L', 'R'),
            trans in ('N', T<:Complex ? 'C' : 'T'),
            (library, K) in ((Colmat.backend_library(), J), (reference, Int64))

            order=k+tail
            rows, cols=side=='L' ? (order, 3) : (3, order)
            c=randn(rng, T, rows, cols)
            op=trans=='N' ? h : h'
            expected=side=='L' ? op*c : c*op
            ar=side=='L' ? k : rows
            br=side=='L' ? tail : rows
            a=fill(T(777), ar+2, side=='L' ? cols : k)
            # Matched legal strides avoid the reference row-forward-left
            # workspace defect demonstrated by the standalone reproducer.
            b=fill(T(777), max(br, k)+2, side=='L' ? cols : tail)
            head=direct=='F' ? (1:k) : ((tail + 1):order)
            rest=direct=='F' ? ((k + 1):order) : (1:tail)
            a[1:ar, :]=side=='L' ? c[head, :] : c[:, head]
            b[1:br, :]=side=='L' ? c[rest, :] : c[:, rest]
            ldwork=if library==reference
                (side=='L' ? size(b, 1) : max(rows, k))
            else
                (side=='L' ? k : rows)
            end
            r=direct_tprfb!(
                v,
                t,
                a,
                b,
                K;
                side,
                trans,
                direct,
                store,
                m=side=='L' ? tail : rows,
                n=side=='L' ? cols : tail,
                k,
                l,
                ldwork,
                library,
            )
            actual=similar(c)
            if side=='L'
                actual[head, :]=a[1:ar, :]
                actual[rest, :]=b[1:br, :]
            else
                actual[:, head]=a[1:ar, :]
                actual[:, rest]=b[1:br, :]
            end
            @test r.diagnostic==0
            @test actual≈expected atol=tol rtol=tol
            @test r.work[[1, end]]==fill(T(777), 2)
            @test all(==(T(777)), a[(ar + 1):end, :])
            @test all(==(T(777)), b[(br + 1):end, :])
            @test isequal(v, original_v)
            @test isequal(t, original_t)
        end
    end
end

@testset "Triangular-pentagonal ABI argument diagnostics" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
        for lq in (false, true), blocked in (false, true)
            defaults=(m=3, n=3, l=1, nb=2, lda=5, ldb=5, ldt=5)
            cases=[:m=>-1, :n=>-1, :l=>-1, :l=>4, :lda=>2, :ldb=>2, :ldt=>0]
            blocked && append!(cases, [:nb=>0, :nb=>4])
            for (name, value) in cases
                a=fill(T(777), 5, 3)
                b=copy(a)
                kw=merge(defaults, NamedTuple{(name,)}((value,)))
                r=direct_tp_factor!(a, b, J, Val(lq), Val(blocked); kw...)
                arg=if name==:m
                    1
                elseif name==:n
                    2
                elseif name==:l
                    3
                elseif name==:nb
                    4
                else
                    (
                        if name==:lda
                            5
                        elseif name==:ldb
                            7
                        else
                            9
                        end
                    )+Int(blocked)
                end
                @test r.info==-arg
                @test r.diagnostic==arg
                @test r.guard==fill(J(-777), 2)
                @test all(==(T(777)), a) && all(==(T(777)), b)
                @test all(==(T(777)), r.t) && all(==(T(777)), r.work)
            end
        end
        for lq in (false, true)
            defaults=(
                side=('L'),
                trans=('N'),
                m=3,
                n=3,
                k=3,
                l=1,
                nb=2,
                ldv=5,
                ldt=5,
                lda=5,
                ldb=5,
            )
            for (name, value, arg) in (
                (:side, '?', 1),
                (:trans, '?', 2),
                (:m, -1, 3),
                (:n, -1, 4),
                (:k, -1, 5),
                (:l, -1, 6),
                (:l, 4, 6),
                (:nb, 0, 7),
                (:nb, 4, 7),
                (:ldv, 2, 9),
                (:ldt, 1, 11),
                (:lda, 2, 13),
                (:ldb, 2, 15),
            )
                a=fill(T(777), 5, 3)
                b=copy(a)
                v=copy(a)
                t=copy(a)
                kw=merge(defaults, NamedTuple{(name,)}((value,)))
                r=direct_tp_apply!(v, t, a, b, J, Val(lq); kw...)
                @test r.info==-arg
                @test r.diagnostic==arg
                @test r.guard==fill(J(-777), 2)
                @test all(==(T(777)), a) && all(==(T(777)), b)
                @test all(==(T(777)), r.work)
            end
        end
        defaults=(
            side=('L'),
            trans=('N'),
            direct=('F'),
            store=('C'),
            m=3,
            n=3,
            k=3,
            l=1,
            ldv=5,
            ldt=5,
            lda=5,
            ldb=5,
            ldwork=3,
        )
        for (name, value, arg) in (
            (:side, '?', 1),
            (:trans, '?', 2),
            (:direct, '?', 3),
            (:store, '?', 4),
            (:m, -1, 5),
            (:n, -1, 6),
            (:k, -1, 7),
            (:l, -1, 8),
            (:l, 4, 8),
            (:ldv, 2, 10),
            (:ldt, 2, 12),
            (:lda, 2, 14),
            (:ldb, 2, 16),
            (:ldwork, 2, 18),
        )
            a=fill(T(777), 5, 3)
            b=copy(a)
            v=copy(a)
            t=copy(a)
            kw=merge(defaults, NamedTuple{(name,)}((value,)))
            r=direct_tprfb!(v, t, a, b, J; kw...)
            @test r.diagnostic==arg
            @test all(==(T(777)), a) && all(==(T(777)), b)
            @test all(==(T(777)), r.work)
        end
    end
end

@testset "All triangular-pentagonal symbols through LBT" begin
    rng=MersenneTwister(30599)
    library=BLAS.libblastrampoline
    for (T, prefix) in
        ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
        J in (Int32, Int64)

        for family in ("tpqrt", "tpqrt2", "tplqt", "tplqt2", "tpmqrt", "tpmlqt", "tprfb")
            provider=BLAS.lbt_find_backing_library(
                prefix*family*"_", J===Int32 ? :lp64 : :ilp64
            )
            @test provider!==nothing
            @test realpath(provider.libname)==realpath(Colmat.backend_library())
        end
        for lq in (false, true), blocked in (false, true)
            k, tail, l, nb=3, 4, 2, blocked ? 2 : 3
            a, b, logical_a, logical_b=tp_fixture(T, k, tail, l, lq, rng)
            m, n=lq ? (k, tail) : (tail, k)
            r=direct_tp_factor!(a, b, J, Val(lq), Val(blocked); m, n, l, nb, library)
            @test r.info==r.diagnostic==0
            q=tp_full_q(b, r.t, J, lq, k, tail, l, nb; library)
            tri=lq ? Matrix(LowerTriangular(a[1:k, :])) : Matrix(UpperTriangular(a[1:k, :]))
            tol=3000eps(real(T))
            @test (lq ? tri*q[1:k, :] : q[:, 1:k]*tri)≈(
                lq ? hcat(logical_a, logical_b) : vcat(logical_a, logical_b)
            ) atol=tol rtol=tol
        end
        v, t, h=tp_block_fixture(T, 3, 4, 2, 'B', 'R', rng)
        a=fill(T(777), 5, 2)
        b=fill(T(777), 6, 2)
        a[1:3, :]=randn(rng, T, 3, 2)
        b[1:4, :]=randn(rng, T, 4, 2)
        expected=h*vcat(b[1:4, :], a[1:3, :])
        r=direct_tprfb!(
            v, t, a, b, J; direct=('B'), store=('R'), m=4, n=2, k=3, l=2, library
        )
        @test r.diagnostic==0
        @test vcat(b[1:4, :], a[1:3, :])≈expected atol=3000eps(real(T)) rtol=3000eps(
            real(T)
        )
        for side in ('L', 'R'), (m, n, k) in ((0, 3, 2), (3, 0, 2), (3, 3, 0))
            v=fill(T(NaN), 5, 5)
            t=copy(v)
            a=fill(T(777), 5, 5)
            b=copy(a)
            r=direct_tprfb!(v, t, a, b, J; side, m, n, k, l=0, ldwork=5, library)
            @test r.diagnostic==0
            @test all(==(T(777)), a) && all(==(T(777)), b)
            @test all(==(T(777)), r.work)
        end
    end
end
