using Test, Colmat, LinearAlgebra, Libdl, Random
isdefined(@__MODULE__, :lapacke288_selected_entry) || include("lapacke_calls.jl")
isdefined(@__MODULE__, :lapacke288_gpair_store) ||
    include("lapacke_generalized_qr_pairs.jl")
for (T, prefix) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    J in (Int32, Int64)

    @eval function lapacke288_gdriver(
        ::Type{$T},
        ::Type{$J},
        lse,
        high,
        layout,
        f,
        s,
        p,
        a,
        la,
        b,
        lb,
        c,
        d,
        x,
        w,
        lw;
        route=:dll,
    )
        name="LAPACKE_"*$prefix*(lse ? "gglse" : "ggglm")*(high ? "" : "_work")
        symbol=name*($J===Int64 ? "64_" : "")
        native=Libdl.dlopen(Colmat.backend_library())
        @test BLAS.lbt_get_forward(name, $J===Int64 ? :ilp64 : :lp64)==Libdl.dlsym(
            native, symbol
        )
        pointer=Libdl.dlsym(
            route==:dll ? native : Libdl.dlopen(BLAS.libblastrampoline), symbol
        )
        lapacke288_selected_entry(name, $J, route) do
            if high
                ccall(
                    pointer,
                    $J,
                    (Cint, $J, $J, $J, Ptr{$T}, $J, Ptr{$T}, $J, Ptr{$T}, Ptr{$T}, Ptr{$T}),
                    layout,
                    f,
                    s,
                    p,
                    a,
                    la,
                    b,
                    lb,
                    c,
                    d,
                    x,
                )
            else
                ccall(
                    pointer,
                    $J,
                    (
                        Cint,
                        $J,
                        $J,
                        $J,
                        Ptr{$T},
                        $J,
                        Ptr{$T},
                        $J,
                        Ptr{$T},
                        Ptr{$T},
                        Ptr{$T},
                        Ptr{$T},
                        $J,
                    ),
                    layout,
                    f,
                    s,
                    p,
                    a,
                    la,
                    b,
                    lb,
                    c,
                    d,
                    x,
                    w,
                    lw,
                )
            end
        end
    end
    @eval function lapacke288_gdriver_fortran(
        ::Type{$T}, ::Type{$J}, lse, f, s, p, a, b, c, d, x, library
    )
        name=$prefix*(lse ? "gglse" : "ggglm")*($J===Int64 ? "_64_" : "_")
        la=size(a, 1)
        lb=size(b, 1)
        lw=(lse ? s : f)==0 ? 1 : f+s+p
        w=fill($T(779), lw+2)
        info=Ref{$J}(-777)
        ccall(
            Libdl.dlsym(Libdl.dlopen(library), name),
            Cvoid,
            (
                Ref{$J},
                Ref{$J},
                Ref{$J},
                Ptr{$T},
                Ref{$J},
                Ptr{$T},
                Ref{$J},
                Ptr{$T},
                Ptr{$T},
                Ptr{$T},
                Ptr{$T},
                Ref{$J},
                Ref{$J},
            ),
            f,
            s,
            p,
            a,
            la,
            b,
            lb,
            c,
            d,
            x,
            view(w, 2:(length(w) - 1)),
            lw,
            info,
        )
        @test w[[1, end]]==$T[779, 779]
        info[]
    end
end
# Compute conditioning from independent wide LAPACK. The stacked matrix
# alone hides nearly dependent equality constraints when A is well conditioned.
for (T, prefix) in ((Float64, "d"), (ComplexF64, "z"))
    @eval function lapacke288_gdriver_svd(input::Matrix{$T}, reference)
        m, n=size(input)
        k=min(m, n)
        u=Matrix{$T}(I, m, m)
        vt=Matrix{$T}(I, n, n)
        values=zeros(Float64, k)
        k==0 && return (; u, values, vt)
        a=copy(input)
        rw=zeros(Float64, max(1, 5k))
        info=Ref{Int64}(-777)
        pointer=Libdl.dlsym(Libdl.dlopen(reference), $prefix*"gesvd_64_")
        function execute!(w, lw)
            if $T===Float64
                ccall(
                    pointer,
                    Cvoid,
                    (
                        Ref{UInt8},
                        Ref{UInt8},
                        Ref{Int64},
                        Ref{Int64},
                        Ptr{$T},
                        Ref{Int64},
                        Ptr{Float64},
                        Ptr{$T},
                        Ref{Int64},
                        Ptr{$T},
                        Ref{Int64},
                        Ptr{$T},
                        Ref{Int64},
                        Ref{Int64},
                        Csize_t,
                        Csize_t,
                    ),
                    UInt8('A'),
                    UInt8('A'),
                    m,
                    n,
                    a,
                    m,
                    values,
                    u,
                    m,
                    vt,
                    n,
                    w,
                    lw,
                    info,
                    1,
                    1,
                )
            else
                ccall(
                    pointer,
                    Cvoid,
                    (
                        Ref{UInt8},
                        Ref{UInt8},
                        Ref{Int64},
                        Ref{Int64},
                        Ptr{$T},
                        Ref{Int64},
                        Ptr{Float64},
                        Ptr{$T},
                        Ref{Int64},
                        Ptr{$T},
                        Ref{Int64},
                        Ptr{$T},
                        Ref{Int64},
                        Ptr{Float64},
                        Ref{Int64},
                        Csize_t,
                        Csize_t,
                    ),
                    UInt8('A'),
                    UInt8('A'),
                    m,
                    n,
                    a,
                    m,
                    values,
                    u,
                    m,
                    vt,
                    n,
                    w,
                    lw,
                    rw,
                    info,
                    1,
                    1,
                )
            end
            info[]
        end
        query=zeros($T, 1)
        @test execute!(query, -1)==0
        @test isfinite(real(query[1])) && real(query[1])>=1
        w=zeros($T, ceil(Int, real(query[1])))
        @test execute!(w, length(w))==0
        @test all(isfinite, values) && minimum(values)>0
        (; u, values, vt)
    end
end
function lapacke288_gdriver_condition(lse, a, b, scale, reference)
    an=a/scale
    bn=b/scale
    asvd=lapacke288_gdriver_svd(an, reference)
    bsvd=lapacke288_gdriver_svd(bn, reference)
    if lse
        basis=Matrix(bsvd.vt'[:, (size(b, 1) + 1):end])
        reduced=lapacke288_gdriver_svd(an*basis, reference)
        constraint=bsvd.values
        other=asvd.values
    else
        basis=asvd.u[:, (size(a, 2) + 1):end]
        reduced=lapacke288_gdriver_svd(Matrix(basis'*bn), reference)
        constraint=asvd.values
        other=bsvd.values
    end
    constraint_condition=isempty(constraint) ? 1.0 : first(constraint)/last(constraint)
    free_condition=if isempty(reduced.values)
        1.0
    else
        first(reduced.values)/last(reduced.values)
    end
    coupling=if isempty(reduced.values)||isempty(other)
        1.0
    else
        max(1.0, first(other)/last(reduced.values))
    end
    value=max(free_condition, constraint_condition*coupling)
    @test isfinite(value) && value>=1
    return (; value, basis)
end

@testset "Constrained least-squares C solutions and destructive payload" begin
    rng=MersenneTwister(288453)
    native=Colmat.backend_library()
    reference=only(
        filter(
            l->l.interface==:ilp64&&occursin("openblas", lowercase(l.libname)),
            BLAS.get_config().loaded_libs,
        ),
    ).libname
    diagnostic=Libdl.dlsym(Libdl.dlopen(native), :colmat_blas_last_error)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        lse in (false, true)

        shapes=if lse
            (
                (0, 0, 0),
                (0, 3, 3),
                (3, 0, 0),
                (3, 3, 0),
                (3, 3, 3),
                (5, 3, 1),
                (3, 5, 2),
                (2, 5, 4),
                (1, 1, 1),
                (5, 2, 0),
            )
        else
            (
                (0, 0, 0),
                (0, 0, 3),
                (3, 0, 3),
                (3, 3, 0),
                (3, 3, 3),
                (5, 3, 2),
                (3, 1, 5),
                (4, 3, 6),
                (1, 1, 1),
                (2, 0, 4),
            )
        end
        for (f, s, p) in shapes,
            exponent in (-1, 0, 1),
            layout in (101, 102),
            route in (:dll, :lbt_trampoline),
            high in (false, true)

            R=real(T)
            scale=R(2)^(exponent*(R===Float32 ? 40 : 400))
            ar, ac, br, bc=lse ? (f, s, p, s) : (f, s, f, p)
            ao=scale*randn(rng, T, ar, ac)
            bo=scale*randn(rng, T, br, bc)
            co=scale*randn(rng, T, f)
            dn=lse ? p : s
            xn=lse ? s : p
            d0=lse ? scale*randn(rng, T, p) : zeros(T, s)
            # Native Fortran is an overwrite oracle only. Independent wide
            # LAPACK supplies the solution and minimum-norm comparison.
            an=vcat(ao, fill(T(777), 2, ac))
            bn=vcat(bo, fill(T(777), 3, bc))
            cn=copy(co)
            dnative=copy(d0)
            xnative=zeros(T, xn)
            @test lapacke288_gdriver_fortran(
                T, J, lse, f, s, p, an, bn, cn, dnative, xnative, native
            )==0
            @test ccall(diagnostic, Cint, ())==0
            Wide=promote_type(T, Float64)
            aw=vcat(Matrix{Wide}(ao), fill(Wide(777), 2, ac))
            bw=vcat(Matrix{Wide}(bo), fill(Wide(777), 3, bc))
            cw=Vector{Wide}(co)
            dw=Vector{Wide}(d0)
            xw=zeros(Wide, xn)
            @test lapacke288_gdriver_fortran(
                Wide, Int64, lse, f, s, p, aw, bw, cw, dw, xw, reference
            )==0
            la=(layout==101 ? ac : ar)+3
            lb=(layout==101 ? bc : br)+4
            aa=lapacke288_gpair_store(ao, layout, la)
            bb=lapacke288_gpair_store(bo, layout, lb)
            cc=vcat(T(777), co, T(777))
            dd=vcat(T(777), d0, T(777))
            xx=fill(T(777), xn+2)
            old=deepcopy((aa, bb, cc, dd, xx))
            minimum=(lse ? s : f)==0 ? 1 : f+s+p
            w=fill(T(779), minimum+2)
            av=view(aa, 2:(length(aa) - 1))
            bv=view(bb, 2:(length(bb) - 1))
            cv=view(cc, 2:(length(cc) - 1))
            dv=view(dd, 2:(length(dd) - 1))
            xv=view(xx, 2:(length(xx) - 1))
            wv=view(w, 2:(length(w) - 1))
            call(high, lw) = lapacke288_gdriver(
                T,
                J,
                lse,
                high,
                layout,
                f,
                s,
                p,
                av,
                la,
                bv,
                lb,
                cv,
                dv,
                xv,
                wv,
                lw;
                route,
            )
            if !high
                @test call(false, -1)==0
                @test real(w[2])>=minimum && isequal((aa, bb, cc, dd, xx), old)
                query=w[2]
                w[2]=T(779)
                @test call(false, 0)==-13
                @test w[2]==query && isequal((aa, bb, cc, dd, xx), old)
                @test ccall(diagnostic, Cint, ())==12
            end
            @test call(high, minimum)==0
            a=lapacke288_gpair_load(aa, layout, ar, ac, la)
            b=lapacke288_gpair_load(bb, layout, br, bc, lb)
            tol=1000eps(R)*max(1, f, s, p)
            conditioning=lapacke288_gdriver_condition(
                lse, Matrix{Wide}(ao), Matrix{Wide}(bo), Float64(scale), reference
            )
            condition=conditioning.value
            backward=100eps(R)*max(1, f, s, p)
            forward=max(tol, 20eps(R)*condition)
            @test a≈an[1:ar, :] atol=tol*max(norm(ao), scale) rtol=tol
            @test b≈bn[1:br, :] atol=tol*max(norm(bo), scale) rtol=tol
            @test cc[2:(end - 1)]≈cn atol=tol*max(norm(co), norm(cn), scale) rtol=tol
            @test dd[2:(end - 1)]≈dnative atol=tol*max(norm(d0), norm(dnative), scale) rtol=tol
            @test xx[2:(end - 1)]≈xnative atol=tol*max(norm(xnative), one(R)) rtol=tol
            @test xx[2:(end - 1)]≈xw atol=forward*max(norm(xw), 1) rtol=forward
            if lse
                solution=xx[2:(end - 1)]
                @test norm(bo*solution-d0)<=backward*max(
                    norm(bo)*norm(solution), norm(d0), scale
                )
                basis=conditioning.basis
                @test norm(
                    basis'*(
                        Matrix{Wide}(ao)'*(
                            Matrix{Wide}(ao)*Vector{Wide}(solution)-Vector{Wide}(co)
                        )
                    ),
                )<=backward*max(norm(ao)^2*norm(solution), norm(ao)*norm(co), scale^2)
                @test norm(cc[(s - p + 2):(f + 1)])≈norm(ao*solution-co) atol=tol*max(
                    norm(co), scale
                ) rtol=tol
            else
                x=dd[2:(end - 1)]
                y=xx[2:(end - 1)]
                @test x≈dw atol=forward*max(norm(dw), 1) rtol=forward
                @test norm(ao*x+bo*y-co)<=backward*max(
                    norm(ao)*norm(x)+norm(bo)*norm(y), norm(co), scale
                )
                @test norm(y)≈norm(xw) atol=forward*max(norm(xw), 1) rtol=forward
            end
            if T!==Wide
                ars=vcat(copy(ao), fill(T(777), 2, ac))
                brs=vcat(copy(bo), fill(T(777), 3, bc))
                crs=copy(co)
                drs=copy(d0)
                xrs=zeros(T, xn)
                @test lapacke288_gdriver_fortran(
                    T, Int64, lse, f, s, p, ars, brs, crs, drs, xrs, reference
                )==0
                @test xrs≈xw atol=forward*max(norm(xw), 1) rtol=forward
                if lse
                    @test norm(bo*xrs-d0)<=backward*max(norm(bo)*norm(xrs), norm(d0), scale)
                else
                    @test drs≈dw atol=forward*max(norm(dw), 1) rtol=forward
                    @test norm(ao*drs+bo*xrs-co)<=backward*max(
                        norm(ao)*norm(drs)+norm(bo)*norm(xrs), norm(co), scale
                    )
                end
            end
            for (x, oldx, rows, cols, ld) in
                ((aa, old[1], ar, ac, la), (bb, old[2], br, bc, lb))
                touched=Set(
                    1+(layout==101 ? (i-1)*ld+j : (j-1)*ld+i) for j in 1:cols for
                    i in 1:rows
                )
                @test all(isequal(x[i], oldx[i]) for i in eachindex(x) if !(i in touched))
            end
            @test cc[[1, end]]==dd[[1, end]]==xx[[1, end]]==T[777, 777]
            @test w[[1, end]]==T[779, 779]
        end
    end
end
