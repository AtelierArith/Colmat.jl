using Test, LinearAlgebra, Random, Libdl, Colmat

function structured_lu_last_error()
    return ccall(
        Libdl.dlsym(Libdl.dlopen(Colmat.backend_library()), :colmat_blas_last_error),
        Cint,
        (),
    )
end

function structured_lu_reference()
    return only(
        filter(
            l ->
                l.interface == :ilp64 &&
                occursin("openblas", lowercase(basename(l.libname))),
            BLAS.get_config().loaded_libs,
        ),
    ).libname
end
for (T, prefix) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    J in (Int32, Int64)

    suffix=J===Int64 ? "64_" : ""
    @eval function structured_band_factor!(
        ab::Matrix{$T},
        ::Type{$J};
        m,
        n,
        kl,
        ku,
        family="gbtrf",
        ld=size(ab, 1),
        library=Colmat.backend_library(),
    )
        p=fill($J(777), max(0, min(m, n))+2)
        info=fill($J(777), 3)
        f=Libdl.dlsym(Libdl.dlopen(library), Symbol($prefix*family*"_"*$suffix))
        ccall(
            f,
            Cvoid,
            (Ref{$J}, Ref{$J}, Ref{$J}, Ref{$J}, Ptr{$T}, Ref{$J}, Ptr{$J}, Ptr{$J}),
            m,
            n,
            kl,
            ku,
            ab,
            ld,
            view(p, 2:(length(p) - 1)),
            view(info, 2:2),
        )
        @test info[[1, 3]]==[777, 777]
        @test p[[1, end]]==[777, 777]
        (; info=info[2], p=p[2:(end - 1)])
    end
    @eval function structured_band_solve!(
        ab::Matrix{$T},
        b::Matrix{$T},
        p::Vector{$J},
        ::Type{$J};
        n,
        kl,
        ku,
        tr=('N'),
        nrhs=size(b, 2),
        ld=size(ab, 1),
        ldb=size(b, 1),
        library=Colmat.backend_library(),
    )
        info=fill($J(777), 3)
        f=Libdl.dlsym(Libdl.dlopen(library), Symbol($prefix*"gbtrs_"*$suffix))
        ccall(
            f,
            Cvoid,
            (
                Ref{UInt8},
                Ref{$J},
                Ref{$J},
                Ref{$J},
                Ref{$J},
                Ptr{$T},
                Ref{$J},
                Ptr{$J},
                Ptr{$T},
                Ref{$J},
                Ptr{$J},
            ),
            tr,
            n,
            kl,
            ku,
            nrhs,
            ab,
            ld,
            p,
            view(b, 2:size(b, 1), :),
            ldb,
            view(info, 2:2),
        )
        @test info[[1, 3]]==[777, 777]
        info[2]
    end
    @eval function structured_band_driver!(
        ab::Matrix{$T},
        b::Matrix{$T},
        ::Type{$J};
        n,
        kl,
        ku,
        nrhs=size(b, 2),
        ld=size(ab, 1),
        ldb=size(b, 1),
        library=Colmat.backend_library(),
    )
        p=fill($J(777), max(0, n)+2)
        info=fill($J(777), 3)
        f=Libdl.dlsym(Libdl.dlopen(library), Symbol($prefix*"gbsv_"*$suffix))
        ccall(
            f,
            Cvoid,
            (
                Ref{$J},
                Ref{$J},
                Ref{$J},
                Ref{$J},
                Ptr{$T},
                Ref{$J},
                Ptr{$J},
                Ptr{$T},
                Ref{$J},
                Ptr{$J},
            ),
            n,
            kl,
            ku,
            nrhs,
            ab,
            ld,
            view(p, 2:(length(p) - 1)),
            view(b, 2:size(b, 1), :),
            ldb,
            view(info, 2:2),
        )
        @test info[[1, 3]]==[777, 777]
        @test p[[1, end]]==[777, 777]
        (; info=info[2], p=p[2:(end - 1)])
    end
    @eval function structured_gt_factor!(
        dl::Vector{$T},
        d::Vector{$T},
        du::Vector{$T},
        du2::Vector{$T},
        ::Type{$J};
        n=length(d)-2,
        library=Colmat.backend_library(),
    )
        p=fill($J(777), max(0, n)+2)
        info=fill($J(777), 3)
        f=Libdl.dlsym(Libdl.dlopen(library), Symbol($prefix*"gttrf_"*$suffix))
        ccall(
            f,
            Cvoid,
            (Ref{$J}, Ptr{$T}, Ptr{$T}, Ptr{$T}, Ptr{$T}, Ptr{$J}, Ptr{$J}),
            n,
            view(dl, 2:(length(dl) - 1)),
            view(d, 2:(length(d) - 1)),
            view(du, 2:(length(du) - 1)),
            view(du2, 2:(length(du2) - 1)),
            view(p, 2:(length(p) - 1)),
            view(info, 2:2),
        )
        @test info[[1, 3]]==[777, 777]
        @test p[[1, end]]==[777, 777]
        (; info=info[2], p=p[2:(end - 1)])
    end
    @eval function structured_gt_solve!(
        dl::Vector{$T},
        d::Vector{$T},
        du::Vector{$T},
        du2::Vector{$T},
        p::Vector{$J},
        b::Matrix{$T},
        ::Type{$J};
        n=length(d)-2,
        tr=('N'),
        family="gttrs",
        itr=if tr=='N'
            0
        elseif tr=='T'
            1
        else
            2
        end,
        nrhs=size(b, 2),
        ldb=size(b, 1),
        library=Colmat.backend_library(),
    )
        f=Libdl.dlsym(Libdl.dlopen(library), Symbol($prefix*family*"_"*$suffix))
        info=fill($J(777), 3)
        if family=="gtts2"
            ccall(
                f,
                Cvoid,
                (
                    Ref{$J},
                    Ref{$J},
                    Ref{$J},
                    Ptr{$T},
                    Ptr{$T},
                    Ptr{$T},
                    Ptr{$T},
                    Ptr{$J},
                    Ptr{$T},
                    Ref{$J},
                ),
                itr,
                n,
                nrhs,
                view(dl, 2:(length(dl) - 1)),
                view(d, 2:(length(d) - 1)),
                view(du, 2:(length(du) - 1)),
                view(du2, 2:(length(du2) - 1)),
                p,
                view(b, 2:size(b, 1), :),
                ldb,
            )
            return 0
        end
        ccall(
            f,
            Cvoid,
            (
                Ref{UInt8},
                Ref{$J},
                Ref{$J},
                Ptr{$T},
                Ptr{$T},
                Ptr{$T},
                Ptr{$T},
                Ptr{$J},
                Ptr{$T},
                Ref{$J},
                Ptr{$J},
            ),
            tr,
            n,
            nrhs,
            view(dl, 2:(length(dl) - 1)),
            view(d, 2:(length(d) - 1)),
            view(du, 2:(length(du) - 1)),
            view(du2, 2:(length(du2) - 1)),
            p,
            view(b, 2:size(b, 1), :),
            ldb,
            view(info, 2:2),
        )
        @test info[[1, 3]]==[777, 777]
        info[2]
    end
    @eval function structured_gt_driver!(
        dl::Vector{$T},
        d::Vector{$T},
        du::Vector{$T},
        b::Matrix{$T},
        ::Type{$J};
        n=length(d)-2,
        nrhs=size(b, 2),
        ldb=size(b, 1),
        library=Colmat.backend_library(),
    )
        f=Libdl.dlsym(Libdl.dlopen(library), Symbol($prefix*"gtsv_"*$suffix))
        info=fill($J(777), 3)
        ccall(
            f,
            Cvoid,
            (Ref{$J}, Ref{$J}, Ptr{$T}, Ptr{$T}, Ptr{$T}, Ptr{$T}, Ref{$J}, Ptr{$J}),
            n,
            nrhs,
            view(dl, 2:(length(dl) - 1)),
            view(d, 2:(length(d) - 1)),
            view(du, 2:(length(du) - 1)),
            view(b, 2:size(b, 1), :),
            ldb,
            view(info, 2:2),
        )
        @test info[[1, 3]]==[777, 777]
        info[2]
    end
end
function structured_band_storage(a, kl, ku)
    m, n=size(a)
    ab=fill(eltype(a)(NaN), 2kl+ku+3, n)
    ab[(end - 1):end, :].=777
    for j in 1:n, i in max(1, j - ku):min(m, j + kl)
        ab[kl + ku + 1 + i - j, j]=a[i, j]
    end
    return ab
end
function structured_band_matrix(T, m, n, kl, ku, mode, rng)
    a=zeros(T, m, n)
    for j in 1:n, i in max(1, j - ku):min(m, j + kl)
        a[i, j]=randn(rng, T)/8
    end
    for i in 1:min(m, n)
        a[i, i]+=T(8)
    end
    if mode==2 && m>=2 && n>=1 && kl>=1
        a[1, 1]=T(4)
        a[min(m, kl+1), 1]=T(20)
    elseif mode==3 && n>0
        a[:, 1].=0
    elseif mode==4 && m>0 && n>0
        a[:, end].=0
    end
    return a
end
function structured_band_reconstruct(ab, p, m, n, kl, ku)
    a=zeros(eltype(ab), m, n)
    kv=kl+ku
    for j in 1:n, i in max(1, j - kv):min(m, j)
        a[i, j]=ab[kv + 1 + i - j, j]
    end
    for j in min(m, n):-1:1
        for i in (j + 1):min(m, j + kl), c in j:n
            a[i, c]+=ab[kv + 1 + i - j, j]*a[j, c]
        end
        a[[j, p[j]], j:n]=a[[p[j], j], j:n]
    end
    return a
end
function structured_rhs(a, tr, nrhs, rng)
    op=if tr=='N'
        a
    elseif tr=='T'
        transpose(a)
    else
        adjoint(a)
    end
    x=randn(rng, eltype(a), size(a, 1), nrhs)
    b=fill(eltype(a)(777), size(a, 1)+3, nrhs)
    b[2:(size(a, 1) + 1), :]=op*x
    return (; b, x)
end
function structured_gt_storage(a)
    T=eltype(a)
    n=size(a, 1)
    guarded(v) = vcat(T(777), v, T(777))
    return (;
        dl=guarded(n>=2 ? diag(a, -1) : T[]),
        d=guarded(diag(a)),
        du=guarded(n>=2 ? diag(a, 1) : T[]),
        du2=fill(T(777), max(0, n-2)+2),
    )
end
function structured_gt_guards(f)
    for v in values(f)
        @test v[[1, end]]==[777, 777]
    end
end

@testset "Band LU rectangular storage pivots fill and reconstruction" begin
    rng=MersenneTwister(282)
    reference=structured_lu_reference()
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        (m, n) in ((0, 0), (0, 4), (4, 0), (1, 1), (2, 5), (5, 2), (5, 5), (8, 8)),
        kl in (0, 1, 3, 9),
        ku in (0, 1, 3, 9),
        mode in 1:4,
        family in ("gbtrf", "gbtf2")

        a=structured_band_matrix(T, m, n, kl, ku, mode, rng)
        ab=structured_band_storage(a, kl, ku)
        external=copy(ab)
        result=structured_band_factor!(ab, J; m, n, kl, ku, family)
        expected=structured_band_factor!(
            external, Int64; m, n, kl, ku, family, library=reference
        )
        R=typeof(real(zero(T)))
        tol=500eps(R)
        @test result.info==expected.info
        @test result.p==expected.p
        @test isapprox(ab, external; rtol=tol, atol=tol, nans=true)
        @test structured_band_reconstruct(ab, result.p, m, n, kl, ku)≈a rtol=tol atol=tol
        @test all(==(777), ab[(end - 1):end, :])
    end
end

@testset "Band solve and driver reuse external factors for N T C" begin
    rng=MersenneTwister(1282)
    reference=structured_lu_reference()
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        n in (0, 1, 2, 5, 8),
        kl in (0, 1, 3, 9),
        ku in (0, 1, 3, 9),
        mode in 1:2,
        nrhs in (0, 1, 3)

        a=structured_band_matrix(T, n, n, kl, ku, mode, rng)
        initial=structured_band_storage(a, kl, ku)
        ab=copy(initial)
        external=copy(initial)
        result=structured_band_factor!(ab, J; m=n, n, kl, ku)
        other=structured_band_factor!(external, Int64; m=n, n, kl, ku, library=reference)
        @test result.info==other.info==0
        R=typeof(real(zero(T)))
        tol=1000eps(R)
        for tr in ('N', 'T', 'C')
            fixture=structured_rhs(a, tr, nrhs, rng)
            b=copy(fixture.b)
            expected=copy(b)
            saved=copy(ab)
            @test structured_band_solve!(ab, b, result.p, J; n, kl, ku, tr)==0
            @test structured_band_solve!(
                external, expected, other.p, Int64; n, kl, ku, tr, library=reference
            )==0
            @test b≈expected rtol=tol atol=tol
            @test b[2:(n + 1), :]≈fixture.x rtol=tol atol=tol
            @test all(==(777), b[[1, n+2, n+3], :])
            @test isequal(ab, saved)
            from_external=copy(fixture.b)
            @test structured_band_solve!(
                external, from_external, J.(other.p), J; n, kl, ku, tr
            )==0
            @test from_external≈b rtol=tol atol=tol
            to_external=copy(fixture.b)
            @test structured_band_solve!(
                ab, to_external, Int64.(result.p), Int64; n, kl, ku, tr, library=reference
            )==0
            @test to_external≈b rtol=tol atol=tol
        end
        fixture=structured_rhs(a, 'N', nrhs, rng)
        b=copy(fixture.b)
        driver_ab=copy(initial)
        driver=structured_band_driver!(driver_ab, b, J; n, kl, ku)
        @test driver.info==0
        @test b[2:(n + 1), :]≈fixture.x rtol=tol atol=tol
        @test all(==(777), b[[1, n+2, n+3], :])
    end
end

@testset "Tridiagonal factors DU2 and external solve interoperability" begin
    rng=MersenneTwister(2282)
    reference=structured_lu_reference()
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64), n in 0:9,
        mode in 1:4

        a=structured_band_matrix(T, n, n, 1, 1, mode, rng)
        f=structured_gt_storage(a)
        external=deepcopy(f)
        result=structured_gt_factor!(f.dl, f.d, f.du, f.du2, J)
        other=structured_gt_factor!(
            external.dl, external.d, external.du, external.du2, Int64; library=reference
        )
        R=typeof(real(zero(T)))
        tol=1000eps(R)
        @test result.info==other.info
        @test result.p==other.p
        for name in keys(f)
            @test getproperty(f, name)≈getproperty(external, name) rtol=tol atol=tol
        end
        structured_gt_guards(f)
        if result.info==0
            for tr in ('N', 'T', 'C'), nrhs in (0, 1, 3), family in ("gttrs", "gtts2")
                fixture=structured_rhs(a, tr, nrhs, rng)
                b=copy(fixture.b)
                saved=deepcopy(f)
                @test structured_gt_solve!(
                    f.dl, f.d, f.du, f.du2, result.p, b, J; tr, family
                )==0
                @test b[2:(n + 1), :]≈fixture.x rtol=tol atol=tol
                @test all(==(777), b[[1, n+2, n+3], :])
                @test isequal(f, saved)
                from_external=copy(fixture.b)
                @test structured_gt_solve!(
                    external.dl,
                    external.d,
                    external.du,
                    external.du2,
                    J.(other.p),
                    from_external,
                    J;
                    tr,
                    family,
                )==0
                @test from_external≈b rtol=tol atol=tol
                to_external=copy(fixture.b)
                @test structured_gt_solve!(
                    f.dl,
                    f.d,
                    f.du,
                    f.du2,
                    Int64.(result.p),
                    to_external,
                    Int64;
                    tr,
                    family,
                    library=reference,
                )==0
                @test to_external≈b rtol=tol atol=tol
            end
        end
    end
end

@testset "GTSV driver output layout and singular progression" begin
    rng=MersenneTwister(3282)
    reference=structured_lu_reference()
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64), n in 0:9, mode in 1:6,
        nrhs in (0, 1, 3)

        a=structured_band_matrix(T, n, n, 1, 1, min(mode, 4), rng)
        if mode==5 && n>=2
            a.=0
            for i in 1:n
                a[i, i]=T(2)
            end
            a[1, 2]=T(1)
            a[2, 1]=T(2)
            a[2, 2]=T(1)
        elseif mode==6
            a.=0
        end
        f=structured_gt_storage(a)
        external=deepcopy(f)
        # Reference GTSV can back-substitute into column one even at NRHS=0.
        # Allocate that column so its out-of-contract access remains bounded;
        # only Colmat is required to preserve this unused B storage.
        b=fill(T(777), n+3, max(1, nrhs))
        b[2:(n + 1), :]=randn(rng, T, n, max(1, nrhs))
        before=copy(b)
        expected=copy(b)
        info=structured_gt_driver!(f.dl, f.d, f.du, b, J; nrhs)
        other=structured_gt_driver!(
            external.dl, external.d, external.du, expected, Int64; nrhs, library=reference
        )
        R=typeof(real(zero(T)))
        tol=2000eps(R)
        @test info==other
        for name in (:dl, :d, :du)
            @test getproperty(f, name)≈getproperty(external, name) rtol=tol atol=tol
        end
        if nrhs==0
            @test isequal(b, before)
        else
            @test b≈expected rtol=tol atol=tol
        end
        @test all(==(777), b[[1, n+2, n+3], :])
        structured_gt_guards(f)
        if info==0 && nrhs>0
            @test a*b[2:(n + 1), :]≈before[2:(n + 1), :] rtol=tol atol=tol
        end
    end
end

@testset "Band singular drivers complete factors and preserve B" begin
    rng=MersenneTwister(4282)
    reference=structured_lu_reference()
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        n in (1, 3, 7),
        kl in (0, 1, 3),
        ku in (0, 1, 3),
        mode in (3, 4),
        nrhs in (0, 2)

        a=structured_band_matrix(T, n, n, kl, ku, mode, rng)
        ab=structured_band_storage(a, kl, ku)
        external=copy(ab)
        b=fill(T(777), n+3, max(1, nrhs))
        before=copy(b)
        otherb=copy(b)
        actual=structured_band_driver!(ab, b, J; n, kl, ku, nrhs)
        expected=structured_band_driver!(
            external, otherb, Int64; n, kl, ku, nrhs, library=reference
        )
        R=typeof(real(zero(T)))
        tol=1000eps(R)
        @test actual.info==expected.info>0
        @test actual.p==expected.p
        @test isapprox(ab, external; rtol=tol, atol=tol, nans=true)
        @test isequal(b, before)
        @test isequal(otherb, before)
    end
end

@testset "All 32 structured LU symbols register and compute through LBT" begin
    rng=MersenneTwister(5282)
    for (T, prefix) in
        ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
        J in (Int32, Int64)

        abi=J===Int64 ? :ilp64 : :lp64
        suffix=J===Int64 ? "64_" : ""
        for family in ("gbtrf", "gbtf2", "gbtrs", "gbsv", "gttrf", "gttrs", "gtts2", "gtsv")
            native=Libdl.dlsym(
                Libdl.dlopen(Colmat.backend_library()), Symbol(prefix*family*"_"*suffix)
            )
            @test BLAS.lbt_get_forward(prefix*family*"_", abi)==native
        end
        n=3
        kl=ku=1
        a=structured_band_matrix(T, n, n, kl, ku, 1, rng)
        a[1, 1]=0
        a[1, 2]=T(2)
        a[2, 1]=T(3)
        a[2, 2]=T(4)
        fixture=structured_rhs(a, 'C', 2, rng)
        R=typeof(real(zero(T)))
        tol=1000eps(R)
        for family in ("gbtrf", "gbtf2")
            ab=structured_band_storage(a, kl, ku)
            result=structured_band_factor!(
                ab, J; m=n, n, kl, ku, family, library=BLAS.libblastrampoline
            )
            @test result.info==0
            b=copy(fixture.b)
            @test structured_band_solve!(
                ab, b, result.p, J; n, kl, ku, tr=('C'), library=BLAS.libblastrampoline
            )==0
            @test b[2:(n + 1), :]≈fixture.x rtol=tol atol=tol
        end
        normal=structured_rhs(a, 'N', 2, rng)
        ab=structured_band_storage(a, kl, ku)
        b=copy(normal.b)
        @test structured_band_driver!(ab, b, J; n, kl, ku, library=BLAS.libblastrampoline).info==0
        @test b[2:(n + 1), :]≈normal.x rtol=tol atol=tol
        f=structured_gt_storage(a)
        result=structured_gt_factor!(
            f.dl, f.d, f.du, f.du2, J; library=BLAS.libblastrampoline
        )
        @test result.info==0
        for family in ("gttrs", "gtts2")
            b=copy(fixture.b)
            @test structured_gt_solve!(
                f.dl,
                f.d,
                f.du,
                f.du2,
                result.p,
                b,
                J;
                tr=('C'),
                family,
                library=BLAS.libblastrampoline,
            )==0
            @test b[2:(n + 1), :]≈fixture.x rtol=tol atol=tol
        end
        f=structured_gt_storage(a)
        b=copy(normal.b)
        @test structured_gt_driver!(f.dl, f.d, f.du, b, J; library=BLAS.libblastrampoline)==0
        @test b[2:(n + 1), :]≈normal.x rtol=tol atol=tol
    end
end

@testset "Structured LU argument order INFO width and empty calls" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
        ab=fill(T(777), 8, 3)
        b=fill(T(777), 6, 1)
        p=J[1, 2, 3]
        for family in ("gbtrf", "gbtf2"),
            (changes, arg) in (
                ((; m=-1, n=-1, kl=-1), 1),
                ((; n=-1), 2),
                ((; kl=-1), 3),
                ((; ku=-1), 4),
                ((; ld=3), 6),
            )

            args=merge((; m=3, n=3, kl=1, ku=1, ld=8), changes)
            actual=structured_band_factor!(ab, J; args..., family)
            @test actual.info==-arg
            @test structured_lu_last_error()==arg
            @test all(==(777), actual.p)
        end
        for (changes, arg) in (
            ((; tr=('X'), n=-1), 1),
            ((; n=-1), 2),
            ((; kl=-1), 3),
            ((; ku=-1), 4),
            ((; nrhs=-1), 5),
            ((; ld=3), 7),
            ((; ldb=2), 10),
        )
            args=merge((; n=3, kl=1, ku=1, tr=('N'), nrhs=1, ld=8, ldb=6), changes)
            @test structured_band_solve!(ab, b, p, J; args...)==-arg
            @test structured_lu_last_error()==arg
        end
        for (changes, arg) in (
            ((; n=-1, kl=-1), 1),
            ((; kl=-1), 2),
            ((; ku=-1), 3),
            ((; nrhs=-1), 4),
            ((; ld=3), 6),
            ((; ldb=2), 9),
        )
            args=merge((; n=3, kl=1, ku=1, nrhs=1, ld=8, ldb=6), changes)
            @test structured_band_driver!(ab, b, J; args...).info==-arg
            @test structured_lu_last_error()==arg
        end
        a=Matrix{T}(I, 3, 3)
        f=structured_gt_storage(a)
        @test structured_gt_factor!(f.dl, f.d, f.du, f.du2, J; n=-1).info == -1
        @test structured_lu_last_error()==1
        for (changes, arg) in
            (((; tr=('X'), n=-1), 1), ((; n=-1), 2), ((; nrhs=-1), 3), ((; ldb=2), 10))
            args=merge((; n=3, nrhs=1, tr=('N'), ldb=6), changes)
            @test structured_gt_solve!(f.dl, f.d, f.du, f.du2, p, b, J; args...)==-arg
            @test structured_lu_last_error()==arg
        end
        for (changes, arg) in (((; n=-1, nrhs=-1), 1), ((; nrhs=-1), 2), ((; ldb=2), 7))
            args=merge((; n=3, nrhs=1, ldb=6), changes)
            @test structured_gt_driver!(f.dl, f.d, f.du, b, J; args...)==-arg
            @test structured_lu_last_error()==arg
        end
        for (changes, arg) in
            (((; itr=3, n=-1), 1), ((; n=-1), 2), ((; nrhs=-1), 3), ((; ldb=2), 10))
            args=merge((; n=3, nrhs=1, itr=0, ldb=6), changes)
            structured_gt_solve!(f.dl, f.d, f.du, f.du2, p, b, J; args..., family="gtts2")
            @test structured_lu_last_error()==arg
        end
        @test all(==(777), ab)
        @test all(==(777), b)
        @test p==[1, 2, 3]
        if J===Int64
            negative=-(Int64(1)<<32)+1
            large=(Int64(1)<<32)+1
            @test structured_band_factor!(ab, J; m=negative, n=0, kl=1, ku=1).info == -1
            @test structured_lu_last_error()==1
            @test structured_gt_factor!(f.dl, f.d, f.du, f.du2, J; n=negative).info == -1
            @test structured_lu_last_error()==1
            @test structured_band_solve!(
                ab, b, p, J; n=large, kl=1, ku=1, nrhs=0, ldb=large
            )==0
            @test structured_gt_solve!(
                f.dl, f.d, f.du, f.du2, p, b, J; n=large, nrhs=0, ldb=large
            )==0
        end
    end
end
