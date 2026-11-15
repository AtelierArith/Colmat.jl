using Test, LinearAlgebra, Random, Libdl, Colmat
function pt_reference()
    return only(
        filter(
            l->l.interface==:ilp64&&occursin("openblas", lowercase(basename(l.libname))),
            BLAS.get_config().loaded_libs,
        ),
    ).libname
end
function pt_last_error()
    return ccall(
        Libdl.dlsym(Libdl.dlopen(Colmat.backend_library()), :colmat_blas_last_error),
        Cint,
        (),
    )
end
for (T, R, prefix) in (
        (Float32, Float32, "s"),
        (Float64, Float64, "d"),
        (ComplexF32, Float32, "c"),
        (ComplexF64, Float64, "z"),
    ),
    J in (Int32, Int64)

    suffix=J===Int64 ? "64_" : ""
    complex=T<:Complex
    @eval function direct_pt_factor!(
        d::Vector{$R},
        e::Vector{$T},
        ::Type{$J};
        n=length(d)-2,
        library=Colmat.backend_library(),
    )
        info=fill($J(777), 3)
        f=Libdl.dlsym(Libdl.dlopen(library), Symbol($prefix*"pttrf_"*$suffix))
        ccall(
            f,
            Cvoid,
            (Ref{$J}, Ptr{$R}, Ptr{$T}, Ptr{$J}),
            n,
            view(d, 2:(length(d) - 1)),
            view(e, 2:(length(e) - 1)),
            view(info, 2:2),
        )
        @test info[[1, 3]]==[777, 777]
        @test d[[1, end]]==[777, 777]
        @test e[[1, end]]==[777, 777]
        info[2]
    end
    @eval function direct_pt_solve!(
        d::Vector{$R},
        e::Vector{$T},
        b::Matrix{$T},
        ::Type{$J};
        n=length(d)-2,
        u=('L'),
        iu=u=='U' ? 1 : 0,
        family="pttrs",
        nrhs=size(b, 2),
        ldb=size(b, 1),
        library=Colmat.backend_library(),
    )
        info=fill($J(777), 3)
        f=Libdl.dlsym(Libdl.dlopen(library), Symbol($prefix*family*"_"*$suffix))
        if family=="ptsv"
            ccall(
                f,
                Cvoid,
                (Ref{$J}, Ref{$J}, Ptr{$R}, Ptr{$T}, Ptr{$T}, Ref{$J}, Ptr{$J}),
                n,
                nrhs,
                view(d, 2:(length(d) - 1)),
                view(e, 2:(length(e) - 1)),
                view(b, 2:size(b, 1), :),
                ldb,
                view(info, 2:2),
            )
        elseif family=="pttrs"
            if $complex
                ccall(
                    f,
                    Cvoid,
                    (
                        Ref{UInt8},
                        Ref{$J},
                        Ref{$J},
                        Ptr{$R},
                        Ptr{$T},
                        Ptr{$T},
                        Ref{$J},
                        Ptr{$J},
                    ),
                    u,
                    n,
                    nrhs,
                    view(d, 2:(length(d) - 1)),
                    view(e, 2:(length(e) - 1)),
                    view(b, 2:size(b, 1), :),
                    ldb,
                    view(info, 2:2),
                )
            else
                ccall(
                    f,
                    Cvoid,
                    (Ref{$J}, Ref{$J}, Ptr{$R}, Ptr{$T}, Ptr{$T}, Ref{$J}, Ptr{$J}),
                    n,
                    nrhs,
                    view(d, 2:(length(d) - 1)),
                    view(e, 2:(length(e) - 1)),
                    view(b, 2:size(b, 1), :),
                    ldb,
                    view(info, 2:2),
                )
            end
        else
            if $complex
                ccall(
                    f,
                    Cvoid,
                    (Ref{$J}, Ref{$J}, Ref{$J}, Ptr{$R}, Ptr{$T}, Ptr{$T}, Ref{$J}),
                    iu,
                    n,
                    nrhs,
                    view(d, 2:(length(d) - 1)),
                    view(e, 2:(length(e) - 1)),
                    view(b, 2:size(b, 1), :),
                    ldb,
                )
            else
                ccall(
                    f,
                    Cvoid,
                    (Ref{$J}, Ref{$J}, Ptr{$R}, Ptr{$T}, Ptr{$T}, Ref{$J}),
                    n,
                    nrhs,
                    view(d, 2:(length(d) - 1)),
                    view(e, 2:(length(e) - 1)),
                    view(b, 2:size(b, 1), :),
                    ldb,
                )
            end
            info[2]=0
        end
        @test info[[1, 3]]==[777, 777]
        @test d[[1, end]]==[777, 777]
        @test e[[1, end]]==[777, 777]
        info[2]
    end
end
pt_guard(v) = vcat(eltype(v)(777), v, eltype(v)(777))
function pt_original(T, n, rng)
    R=typeof(real(zero(T)))
    d=R[2+i/16 for i in 1:n]
    e=randn(rng, T, max(0, n-1))/8
    diagonal=copy(d)
    off=copy(e)
    for j in 1:(n - 1)
        diagonal[j + 1]+=R(abs2(e[j]))*d[j]
        off[j]*=d[j]
    end
    return (; d=pt_guard(diagonal), e=pt_guard(off))
end
function pt_matrix(d, e, u)
    T=eltype(e)
    n=length(d)-2
    a=zeros(T, n, n)
    for i in 1:n
        a[i, i]=T(d[i + 1])
    end
    for j in 1:(n - 1)
        value=e[j + 1]
        a[j + 1, j]=u=='L' ? value : conj(value)
        a[j, j + 1]=conj(a[j + 1, j])
    end
    return a
end
function pt_reconstruct(d, e, u)
    T=eltype(e)
    n=length(d)-2
    a=zeros(T, n, n)
    for i in 1:n
        a[i, i]=T(d[i + 1])
    end
    for j in 1:(n - 1)
        a[j + 1, j + 1]+=T(abs2(e[j + 1])*d[j + 1])
        value=e[j + 1]*d[j + 1]
        a[j + 1, j]=u=='L' ? value : conj(value)
        a[j, j + 1]=conj(a[j + 1, j])
    end
    return a
end
function pt_product(a, x)
    b=zeros(eltype(x), size(a, 1), size(x, 2))
    for c in axes(x, 2), i in axes(a, 1), j in axes(a, 2)
        b[i, c]+=a[i, j]*x[j, c]
    end
    return b
end
function pt_rhs(a, nrhs, rng)
    n=size(a, 1)
    x=randn(rng, eltype(a), n, max(1, nrhs))
    b=fill(eltype(a)(777), n+3, max(1, nrhs))
    b[2:(n + 1), :]=pt_product(a, x)
    return (; b, x)
end

@testset "Positive tridiagonal real D complex E and independent LDL reconstruction" begin
    reference=pt_reference()
    rng=MersenneTwister(285)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        n in (0, 1, 2, 3, 4, 7, 16, 35, 64),
        u in (T<:Complex ? ('U', 'L') : ('L',)),
        mode in (1, 2)

        original=pt_original(T, n, rng)
        if mode==2
            original.e[2:(end - 1)].=0
        end
        d=copy(original.d)
        e=copy(original.e)
        rd=copy(d)
        re=copy(e)
        @test direct_pt_factor!(d, e, J)==0
        @test direct_pt_factor!(rd, re, Int64; library=reference)==0
        R=typeof(real(zero(T)))
        tol=256eps(R)
        @test d[2:(end - 1)]≈rd[2:(end - 1)] rtol=tol atol=0
        @test e[2:(end - 1)]≈re[2:(end - 1)] rtol=tol atol=0
        @test pt_reconstruct(d, e, u)≈pt_matrix(original.d, original.e, u) rtol=tol atol=tol
    end
end

@testset "PTTRS PTTS2 upper lower and external factor reuse" begin
    reference=pt_reference()
    rng=MersenneTwister(1285)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64), n in (0, 1, 2, 5, 16, 35),
        u in (T<:Complex ? ('U', 'L') : ('L',))

        original=pt_original(T, n, rng)
        a=pt_matrix(original.d, original.e, u)
        d=copy(original.d)
        e=copy(original.e)
        rd=copy(d)
        re=copy(e)
        @test direct_pt_factor!(d, e, J)==0
        @test direct_pt_factor!(rd, re, Int64; library=reference)==0
        R=typeof(real(zero(T)))
        tol=1500eps(R)
        for nrhs in (0, 1, 2, 4), family in ("pttrs", "ptts2")
            fixture=pt_rhs(a, nrhs, rng)
            savedD=copy(d)
            savedE=copy(e)
            # Always allocate a dummy column for zero-RHS auxiliary reference
            # calls: some PTTS2 implementations still enter their first RHS.
            for (factorsD, factorsE, provider, K) in (
                (d, e, Colmat.backend_library(), J),
                (rd, re, Colmat.backend_library(), J),
                (d, e, reference, Int64),
                (rd, re, reference, Int64),
            )
                b=copy(fixture.b)
                @test direct_pt_solve!(
                    factorsD, factorsE, b, K; n, u, nrhs, family, library=provider
                )==0
                if nrhs==0
                    if provider!=reference||family=="pttrs"||!(T<:Complex)
                        @test isequal(b, fixture.b)
                    end
                else
                    @test b[2:(n + 1), :]≈fixture.x rtol=tol atol=tol
                    @test pt_product(a, b[2:(n + 1), :])≈fixture.b[2:(n + 1), :] rtol=tol atol=tol
                end
                @test all(==(777), b[[1, n+2, n+3], :])
            end
            @test isequal(d, savedD)
            @test isequal(e, savedE)
        end
        if u=='L'
            for nrhs in (0, 1, 3)
                fixture=pt_rhs(a, nrhs, rng)
                b=copy(fixture.b)
                dd=copy(original.d)
                ee=copy(original.e)
                @test direct_pt_solve!(dd, ee, b, J; n, nrhs, family="ptsv")==0
                @test dd≈d rtol=tol atol=tol
                @test ee≈e rtol=tol atol=tol
                if nrhs==0
                    @test isequal(b, fixture.b)
                else
                    @test b[2:(n + 1), :]≈fixture.x rtol=tol atol=tol
                end
                @test all(==(777), b[[1, n+2, n+3], :])
            end
        end
    end
end

@testset "PT nonpositive pivot INFO and partial real D complex E" begin
    reference=pt_reference()
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64), n in (1, 2, 5, 8), position in 1:n, bad in (0, -1),
        nrhs in (0, 2)

        R=typeof(real(zero(T)))
        d=pt_guard(ones(R, n))
        e=pt_guard(zeros(T, max(0, n-1)))
        d[position + 1]=R(bad)
        rd=copy(d)
        re=copy(e)
        savedD=copy(d)
        savedE=copy(e)
        @test direct_pt_factor!(d, e, J)==position
        @test direct_pt_factor!(rd, re, Int64; library=reference)==position
        @test isequal(d, rd)
        @test isequal(e, re)
        b=fill(T(777), n+3, max(1, nrhs))
        before=copy(b)
        @test direct_pt_solve!(savedD, savedE, b, J; n, nrhs, family="ptsv")==position
        @test isequal(b, before)
    end
    for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
        R=typeof(real(zero(T)))
        d=pt_guard(R[2, 1, 9])
        e=pt_guard(T[2, 7])
        rd=copy(d)
        re=copy(e)
        @test direct_pt_factor!(d, e, J)==2
        @test direct_pt_factor!(rd, re, Int64; library=reference)==2
        @test d==rd
        @test e==re
        @test e[3]==7
    end
end

@testset "PT normalization at subnormal and large scales uses independent oracles" begin
    rng=MersenneTwister(2285)
    for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
        R=typeof(real(zero(T)))
        tiny=nextfloat(zero(R))
        for value in (tiny, floatmax(R))
            d=pt_guard(R[value])
            e=pt_guard(T[])
            @test direct_pt_factor!(d, e, J)==0
            for family in ("pttrs", "ptts2", "ptsv")
                b=fill(T(777), 4, 1)
                b[2, 1]=T<:Complex ? T(value, -value) : T(value)
                @test direct_pt_solve!(d, e, b, J; n=1, family)==0
                expected=T<:Complex ? T(1, -1) : T(1)
                @test b[2, 1]==expected
                @test b[[1, 3, 4], :]==fill(T(777), 3, 1)
            end
        end
        for case in 1:2
            diagonal=if case==1
                R[tiny, (T <: Complex ? 4 : 2) * tiny]
            else
                R[3tiny, R===Float32 ? R(1e38) : R(1e308)]
            end
            off=if case==1
                (T<:Complex ? T(tiny, tiny) : T(tiny))
            else
                (
                    if T<:Complex
                        T(
                            R===Float32 ? R(1e-7) : R(1e-15),
                            R===Float32 ? R(2e-8) : R(2e-16),
                        )
                    else
                        T(R===Float32 ? R(1e-7) : R(1e-15))
                    end
                )
            end
            d=pt_guard(diagonal)
            e=pt_guard(T[off])
            @test direct_pt_factor!(d, e, J)==0
            expectedE, expectedD=setprecision(BigFloat, 256) do
                pivot=BigFloat(diagonal[1])
                er=BigFloat(real(off))
                ei=BigFloat(imag(off))
                mult=T<:Complex ? T(R(er/pivot), R(ei/pivot)) : T(R(er/pivot))
                next=R(BigFloat(diagonal[2])-(er*er+ei*ei)/pivot)
                mult, next
            end
            @test isapprox(e[2], expectedE; rtol=64eps(R), atol=0)
            @test isapprox(d[3], expectedD; rtol=64eps(R), atol=0)
            @test isfinite(d[3])&&isfinite(e[2])
            # Choose x=(1,0), so forming the RHS does not lose a tiny pivot
            # underneath an unrelated large contribution from column two.
            b=fill(T(777), 5, 1)
            b[2, 1]=T(diagonal[1])
            b[3, 1]=off
            @test direct_pt_solve!(d, e, b, J; n=2)==0
            @test b[2:3, 1]≈T[1, 0] rtol=64eps(R) atol=64eps(R)
        end
    end
end

@testset "All 16 positive tridiagonal symbols compute through LBT" begin
    rng=MersenneTwister(3285)
    for (T, prefix) in
        ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
        J in (Int32, Int64)

        abi=J===Int64 ? :ilp64 : :lp64
        suffix=J===Int64 ? "64_" : ""
        for family in ("pttrf", "pttrs", "ptts2", "ptsv")
            p=Libdl.dlsym(
                Libdl.dlopen(Colmat.backend_library()), Symbol(prefix*family*"_"*suffix)
            )
            @test BLAS.lbt_get_forward(prefix*family*"_", abi)==p
        end
        original=pt_original(T, 5, rng)
        a=pt_matrix(original.d, original.e, 'L')
        fixture=pt_rhs(a, 2, rng)
        d=copy(original.d)
        e=copy(original.e)
        @test direct_pt_factor!(d, e, J; library=BLAS.libblastrampoline)==0
        R=typeof(real(zero(T)))
        tol=1000eps(R)
        for family in ("pttrs", "ptts2")
            b=copy(fixture.b)
            @test direct_pt_solve!(d, e, b, J; family, library=BLAS.libblastrampoline)==0
            @test b[2:6, :]≈fixture.x rtol=tol atol=tol
        end
        d=copy(original.d)
        e=copy(original.e)
        b=copy(fixture.b)
        @test direct_pt_solve!(d, e, b, J; family="ptsv", library=BLAS.libblastrampoline)==0
        @test b[2:6, :]≈fixture.x rtol=tol atol=tol
    end
end

@testset "PT argument order mixed ABI width and empty factors" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
        R=typeof(real(zero(T)))
        d=fill(R(777), 5)
        e=fill(T(777), 4)
        b=fill(T(777), 6, 1)
        @test direct_pt_factor!(d, e, J; n=-1)==-1
        @test pt_last_error()==1
        for family in ("pttrs", "ptsv"),
            (changes, arg) in (
                ((; n=-1), family=="pttrs"&&T<:Complex ? 2 : 1),
                ((; nrhs=-1), family=="pttrs"&&T<:Complex ? 3 : 2),
                ((; ldb=2), family=="pttrs"&&T<:Complex ? 7 : 6),
            )

            args=merge((; n=3, nrhs=1, ldb=6), changes)
            @test direct_pt_solve!(d, e, b, J; args..., family)==-arg
            @test pt_last_error()==arg
        end
        for (changes, arg) in (
            ((; n=-1), T<:Complex ? 2 : 1),
            ((; nrhs=-1), T<:Complex ? 3 : 2),
            ((; ldb=2), T<:Complex ? 7 : 6),
        )
            args=merge((; n=3, nrhs=1, ldb=6), changes)
            direct_pt_solve!(d, e, b, J; args..., family="ptts2")
            @test pt_last_error()==arg
        end
        if T<:Complex
            @test direct_pt_solve!(d, e, b, J; n=-1, u=('X'))==-1
            @test pt_last_error()==1
            direct_pt_solve!(d, e, b, J; n=0, iu=2, family="ptts2")
            @test pt_last_error()==1
        end
        @test all(==(777), d)
        @test all(==(777), e)
        @test all(==(777), b)
        if J===Int64
            negative=-(Int64(1)<<32)+1
            large=(Int64(1)<<32)+1
            @test direct_pt_factor!(d, e, J; n=negative)==-1
            @test pt_last_error()==1
            @test direct_pt_solve!(d, e, b, J; n=large, nrhs=0, ldb=large)==0
            @test direct_pt_solve!(d, e, b, J; n=large, nrhs=0, ldb=large, family="ptts2")==0
            @test all(==(777), b)
        end
    end
end

@testset "PT nonfinite arithmetic follows real component updates" begin
    reference=pt_reference()
    for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64), case in 1:5
        R=typeof(real(zero(T)))
        diagonal=if case==1
            R[NaN, 2, 3]
        elseif case==2
            R[Inf, 2, 3]
        elseif case==4
            R[-Inf, 2, 3]
        else
            R[2, 3, 4]
        end
        off=if case in (1, 4)
            T[0, 0]
        elseif case==2
            T[1, 0]
        elseif case==3
            T[NaN, 0]
        else
            T[Inf, 0]
        end
        d=pt_guard(diagonal)
        e=pt_guard(off)
        rd=copy(d)
        re=copy(e)
        @test direct_pt_factor!(d, e, J)==direct_pt_factor!(
            rd, re, Int64; library=reference
        )
        @test isapprox(d, rd; nans=true)
        @test isapprox(e, re; nans=true)
    end
end

@testset "PT singleton stride and auxiliary integer width" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        R=typeof(real(zero(T)))
        d=pt_guard(R[1])
        e=pt_guard(T[])
        b=fill(T(777), 4, 1)
        b[2, 1]=T(1)
        @test direct_pt_solve!(d, e, b, Int64; n=1, nrhs=1, ldb=typemax(Int64))==0
        @test b==T[777; 1; 777; 777;;]
        large=typemax(Int64)
        @test direct_pt_solve!(d, e, b, Int64; n=large, nrhs=1, ldb=large)==(
            T<:Complex ? -2 : -1
        )
        @test pt_last_error()==(T<:Complex ? 2 : 1)
        if T<:Complex
            direct_pt_solve!(d, e, b, Int64; n=0, iu=(Int64(1)<<32)+1, family="ptts2")
            @test pt_last_error()==1
        end
    end
end
