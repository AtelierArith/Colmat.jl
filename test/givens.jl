using Test, LinearAlgebra, Random, Libdl, Colmat

function givens_reference()
    return only(
        filter(
            l->l.interface==:ilp64 && occursin("openblas", lowercase(basename(l.libname))),
            BLAS.get_config().loaded_libs,
        ),
    ).libname
end
for (T, prefix) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    J in (Int32, Int64)

    R=real(T)
    suffix=J===Int64 ? "64_" : ""
    @eval function direct_givens(
        f::$T, g::$T, ::Type{$J}; family="lartg", library=Colmat.backend_library()
    )
        fn=Libdl.dlsym(Libdl.dlopen(library), Symbol($prefix*family*"_"*$suffix))
        a=Ref{$T}(f)
        b=Ref{$T}(g)
        c=Ref{$R}(-777)
        s=Ref{$T}(-777)
        r=Ref{$T}(-777)
        if family=="rotg"
            ccall(fn, Cvoid, (Ref{$T}, Ref{$T}, Ref{$R}, Ref{$T}), a, b, c, s)
            r[]=a[]
        else
            ccall(fn, Cvoid, (Ref{$T}, Ref{$T}, Ref{$R}, Ref{$T}, Ref{$T}), a, b, c, s, r)
        end
        (; c=c[], s=s[], r=r[], a=a[], b=b[])
    end
end
for (T, prefix) in ((Float32, "s"), (Float64, "d")), J in (Int32, Int64)
    symbol=Symbol(prefix*"lartgs_"*(J===Int64 ? "64_" : ""))
    @eval function direct_shifted_givens(
        x::$T, y::$T, sigma::$T, ::Type{$J}; library=Colmat.backend_library()
    )
        fn=Libdl.dlsym(Libdl.dlopen(library), $(QuoteNode(symbol)))
        c=Ref{$T}(-777)
        s=Ref{$T}(-777)
        ccall(fn, Cvoid, (Ref{$T}, Ref{$T}, Ref{$T}, Ref{$T}, Ref{$T}), x, y, sigma, c, s)
        (; c=c[], s=s[])
    end
end
function givens_samples(T)
    R=real(T)
    samples=T[0, -0.0, 1, -1, 2, -3, floatmin(R), nextfloat(zero(R)), floatmax(R) / 8]
    if T<:Complex
        append!(
            samples,
            T[
                im,
                -im,
                1 + 2im,
                -3 + 4im,
                T(floatmin(R), -floatmin(R)),
                T(nextfloat(zero(R)), nextfloat(zero(R))),
                T(floatmax(R)/8, -floatmax(R)/8),
            ],
        )
    end
    return samples
end
function big_givens(f, g, family)
    T=typeof(f)
    R=real(T)
    a=T<:Complex ? Complex{BigFloat}(f) : BigFloat(f)
    b=T<:Complex ? Complex{BigFloat}(g) : BigFloat(g)
    if b==0
        c=family=="lartgp" && signbit(real(a)) ? BigFloat(-1) : BigFloat(1)
        si=zero(a)
        rr=family=="lartgp" ? abs(a) : a
    elseif a==0
        c=BigFloat(0)
        if T<:Real && family=="rotg"
            si=BigFloat(1)
            rr=b
        else
            si=conj(b)/abs(b)
            rr=abs(b)
        end
    else
        magnitude=hypot(abs(a), abs(b))
        if T<:Complex
            phase=a/abs(a)
            c=abs(a)/magnitude
            si=phase*conj(b)/magnitude
            rr=phase*magnitude
        else
            signsource=if family=="lartgp"
                BigFloat(1)
            elseif family=="rotg" && abs(a)<=abs(b)
                b
            else
                a
            end
            rr=copysign(magnitude, signsource)
            c=a/rr
            si=b/rr
        end
    end
    return (; c=R(c), s=T(si), r=T(rr))
end
@testset "ROTG/LARTG/LARTGP normalization, phase and extreme scales" begin
    reference=givens_reference()
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        family in (T<:Complex ? ("rotg", "lartg") : ("rotg", "lartg", "lartgp")),
        f in givens_samples(T),
        g in givens_samples(T)

        R=real(T)
        tolerance=100eps(R)
        result=direct_givens(f, g, J; family)
        expected=big_givens(f, g, family)
        external=direct_givens(f, g, Int64; family, library=reference)
        @test result.c>=-tolerance || family in ("rotg", "lartgp")
        @test result.c^2+abs2(result.s)≈one(R) atol=tolerance
        scale=max(abs(real(f)), abs(imag(f)), abs(real(g)), abs(imag(g)))
        if scale>0
            ff=f/scale
            gg=g/scale
            @test -conj(result.s)*ff+result.c*gg≈zero(T) atol=tolerance
            @test result.c*ff+result.s*gg≈result.r/scale atol=tolerance+2nextfloat(zero(R))/scale
        end
        @test result.c≈expected.c atol=tolerance
        @test result.s≈expected.s atol=tolerance
        @test isapprox(result.r, expected.r; rtol=tolerance, atol=16nextfloat(zero(R)))
        # Some OpenBLAS complex ROTG extreme paths violate unit normalization.
        # Keep differential checks in the ordinary domain; BigFloat checks all cases.
        ordinary=all(
            v->v==0 || sqrt(floatmin(R))<abs(v)<sqrt(floatmax(R))/4,
            (real(f), imag(f), real(g), imag(g)),
        )
        if family!="rotg" || ordinary
            @test result.c≈external.c atol=tolerance
            @test result.s≈external.s atol=tolerance
            @test isapprox(result.r, external.r; rtol=tolerance, atol=16nextfloat(zero(R)))
        end
        if family!="rotg"
            @test isequal(result.a, f)
            @test isequal(result.b, g)
        elseif T<:Complex
            @test isequal(result.b, g)
        else
            expectedz=if g==0
                zero(R)
            elseif abs(f)>abs(g)
                expected.s
            elseif expected.c==0
                one(R)
            else
                one(R)/expected.c
            end
            @test isequal(result.b, expectedz) ||
                isapprox(result.b, expectedz; rtol=tolerance, atol=16nextfloat(zero(R)))
        end
        if family=="lartgp"
            @test result.r>=0
        end
    end
end
@testset "Real rotations retain finite coefficients when r overflows" begin
    for T in (Float32, Float64),
        J in (Int32, Int64), family in ("rotg", "lartg", "lartgp"),
        (f, g) in ((floatmax(T), floatmax(T)), (-floatmax(T), floatmax(T)))

        result=direct_givens(f, g, J; family)
        @test isfinite(result.c)
        @test isfinite(result.s)
        @test result.c^2+result.s^2≈one(T) atol=10eps(T)
        @test -result.s*(f/floatmax(T))+result.c*(g/floatmax(T))≈zero(T) atol=10eps(T)
        @test isinf(result.r)
    end
end
@testset "LARTGS shifted rotation" begin
    reference=givens_reference()
    for T in (Float32, Float64),
        J in (Int32, Int64), x in T[0, -0.0, 1, -2, 3, floatmin(T)], y in T[0, -1, 4],
        sigma in T[0, 1, 3]

        result=direct_shifted_givens(x, y, sigma, J)
        external=direct_shifted_givens(x, y, sigma, Int64; library=reference)
        @test result.c^2+result.s^2≈one(T) atol=30eps(T)
        @test result.c≈external.c atol=30eps(T)
        @test result.s≈external.s atol=30eps(T)
        if abs(x)>eps(T) && !(abs(x)==sigma && y==0)
            z=x*x-sigma*sigma
            w=x*y
            @test -result.s*z+result.c*w≈zero(T) atol=200eps(T)*max(one(T), abs(z), abs(w))
        end
    end
end

for (T, prefix) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    J in (Int32, Int64)

    R=real(T)
    suffix=J===Int64 ? "64_" : ""
    @eval function direct_generate_batch!(
        x::Vector{$T},
        y::Vector{$T},
        c::Vector{$R},
        ::Type{$J};
        n,
        incx=1,
        incy=1,
        incc=1,
        library=Colmat.backend_library(),
    )
        fn=Libdl.dlsym(Libdl.dlopen(library), Symbol($prefix*"largv_"*$suffix))
        xp=view(x, 2:(length(x) - 1))
        yp=view(y, 2:(length(y) - 1))
        cp=view(c, 2:(length(c) - 1))
        ccall(
            fn,
            Cvoid,
            (Ref{$J}, Ptr{$T}, Ref{$J}, Ptr{$T}, Ref{$J}, Ptr{$R}, Ref{$J}),
            n,
            xp,
            incx,
            yp,
            incy,
            cp,
            incc,
        )
        nothing
    end
    @eval function direct_apply_batch!(
        family,
        x::Vector{$T},
        y::Vector{$T},
        z::Vector{$T},
        c::Vector{$R},
        s::Vector{$T},
        ::Type{$J};
        n,
        incx=1,
        incy=1,
        incc=1,
        library=Colmat.backend_library(),
    )
        fn=Libdl.dlsym(Libdl.dlopen(library), Symbol($prefix*family*"_"*$suffix))
        xp=view(x, 2:(length(x) - 1))
        yp=view(y, 2:(length(y) - 1))
        zp=view(z, 2:(length(z) - 1))
        cp=view(c, 2:(length(c) - 1))
        sp=view(s, 2:(length(s) - 1))
        if family=="lartv"
            ccall(
                fn,
                Cvoid,
                (Ref{$J}, Ptr{$T}, Ref{$J}, Ptr{$T}, Ref{$J}, Ptr{$R}, Ptr{$T}, Ref{$J}),
                n,
                xp,
                incx,
                yp,
                incy,
                cp,
                sp,
                incc,
            )
        else
            ccall(
                fn,
                Cvoid,
                (Ref{$J}, Ptr{$T}, Ptr{$T}, Ptr{$T}, Ref{$J}, Ptr{$R}, Ptr{$T}, Ref{$J}),
                n,
                xp,
                yp,
                zp,
                incx,
                cp,
                sp,
                incc,
            )
        end
        nothing
    end
    @eval function direct_rotation_sequence!(
        a::Matrix{$T},
        c::Vector{$R},
        s::Vector{$R},
        ::Type{$J};
        m=size(a, 1),
        n=size(a, 2),
        side=('L'),
        pivot=('V'),
        direct=('F'),
        lda=size(a, 1),
        library=Colmat.backend_library(),
    )
        fn=Libdl.dlsym(Libdl.dlopen(library), Symbol($prefix*"lasr_"*$suffix))
        cp=view(c, 2:(length(c) - 1))
        sp=view(s, 2:(length(s) - 1))
        ccall(
            fn,
            Cvoid,
            (
                Ref{UInt8},
                Ref{UInt8},
                Ref{UInt8},
                Ref{$J},
                Ref{$J},
                Ptr{$R},
                Ptr{$R},
                Ptr{$T},
                Ref{$J},
            ),
            side,
            pivot,
            direct,
            m,
            n,
            cp,
            sp,
            a,
            lda,
        )
        a
    end
end
function rotation_storage(values, inc)
    n=length(values)
    T=eltype(values)
    out=fill(T(777), max(1, 1+(n-1)*inc)+2)
    for i in 1:n
        out[2 + (i - 1) * inc]=values[i]
    end
    return out
end
rotation_entries(x, n, inc) = x[2:inc:(1 + n * inc)]
function rotation_guards(x, prior, n, inc)
    used=Set(2:inc:(1 + n * inc))
    return all(i->i in used || isequal(x[i], prior[i]), eachindex(x))
end
function rotation_explicit_product(a, b)
    T=promote_type(eltype(a), eltype(b))
    return [
        sum((a[i, k]*b[k, j] for k in axes(a, 2)); init=zero(T)) for
        i in axes(a, 1), j in axes(b, 2)
    ]
end
@testset "LARGV array coefficients and strided guards" begin
    reference=givens_reference()
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64), n in (0, 1, 17), incx in (1, 3), incy in (1, 2),
        incc in (1, 4)

        R=real(T)
        samples=givens_samples(T)
        f=T[samples[mod1(i, length(samples))] for i in 1:n]
        g=T[samples[mod1(3i+1, length(samples))] for i in 1:n]
        x=rotation_storage(f, incx)
        y=rotation_storage(g, incy)
        c=rotation_storage(fill(R(-777), n), incc)
        xp=copy(x)
        yp=copy(y)
        cp=copy(c)
        xr=copy(x)
        yr=copy(y)
        cr=copy(c)
        direct_generate_batch!(x, y, c, J; n, incx, incy, incc)
        direct_generate_batch!(xr, yr, cr, Int64; n, incx, incy, incc, library=reference)
        @test rotation_guards(x, xp, n, incx)
        @test rotation_guards(y, yp, n, incy)
        @test rotation_guards(c, cp, n, incc)
        for i in 1:n
            expected=big_givens(f[i], g[i], "rotg")
            xx=x[2 + (i - 1) * incx]
            yy=y[2 + (i - 1) * incy]
            cc=c[2 + (i - 1) * incc]
            @test cc≈expected.c atol=100eps(R)
            @test yy≈expected.s atol=100eps(R)
            @test isapprox(xx, expected.r; rtol=100eps(R), atol=16nextfloat(zero(R)))
            @test cc^2+abs2(yy)≈one(R) atol=100eps(R)
            ordinary=all(
                v->v==0 || sqrt(floatmin(R))<abs(v)<sqrt(floatmax(R))/4,
                (real(f[i]), imag(f[i]), real(g[i]), imag(g[i])),
            )
            if ordinary
                @test cc≈cr[2 + (i - 1) * incc] atol=100eps(R)
                @test yy≈yr[2 + (i - 1) * incy] atol=100eps(R)
                @test isapprox(
                    xx, xr[2 + (i - 1) * incx]; rtol=100eps(R), atol=16nextfloat(zero(R))
                )
            end
        end
    end
end
@testset "LARTV and LAR2V scalar formulas and Hermitian diagonals" begin
    reference=givens_reference()
    rng=MersenneTwister(279)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        family in ("lartv", "lar2v"),
        n in (0, 1, 17),
        incx in (1, 3),
        incc in (1, 4)

        R=real(T)
        tolerance=100eps(R)
        incy=family=="lar2v" ? incx : 2
        xv=randn(rng, T, n)
        yv=randn(rng, T, n)
        zv=randn(rng, T, n)
        theta=randn(rng, R, n)
        co=cos.(theta)
        si=T<:Complex ? T.(sin.(theta) .* cis.(randn(rng, R, n))) : sin.(theta)
        if family=="lar2v" && T<:Complex
            xv=T.(real.(xv), R(NaN))
            yv=T.(real.(yv), R(NaN))
        end
        x=rotation_storage(xv, incx)
        y=rotation_storage(yv, incy)
        z=rotation_storage(zv, incx)
        c=rotation_storage(co, incc)
        s=rotation_storage(si, incc)
        xp=copy(x)
        yp=copy(y)
        zp=copy(z)
        cp=copy(c)
        sp=copy(s)
        xr=copy(x)
        yr=copy(y)
        zr=copy(z)
        direct_apply_batch!(family, x, y, z, c, s, J; n, incx, incy, incc)
        direct_apply_batch!(
            family, xr, yr, zr, c, s, Int64; n, incx, incy, incc, library=reference
        )
        @test rotation_guards(x, xp, n, incx)
        @test rotation_guards(y, yp, n, incy)
        @test rotation_guards(z, zp, n, incx)
        @test c==cp
        @test s==sp
        for i in 1:n
            if family=="lartv"
                ex=co[i]*xv[i]+si[i]*yv[i]
                ey=co[i]*yv[i]-conj(si[i])*xv[i]
                @test x[2 + (i - 1) * incx]≈ex atol=tolerance
                @test y[2 + (i - 1) * incy]≈ey atol=tolerance
                @test isequal(z, zp)
            else
                p=T[co[i] conj(si[i]); -si[i] co[i]]
                h=T[real(xv[i]) zv[i]; conj(zv[i]) real(yv[i])]
                expected=rotation_explicit_product(rotation_explicit_product(p, h), p')
                @test x[2 + (i - 1) * incx]≈expected[1, 1] atol=tolerance
                @test y[2 + (i - 1) * incy]≈expected[2, 2] atol=tolerance
                @test z[2 + (i - 1) * incx]≈expected[1, 2] atol=tolerance
                @test imag(x[2 + (i - 1) * incx])==0
                @test imag(y[2 + (i - 1) * incy])==0
            end
            @test x[2 + (i - 1) * incx]≈xr[2 + (i - 1) * incx] atol=tolerance
            @test y[2 + (i - 1) * incy]≈yr[2 + (i - 1) * incy] atol=tolerance
            @test z[2 + (i - 1) * incx]≈zr[2 + (i - 1) * incx] atol=tolerance
        end
    end
end
@testset "LASR left/right, pivot and sequence direction" begin
    reference=givens_reference()
    rng=MersenneTwister(280)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        (m, n) in ((0, 4), (4, 0), (1, 1), (3, 7), (7, 3), (8, 8)),
        side in ('L', 'R'),
        pivot in ('V', 'T', 'B'),
        direct in ('F', 'B')

        R=real(T)
        tolerance=100eps(R)
        original=randn(rng, T, m, n)
        a=fill(T(777), max(1, m+3), n)
        a[1:m, :]=original
        prior=copy(a)
        external=copy(a)
        order=side=='L' ? m : n
        theta=randn(rng, R, max(0, order-1))
        co=cos.(theta)
        si=sin.(theta)
        if order>1
            co[1]=1
            si[1]=0
        end
        c=rotation_storage(co, 1)
        s=rotation_storage(si, 1)
        cp=copy(c)
        sp=copy(s)
        p=Matrix{R}(I, order, order)
        steps=direct=='F' ? (1:(order - 1)) : ((order - 1):-1:1)
        for k in steps
            i, j=if pivot=='V'
                (k, k+1)
            elseif pivot=='T'
                (1, k+1)
            else
                (k, order)
            end
            first=copy(p[i, :])
            second=copy(p[j, :])
            p[i, :]=co[k]*first+si[k]*second
            p[j, :]=co[k]*second-si[k]*first
        end
        expected=if side=='L'
            rotation_explicit_product(p, original)
        else
            rotation_explicit_product(original, p')
        end
        direct_rotation_sequence!(a, c, s, J; m, n, side, pivot, direct)
        direct_rotation_sequence!(
            external, c, s, Int64; m, n, side, pivot, direct, library=reference
        )
        @test a[1:m, :]≈expected atol=tolerance
        @test a[1:m, :]≈external[1:m, :] atol=tolerance
        @test a[(m + 1):end, :]==prior[(m + 1):end, :]
        @test c==cp
        @test s==sp
    end
end

function givens_last_error()
    fn=Libdl.dlsym(Libdl.dlopen(Colmat.backend_library()), :colmat_blas_last_error)
    return ccall(fn, Cint, ())
end
@testset "Rotation argument diagnostics, empty calls and integer width" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
        R=real(T)
        x=rotation_storage(T[3], 1)
        y=rotation_storage(T[4], 1)
        z=rotation_storage(T[2], 1)
        c=rotation_storage(R[1], 1)
        s=rotation_storage(T[0], 1)
        prior=(copy(x), copy(y), copy(z), copy(c), copy(s))
        for badinc in (0, -1)
            for (options, status) in
                (((; incx=badinc), 3), ((; incy=badinc), 5), ((; incc=badinc), 7))
                givens_last_error()
                direct_generate_batch!(x, y, c, J; n=1, options...)
                @test givens_last_error()==status
                @test (x, y, z, c, s)==prior
            end
            for (options, status) in
                (((; incx=badinc), 3), ((; incy=badinc), 5), ((; incc=badinc), 8))
                givens_last_error()
                direct_apply_batch!("lartv", x, y, z, c, s, J; n=1, options...)
                @test givens_last_error()==status
                @test (x, y, z, c, s)==prior
            end
            for (options, status) in (((; incx=badinc), 5), ((; incc=badinc), 8))
                givens_last_error()
                direct_apply_batch!("lar2v", x, y, z, c, s, J; n=1, options...)
                @test givens_last_error()==status
                @test (x, y, z, c, s)==prior
            end
        end
        for n in (0, -1)
            givens_last_error()
            direct_generate_batch!(x, y, c, J; n, incx=0, incy=0, incc=0)
            @test givens_last_error()==0
            @test (x, y, z, c, s)==prior
            for family in ("lartv", "lar2v")
                givens_last_error()
                direct_apply_batch!(family, x, y, z, c, s, J; n, incx=0, incy=0, incc=0)
                @test givens_last_error()==0
                @test (x, y, z, c, s)==prior
            end
        end
        a=fill(T(777), 6, 3)
        ap=copy(a)
        for (options, status) in (
            ((; side=('?')), 1),
            ((; pivot=('?')), 2),
            ((; direct=('?')), 3),
            ((; m=-1), 4),
            ((; n=-1), 5),
            ((; lda=2), 9),
        )
            givens_last_error()
            direct_rotation_sequence!(a, c, real.(s), J; m=3, n=3, options...)
            @test givens_last_error()==status
            @test a==ap
        end
        if J===Int64
            negative=-(Int64(2)^32)+1
            givens_last_error()
            direct_generate_batch!(x, y, c, J; n=negative)
            @test givens_last_error()==0
            @test (x, y, z, c, s)==prior
            givens_last_error()
            direct_rotation_sequence!(a, c, real.(s), J; m=negative, n=1)
            @test givens_last_error()==4
            @test a==ap
        end
        # Identity LASR must not introduce 0*NaN into other components.
        a=fill(T(NaN), 3, 4)
        before=copy(a)
        cc=rotation_storage(ones(R, 2), 1)
        ss=rotation_storage(zeros(R, 2), 1)
        direct_rotation_sequence!(a, cc, ss, J; m=3, n=4)
        @test isequal(a, before)
    end
end
@testset "LARTGS extreme shift scaling" begin
    for T in (Float32, Float64),
        J in (Int32, Int64),
        (x, y, sigma) in (
            (floatmax(T)/8, floatmax(T)/8, floatmax(T)/4),
            (T(1), floatmax(T)/8, floatmax(T)/4),
            (T(0), floatmax(T)/8, floatmax(T)/4),
            (floatmax(T)/8, floatmin(T), T(0)),
            (eps(T), -floatmax(T)/8, T(0)),
            (eps(T), floatmax(T)/7, sqrt(floatmax(T))/2),
        )

        result=direct_shifted_givens(x, y, sigma, J)
        @test isfinite(result.c)
        @test isfinite(result.s)
        @test result.c^2+result.s^2≈one(T) atol=100eps(T)
        a, b, shift=BigFloat(x), BigFloat(y), BigFloat(sigma)
        if abs(a)<BigFloat(eps(T))/2 && shift!=0
            z=-shift*shift
            w=BigFloat(0)
        else
            z=a*a-shift*shift
            w=a*b
        end
        magnitude=hypot(z, w)
        @test isapprox(result.c, T(z/magnitude); rtol=100eps(T), atol=zero(T))
        @test isapprox(result.s, T(w/magnitude); rtol=100eps(T), atol=zero(T))
        scale=max(abs(z), abs(w))
        z/=scale
        w/=scale
        @test abs(-BigFloat(result.s)*z+BigFloat(result.c)*w)<=100eps(T)
        @test BigFloat(result.c)*z+BigFloat(result.s)*w>=0
    end
end
@testset "All plane-rotation symbols through numerical LBT calls" begin
    for (T, prefix) in
        ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
        J in (Int32, Int64)

        R=real(T)
        families=if T<:Complex
            ("rotg", "lartg", "largv", "lartv", "lar2v", "lasr")
        else
            ("rotg", "lartg", "lartgp", "lartgs", "largv", "lartv", "lar2v", "lasr")
        end
        for family in families
            backing=BLAS.lbt_find_backing_library(
                prefix*family*"_", J===Int32 ? :lp64 : :ilp64
            )
            @test backing!==nothing
            @test realpath(backing.libname)==realpath(Colmat.backend_library())
            if family in ("rotg", "lartg", "lartgp")
                result=direct_givens(T(3), T(4), J; family, library=BLAS.libblastrampoline)
                @test result.c≈R(0.6)
                @test result.s≈T(0.8)
                @test result.r≈T(5)
            elseif family=="lartgs"
                result=direct_shifted_givens(
                    T(3), T(4), T(1), J; library=BLAS.libblastrampoline
                )
                @test result.c^2+result.s^2≈one(T)
            elseif family=="largv"
                x=rotation_storage(T[3], 1)
                y=rotation_storage(T[4], 1)
                c=rotation_storage(R[0], 1)
                direct_generate_batch!(x, y, c, J; n=1, library=BLAS.libblastrampoline)
                @test x[2]≈T(5)
                @test y[2]≈T(0.8)
                @test c[2]≈R(0.6)
            elseif family in ("lartv", "lar2v")
                x=rotation_storage(T[3], 1)
                y=rotation_storage(T[4], 1)
                z=rotation_storage(T[1], 1)
                c=rotation_storage(R[0.6], 1)
                s=rotation_storage(T[0.8], 1)
                direct_apply_batch!(
                    family, x, y, z, c, s, J; n=1, library=BLAS.libblastrampoline
                )
                if family=="lartv"
                    @test x[2]≈T(5)
                    @test abs(y[2])<=10eps(R)
                else
                    @test real(x[2]+y[2])≈R(7)
                    @test imag(x[2])==imag(y[2])==0
                end
            else
                a=T[3 1; 4 2]
                c=rotation_storage(R[0.6], 1)
                s=rotation_storage(R[0.8], 1)
                direct_rotation_sequence!(a, c, s, J; library=BLAS.libblastrampoline)
                @test a≈T[5 2.2; 0 0.4] atol=10eps(R)
            end
        end
    end
end
