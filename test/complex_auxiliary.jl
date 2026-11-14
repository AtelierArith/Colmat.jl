using Test, LinearAlgebra, Libdl, Colmat, Random

for (T, R, p) in ((ComplexF32, Float32, "c"), (ComplexF64, Float64, "z")),
    J in (Int32, Int64)

    suffix=J===Int64 ? "64_" : ""
    @eval function direct_lacgv!(
        x::Vector{$T}, ::Type{$J}; n, inc, library=Colmat.backend_library()
    )
        f=Libdl.dlsym(Libdl.dlopen(library), Symbol($p*"lacgv_"*$suffix))
        ccall(f, Cvoid, (Ref{$J}, Ptr{$T}, Ref{$J}), n, pointer(x, 2), inc)
    end
    @eval function direct_lacp2!(
        a::Matrix{$R}, b::Matrix{$T}, ::Type{$J}; m, n, u, library=Colmat.backend_library()
    )
        f=Libdl.dlsym(Libdl.dlopen(library), Symbol($p*"lacp2_"*$suffix))
        ccall(
            f,
            Cvoid,
            (Ref{UInt8}, Ref{$J}, Ref{$J}, Ptr{$R}, Ref{$J}, Ptr{$T}, Ref{$J}),
            u,
            m,
            n,
            a,
            size(a, 1),
            b,
            size(b, 1),
        )
    end
    @eval function direct_mixed!(
        ::Type{$T}, ::Type{$J}, family, a, b, c; m, n, library=Colmat.backend_library()
    )
        f=Libdl.dlsym(Libdl.dlopen(library), Symbol($p*family*"_"*$suffix))
        work=fill($R(777), 2m*n+2)
        if family=="lacrm"
            ccall(
                f,
                Cvoid,
                (
                    Ref{$J},
                    Ref{$J},
                    Ptr{$T},
                    Ref{$J},
                    Ptr{$R},
                    Ref{$J},
                    Ptr{$T},
                    Ref{$J},
                    Ptr{$R},
                ),
                m,
                n,
                a,
                size(a, 1),
                b,
                size(b, 1),
                c,
                size(c, 1),
                pointer(work, 2),
            )
        else
            ccall(
                f,
                Cvoid,
                (
                    Ref{$J},
                    Ref{$J},
                    Ptr{$R},
                    Ref{$J},
                    Ptr{$T},
                    Ref{$J},
                    Ptr{$T},
                    Ref{$J},
                    Ptr{$R},
                ),
                m,
                n,
                a,
                size(a, 1),
                b,
                size(b, 1),
                c,
                size(c, 1),
                pointer(work, 2),
            )
        end
        @test work[1]==work[end]==777
    end
end
function complex_reference()
    return only(
        filter(
            l->l.interface==:ilp64 && occursin("openblas", lowercase(basename(l.libname))),
            BLAS.get_config().loaded_libs,
        ),
    ).libname
end

@testset "LACGV increments and component bits" begin
    for T in (ComplexF32, ComplexF64), J in (Int32, Int64), n in 0:9, inc in -3:3
        R=typeof(real(zero(T)))
        count=max(1, (n-1)*abs(inc)+1)
        values=R[0, -0.0, Inf, -Inf, NaN, 1, -1]
        x=T[T(values[mod1(i, 7)], values[mod1(i+2, 7)]) for i in 1:(count + 2)]
        expected=copy(x)
        for i in 0:(n - 1)
            idx=2+i*abs(inc)
            expected[idx]=conj(expected[idx])
        end
        before=copy(x)
        direct_lacgv!(x, J; n, inc)
        @test isequal(x, expected)
        W=R===Float32 ? UInt32 : UInt64
        @test reinterpret(W, x)==reinterpret(W, expected)
        if J===Int64
            direct_lacgv!(before, J; n, inc, library=complex_reference())
            @test isequal(x, before)
        end
    end
end

@testset "LACP2 selected storage and zero imaginary components" begin
    for T in (ComplexF32, ComplexF64),
        J in (Int32, Int64), m in 0:6, n in 0:6,
        u in ('U', 'L', 'A', '?')

        R=typeof(real(zero(T)))
        values=R[0, -0.0, Inf, -Inf, NaN, 1, -1]
        a=R[values[mod1(i+3j, 7)] for i in 1:(m + 2), j in 1:(n + 1)]
        b=fill(T(777, 888), m+3, n+1)
        expected=copy(b)
        for j in 1:n, i in 1:m
            selected=if u=='U'
                i<=j
            elseif u=='L'
                i>=j
            else
                true
            end
            if selected
                expected[i, j]=T(a[i, j], 0)
            end
        end
        before=copy(a)
        direct_lacp2!(a, b, J; m, n, u)
        @test isequal(b, expected)
        @test isequal(a, before)
        if J===Int64
            external=fill(T(777, 888), size(b))
            direct_lacp2!(a, external, J; m, n, u, library=complex_reference())
            @test isequal(b, external)
        end
    end
end

@testset "LACRM/LARCM native real component products" begin
    rng=MersenneTwister(280)
    for T in (ComplexF32, ComplexF64),
        J in (Int32, Int64),
        (m, n) in vcat([(m, n) for m in 0:7 for n in 0:7], [(33, 17), (17, 33), (64, 64)]),
        family in ("lacrm", "larcm")

        R=typeof(real(zero(T)))
        a=family=="lacrm" ? fill(T(NaN), m+2, n+1) : fill(R(NaN), m+2, m+1)
        b=family=="lacrm" ? fill(R(NaN), n+3, n+1) : fill(T(NaN), m+3, n+1)
        if family=="lacrm"
            a[1:m, 1:n]=randn(rng, T, m, n)
            b[1:n, 1:n]=randn(rng, R, n, n)
        else
            a[1:m, 1:m]=randn(rng, R, m, m)
            b[1:m, 1:n]=randn(rng, T, m, n)
        end
        c=fill(T(777, 888), m+4, n+1)
        c[1:m, 1:n].=T(NaN)
        beforea=copy(a)
        beforeb=copy(b)
        expected=copy(c)
        expected[1:m, 1:n]=if family=="lacrm"
            a[1:m, 1:n]*b[1:n, 1:n]
        else
            a[1:m, 1:m]*b[1:m, 1:n]
        end
        direct_mixed!(T, J, family, a, b, c; m, n)
        @test isapprox(c[1:m, 1:n], expected[1:m, 1:n]; rtol=50eps(R), atol=50eps(R))
        @test isequal(c[(m + 1):end, :], expected[(m + 1):end, :])
        @test isequal(c[:, (n + 1):end], expected[:, (n + 1):end])
        @test isequal(a, beforea)
        @test isequal(b, beforeb)
        if J===Int64
            external=fill(T(777, 888), size(c))
            external[1:m, 1:n].=T(NaN)
            direct_mixed!(T, J, family, a, b, external; m, n, library=complex_reference())
            @test isapprox(c[1:m, 1:n], external[1:m, 1:n]; rtol=50eps(R), atol=50eps(R))
        end
    end
    for T in (ComplexF32, ComplexF64), J in (Int32, Int64), family in ("lacrm", "larcm")
        R=typeof(real(zero(T)))
        for value in (T(Inf, 1), T(1, Inf), T(NaN, 1), T(1, NaN))
            a=family=="lacrm" ? fill(value, 1, 1) : fill(R(2), 1, 1)
            b=family=="lacrm" ? fill(R(2), 1, 1) : fill(value, 1, 1)
            c=fill(T(777, 888), 1, 1)
            direct_mixed!(T, J, family, a, b, c; m=1, n=1)
            @test isequal(c[1], T(2real(value), 2imag(value)))
        end
    end
end
