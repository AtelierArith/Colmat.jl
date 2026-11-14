using Test, LinearAlgebra, Random, Libdl, Colmat
for (T, p) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    J in (Int32, Int64)

    symbol=Symbol(p*"gghrd_"*(J===Int64 ? "64_" : ""))
    blocked=Symbol(p*"gghd3_"*(J===Int64 ? "64_" : ""))
    @eval function direct_generalized_hessenberg!(
        a::Matrix{$T},
        b::Matrix{$T},
        q::Matrix{$T},
        z::Matrix{$T},
        ::Type{$J};
        lo=1,
        hi=size(a, 2),
        cq=('I'),
        cz=('I'),
        block=false,
        query=false,
        library=Colmat.backend_library(),
    )
        n=size(a, 2)
        info=Ref{$J}(0)
        work=zeros($T, 1)
        f=Libdl.dlsym(
            Libdl.dlopen(library), block ? $(QuoteNode(blocked)) : $(QuoteNode(symbol))
        )
        if block
            ccall(
                f,
                Cvoid,
                (
                    Ref{UInt8},
                    Ref{UInt8},
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
                    Ref{$J},
                ),
                cq,
                cz,
                n,
                lo,
                hi,
                a,
                max(1, size(a, 1)),
                b,
                max(1, size(b, 1)),
                q,
                max(1, size(q, 1)),
                z,
                max(1, size(z, 1)),
                work,
                query ? -1 : 1,
                info,
            )
        else
            ccall(
                f,
                Cvoid,
                (
                    Ref{UInt8},
                    Ref{UInt8},
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
                    Ref{$J},
                ),
                cq,
                cz,
                n,
                lo,
                hi,
                a,
                max(1, size(a, 1)),
                b,
                max(1, size(b, 1)),
                q,
                max(1, size(q, 1)),
                z,
                max(1, size(z, 1)),
                info,
            )
        end
        return info[], work
    end
end
@testset "Generalized Hessenberg active ranges and both ABIs" begin
    Random.seed!(269)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64), block in (false, true),
        (n, lo, hi) in ((0, 1, 0), (1, 1, 1), (2, 1, 2), (7, 1, 7), (7, 2, 6))

        original=randn(T, n, n)
        triangular=triu(randn(T, n, n))
        for j in 1:n, i in (j + 1):n
            if !(lo<=i<=hi && lo<=j<=hi)
                original[i, j]=0
            end
        end
        for (cq, cz) in (('I', 'I'), ('V', 'V'), ('N', 'N'), ('I', 'N'), ('N', 'I'))
            a=fill(T(321), n+2, n)
            b=copy(a)
            a[1:n, :]=original
            b[1:n, :]=triangular
            for j in 1:n, i in (j + 1):n
                b[i, j]=T(NaN)
            end
            q0=Matrix(qr(randn(T, n, n)).Q)
            z0=Matrix(qr(randn(T, n, n)).Q)
            q=fill(T(321), n+2, n)
            z=copy(q)
            q[1:n, :]=cq=='V' ? q0 : fill(T(NaN), n, n)
            z[1:n, :]=cz=='V' ? z0 : fill(T(NaN), n, n)
            info, _=direct_generalized_hessenberg!(a, b, q, z, J; lo, hi, cq, cz, block)
            @test info==0
            @test all(==(T(321)), a[(n + 1):end, :]) && all(==(T(321)), b[(n + 1):end, :])
            @test all(==(T(321)), q[(n + 1):end, :]) && all(==(T(321)), z[(n + 1):end, :])
            @test norm(tril(a[1:n, :], -2))==0
            @test norm(tril(b[1:n, :], -1))==0
            if cq!='N'
                @test q[1:n, :]'*q[1:n, :] ≈ Matrix{T}(I, n, n)
            else
                @test all(isnan, q[1:n, :])
            end
            if cz!='N'
                @test z[1:n, :]'*z[1:n, :] ≈ Matrix{T}(I, n, n)
            else
                @test all(isnan, z[1:n, :])
            end
            if cq!='N' && cz!='N'
                left=cq=='V' ? q0'*q[1:n, :] : q[1:n, :]
                right=cz=='V' ? z0'*z[1:n, :] : z[1:n, :]
                @test left'*original*right ≈ a[1:n, :]
                @test left'*triangular*right ≈ b[1:n, :]
            end
        end
    end
end

@testset "Generalized Hessenberg blocked path, query and registration" begin
    Random.seed!(2691)
    for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
        n=40
        original=randn(T, n, n)
        triangular=triu(randn(T, n, n))
        a=copy(original)
        b=copy(triangular)
        q=zeros(T, n, n)
        z=copy(q)
        info, _=direct_generalized_hessenberg!(a, b, q, z, J; block=true)
        @test info==0
        @test q'*original*z ≈ a
        @test q'*triangular*z ≈ b
        @test q'*q ≈ Matrix{T}(I, n, n)
        @test z'*z ≈ Matrix{T}(I, n, n)
        @test norm(tril(a, -2))==0 && norm(tril(b, -1))==0
        saved=map(copy, (a, b, q, z))
        info, work=direct_generalized_hessenberg!(a, b, q, z, J; block=true, query=true)
        @test info==0 && real(work[1])>=1
        @test all(isequal.(saved, (a, b, q, z)))
        for block in (false, true)
            info, _=direct_generalized_hessenberg!(a, b, q, z, J; block, lo=0)
            @test info == -4
            @test all(isequal.(saved, (a, b, q, z)))
            info, _=direct_generalized_hessenberg!(a, b, q, z, J; block, cq=('?'))
            @test info == -1
            @test all(isequal.(saved, (a, b, q, z)))
        end
        prefix=if T===Float32
            "s"
        elseif T===Float64
            "d"
        elseif T===ComplexF32
            "c"
        else
            "z"
        end
        for family in ("gghrd", "gghd3")
            backing=BLAS.lbt_find_backing_library(
                prefix*family*"_", J===Int32 ? :lp64 : :ilp64
            )
            @test backing!==nothing &&
                realpath(backing.libname)==realpath(Colmat.backend_library())
        end
    end
end
