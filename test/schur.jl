using Test, LinearAlgebra, Random, Libdl, Colmat

for (T, p) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    J in (Int32, Int64)

    symbol=Symbol(p*"hseqr_"*(J===Int64 ? "64_" : ""))
    if T<:Real
        @eval function direct_schur!(
            h::Matrix{$T},
            z::Matrix{$T},
            ::Type{$J};
            lo=1,
            hi=size(h, 2),
            comp=('I'),
            query=false,
        )
            n=size(h, 2)
            wr=zeros($T, n)
            wi=zeros($T, n)
            work=zeros($T, max(1, n))
            info=Ref{$J}(0)
            f=Libdl.dlsym(Libdl.dlopen(Colmat.backend_library()), $(QuoteNode(symbol)))
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
                    Ptr{$T},
                    Ptr{$T},
                    Ref{$J},
                    Ptr{$T},
                    Ref{$J},
                    Ref{$J},
                ),
                'S',
                comp,
                n,
                lo,
                hi,
                h,
                max(1, size(h, 1)),
                wr,
                wi,
                z,
                max(1, size(z, 1)),
                work,
                query ? -1 : length(work),
                info,
            )
            return complex.(wr, wi), info[], work
        end
    else
        @eval function direct_schur!(
            h::Matrix{$T},
            z::Matrix{$T},
            ::Type{$J};
            lo=1,
            hi=size(h, 2),
            comp=('I'),
            query=false,
        )
            n=size(h, 2)
            w=zeros($T, n)
            work=zeros($T, max(1, n))
            info=Ref{$J}(0)
            f=Libdl.dlsym(Libdl.dlopen(Colmat.backend_library()), $(QuoteNode(symbol)))
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
                    Ptr{$T},
                    Ref{$J},
                    Ptr{$T},
                    Ref{$J},
                    Ref{$J},
                ),
                'S',
                comp,
                n,
                lo,
                hi,
                h,
                max(1, size(h, 1)),
                w,
                z,
                max(1, size(z, 1)),
                work,
                query ? -1 : length(work),
                info,
            )
            return w, info[], work
        end
    end
end

@testset "HSEQR direct ABI, active ranges and vector accumulation" begin
    Random.seed!(300)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        (n, lo, hi) in ((7, 1, 7), (7, 2, 6), (1, 1, 1), (0, 1, 0))

        h=triu(randn(T, n, n), -1)
        lo>1 && (h[lo, lo - 1]=0)
        hi<n && (h[hi + 1, hi]=0)
        original=copy(h)
        z=Matrix{T}(I, n, n)
        values, info, _=direct_schur!(h, z, J; lo, hi)
        @test info==0
        @test z*h*z' ≈ original
        @test z'*z ≈ Matrix{T}(I, n, n)
        @test all(isfinite, values)
        @test norm(tril(h, T<:Real ? -2 : -1))==0
        q=Matrix(qr(randn(T, n, n)).Q)
        z=copy(q)
        h=copy(original)
        _, info, _=direct_schur!(h, z, J; lo, hi, comp=('V'))
        @test info==0
        @test z*h*z' ≈ q*original*q'
        before=copy(h)
        oldz=copy(z)
        _, info, work=direct_schur!(h, z, J; lo, hi, query=true)
        @test info==0
        @test real(work[1])>=max(1, n)
        @test h==before && z==oldz
    end
end
