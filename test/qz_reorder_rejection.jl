using Test, LinearAlgebra, Colmat
if !isdefined(@__MODULE__, :direct_tgexc!)
    include("qz_reorder.jl")
end

@testset "QZ rejected exchanges retain the last accepted state" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        upward in (false, true)

        R=typeof(real(zero(T)))
        tol=R===Float32 ? 1e-4 : 1e-11
        av=upward ? T[0, 2, 3] : T[1, 2, 0]
        bv=upward ? T[0, 1, 1] : T[1, 1, 0]
        a=Matrix(Diagonal(av))
        b=Matrix(Diagonal(bv))
        olda=copy(a)
        oldb=copy(b)
        q=Matrix{T}(I, 3, 3)
        z=copy(q)
        from, to=upward ? (3, 1) : (1, 3)
        result=direct_tgexc!(a, b, q, z, J; from, to)
        @test result.info==1 && result.diagnostic==0
        @test result.first==from
        @test result.last==(upward && T<:Complex ? 1 : 2)
        @test q*a*z'≈olda rtol=tol atol=tol
        @test q*b*z'≈oldb rtol=tol atol=tol
        values=diag(a) ./ diag(b)
        if upward
            @test isnan(values[1]) && values[2]≈3 && values[3]≈2
        else
            @test values[1]≈2 && values[2]≈1 && isnan(values[3])
        end
        # A single rejected adjacent swap commits no transformation.
        a=Matrix(Diagonal(T[1, 0]))
        b=Matrix(Diagonal(T[1, 0]))
        q=Matrix{T}(I, 2, 2)
        z=copy(q)
        olda=copy(a)
        oldb=copy(b)
        result=direct_tgex2!(a, b, q, z, J)
        @test result.info==1 && result.diagnostic==0
        @test a==olda && b==oldb && q==z==Matrix{T}(I, 2, 2)
    end
end

@testset "TGSEN rejected selection preserves eigenvalue output contracts" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
        R=typeof(real(zero(T)))
        tol=R===Float32 ? 1e-4 : 1e-11
        a=Matrix(Diagonal(T[0, 2, 3]))
        b=Matrix(Diagonal(T[0, 1, 1]))
        q=Matrix{T}(I, 3, 3)
        z=copy(q)
        olda=copy(a)
        oldb=copy(b)
        result=direct_tgsen!(a, b, q, z, [false, false, true], J)
        @test result.info==1 && result.diagnostic==0 && result.m==1
        @test q*a*z'≈olda rtol=tol atol=tol
        @test q*b*z'≈oldb rtol=tol atol=tol
        @test a[2, 2]/b[2, 2]≈T(3)
        @test result.alpha[1]==result.beta[1]==0
        expected=T<:Real ? [0, 3, 2] : [0, 2, 3]
        @test result.alpha≈expected rtol=tol atol=tol
        @test result.beta≈[0, 1, 1] rtol=tol atol=tol
        @test result.pl==result.pr==R(777) && result.dif==fill(R(777), 2)
    end
end
