using Test, LinearAlgebra, Random, Libdl, Colmat
isdefined(@__MODULE__, :direct_geevx!) || include("geevx_expert_helpers.jl")
@testset "GEEVX native N and retained expert estimates, all scalar types and integer ABIs" begin
    Random.seed!(316)
    reference=geevx_reference()
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        n in (0, 1, 2, 4, 7),
        bal in ('N', 'P', 'S', 'B'),
        sense in ('N', 'E', 'V', 'B'),
        left in ('N', 'V'),
        right in ('N', 'V')

        sense in ('E', 'B') && (left!='V' || right!='V') && continue
        tol=2000eps(real(T))*max(1, n)
        original=geevx_fixture(T, n)
        native=nothing
        for (library, K) in ((Colmat.backend_library(), J), (reference, Int64))
            a=vcat(original, fill(T(777), 2, n))
            old=copy(a)
            query=direct_geevx!(a, K; n, bal, left, right, sense, lwork=-1, library)
            @test query.info==query.diagnostic==0
            @test query.workspace>=geevx_minimum(T, n, left, right, sense)
            @test isequal(a, old)
            result=direct_geevx!(a, K; n, bal, left, right, sense, library)
            @test result.info==result.diagnostic==0
            @test all(==(T(777)), a[(n + 1):end, :])
            w=T<:Real ? complex.(result.values, result.imaginary) : result.values
            if left=='V'
                x=geevx_unpack(result.left, result.imaginary)
                @test all(isapprox(norm(col), 1; atol=tol, rtol=tol) for col in eachcol(x))
                @test norm(original'*x-x*Diagonal(conj.(w)))<=tol*max(
                    1, norm(original)*norm(x)
                )
            else
                @test all(==(T(777)), result.left)
            end
            if right=='V'
                x=geevx_unpack(result.right, result.imaginary)
                @test all(isapprox(norm(col), 1; atol=tol, rtol=tol) for col in eachcol(x))
                @test norm(original*x-x*Diagonal(w))<=tol*max(1, norm(original)*norm(x))
            else
                @test all(==(T(777)), result.right)
            end
            if sense in ('E', 'B')
                @test all(x -> isfinite(x) && 0<=x<=1+tol, result.rconde)
            end
            if sense in ('V', 'B')
                @test all(x -> isfinite(x) && x>=0, result.rcondv)
            end
            if sense in ('N', 'V')
                @test all(==(real(T)(777)), result.rconde)
            end
            if sense in ('N', 'E')
                @test all(==(real(T)(777)), result.rcondv)
            end
            if isnothing(native)
                native=(result, copy(a), w)
            elseif sense!='N'
                previous=native[1]
                for field in (
                    :values,
                    :imaginary,
                    :left,
                    :right,
                    :scale,
                    :abnrm,
                    :rconde,
                    :rcondv,
                    :ilo,
                    :ihi,
                    :iwork,
                    :rwork,
                )
                    @test isequal(getproperty(previous, field), getproperty(result, field))
                end
                @test isequal(native[2], a)
            else
                @test sort(native[3]; by=z->(real(z), imag(z)))≈sort(
                    w; by=z->(real(z), imag(z))
                ) atol=tol rtol=tol
            end
        end
    end
end
