using Test, Colmat, LinearAlgebra, Random

@testset "Tridiagonal parallel policies" begin
    rng=MersenneTwister(772)
    policies=(
        FaerParThreshold(),
        FaerParThreshold(0),
        FaerParThreshold(4096),
        FaerParThreshold(:never),
    )
    team=FaerTeam(2)
    try
        for T in (Float32, Float64, ComplexF32, ComplexF64), n in (0, 1, 6, 129)
            R=typeof(real(zero(T)))
            tol=4096eps(R)
            X=randn(rng, T, n, n)
            A=X+X'
            snapshot=copy(A)
            for p in policies, resource in (nothing, team)
                req=scratch_requirements(
                    tridiagonalize,
                    A;
                    team = resource,
                    par_threshold = p,
                )
                @test scratch_requirements(
                    tridiagonalize!,
                    A;
                    team = resource,
                    par_threshold = p,
                )==req
                s=FaerScratch(tridiagonalize, A; team = resource, par_threshold = p)
                try
                    F=tridiagonalize(A; scratch = s, team = resource, par_threshold = p)
                    try
                        Q=Matrix(F.Q)
                        @test norm(A-Q*Matrix(F.H)*Q')<=tol*max(norm(A), one(R))
                        @test Q'Q≈Matrix{T}(I, n, n) atol=tol
                    finally
                        close(F)
                    end
                    if req.bytes>0
                        short=FaerScratch(req.bytes-1; alignment = req.alignment)
                        try
                            @test_throws ArgumentError tridiagonalize(
                                A;
                                scratch = short,
                                team = resource,
                                par_threshold = p,
                            )
                            reserve!(
                                short,
                                tridiagonalize,
                                A;
                                team = resource,
                                par_threshold = p,
                            )
                            F=tridiagonalize(
                                A;
                                scratch = short,
                                team = resource,
                                par_threshold = p,
                            )
                            try
                                Q=Matrix(F.Q)
                                @test norm(A-Q*Matrix(F.H)*Q')<=tol*max(norm(A), one(R))
                            finally
                                close(F)
                            end
                        finally
                            close(short)
                        end
                    end
                    @test A==snapshot
                finally
                    close(s)
                end
                @test_throws ArgumentError tridiagonalize(
                    A;
                    scratch = s,
                    team = resource,
                    par_threshold = p,
                )
            end
            @test scratch_requirements(
                tridiagonalize,
                A;
                team,
                par_threshold = FaerParThreshold(:never),
            )==scratch_requirements(
                tridiagonalize,
                A;
                par_threshold = FaerParThreshold(:never),
            )
        end
        for T in (Float32, Float64, ComplexF32, ComplexF64), p in policies
            R=typeof(real(zero(T)))
            tol=4096eps(R)
            X=randn(rng, T, 5, 5)
            original=X+X'
            a=FaerMatrix(original)
            s=FaerScratch(tridiagonalize!, a; team, par_threshold = p)
            try
                F=fetch(@async tridiagonalize!(a; scratch = s, team, par_threshold = p))
                try
                    Q=Matrix(F.Q)
                    @test original≈Q*Matrix(F.H)*Q' rtol=tol atol=tol
                    @test Matrix(a)==Matrix(F.H)
                finally
                    close(F)
                end
                for wrap in (T<:Real ? (Hermitian, Symmetric) : (Hermitian,))
                    copyto!(a, original)
                    h=wrap(a, :L)
                    reserve!(s, hessenberg, h; team, par_threshold = p)
                    @test scratch_requirements(hessenberg!, h; team, par_threshold = p)==scratch_requirements(
                        tridiagonalize,
                        h;
                        team,
                        par_threshold = p,
                    )
                    for f in (hessenberg, hessenberg!)
                        F=f(h; scratch = s, team, par_threshold = p)
                        try
                            Q=Matrix(F.Q)
                            @test original≈Q*Matrix(F.H)*Q' rtol=tol atol=tol
                            @test f===hessenberg ? Matrix(a)==original :
                                  Matrix(a)==Matrix(F.H)
                        finally
                            close(F)
                        end
                    end
                end
            finally
                close(s)
                close(a)
            end
        end
    finally
        close(team)
    end
    p=FaerParThreshold(:never)
    @test_throws ArgumentError scratch_requirements(
        tridiagonalize,
        zeros(0, 0);
        team,
        par_threshold = p,
    )
    @test_throws ArgumentError tridiagonalize(zeros(0, 0); team, par_threshold = p)
end

struct _TriThresholdConversion <: AbstractMatrix{Float64}
    calls::Base.RefValue{Int}
end
Base.size(::_TriThresholdConversion) = (3, 3)
Base.getindex(::_TriThresholdConversion, i::Int, j::Int) = Float64(i+j)
function Base.Matrix{T}(a::_TriThresholdConversion) where {T}
    a.calls[]+=1
    return ones(T, 3, 3)
end
@testset "Tridiagonal policy validation precedes conversion" begin
    calls=Ref(0)
    a=_TriThresholdConversion(calls)
    @test_throws ArgumentError scratch_requirements(
        tridiagonalize,
        a;
        par_threshold = :never,
    )
    @test_throws ArgumentError tridiagonalize(a; par_threshold = :never)
    @test calls[]==0
end

@testset "Tridiagonal legacy workspace ABI remains derived" begin
    rng=MersenneTwister(773)
    team=FaerTeam(2)
    try
        for T in (Float32, Float64, ComplexF32, ComplexF64), resource in (nothing, team)
            R=typeof(real(zero(T)))
            tol=4096eps(R)
            X=randn(rng, T, 5, 5)
            A=X+X'
            info=zeros(UInt, 2)
            GC.@preserve info begin
                args=(UInt(5), Colmat._tag(T), UInt(0), UInt(0), info)
                Colmat._native_call(
                    Colmat.RustCrate.tridiagonalize_requirements,
                    args...,
                    resource,
                )
            end
            req=scratch_requirements(tridiagonalize, A; team = resource)
            @test (bytes = Int(info[1]), alignment = Int(info[2]))==req
            s=FaerScratch(req.bytes; alignment = req.alignment)
            Q=Matrix{T}(undef, 5, 5)
            d=Vector{R}(undef, 5)
            e=Vector{R}(undef, 4)
            try
                GC.@preserve A Q d e begin
                    args=(
                        s,
                        Ptr{UInt8}(pointer(A)),
                        UInt(5),
                        Ptr{UInt8}(pointer(Q)),
                        Ptr{UInt8}(pointer(d)),
                        Ptr{UInt8}(pointer(e)),
                        Colmat._tag(T),
                        UInt(0),
                        UInt(0),
                    )
                    Colmat._native_call(
                        Colmat.RustCrate.tridiagonalize_workspace,
                        args...,
                        resource,
                    )
                end
                @test A≈Q*Matrix(SymTridiagonal(d, e))*Q' rtol=tol atol=tol
            finally
                close(s)
            end
        end
    finally
        close(team)
    end
end
