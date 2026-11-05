using Test, Colmat, LinearAlgebra, Random

@testset "Bidiagonal parallel policies" begin
    @test FaerParThreshold()==FaerParThreshold(:derived)
    @test FaerParThreshold(0)==FaerParThreshold(1)
    @test FaerParThreshold(0)!=FaerParThreshold()
    @test_throws ArgumentError FaerParThreshold(-1)
    @test_throws ArgumentError FaerParThreshold(typemax(UInt))
    @test_throws ArgumentError FaerParThreshold(big(typemax(UInt))+1)
    @test_throws ArgumentError FaerParThreshold(:unknown)
    rng=MersenneTwister(771)
    team=FaerTeam(2)
    try
        policies=(
            FaerParThreshold(),
            FaerParThreshold(0),
            FaerParThreshold(4096),
            FaerParThreshold(:never),
        )
        for T in (Float32, Float64, ComplexF32, ComplexF64),
            (m, n) in ((0, 3), (3, 0), (1, 1), (5, 3), (3, 5), (131, 129), (129, 131))

            R=typeof(real(zero(T)))
            tol=4096eps(R)
            k=min(m, n)
            A=randn(rng, T, m, n)
            snapshot=copy(A)
            for p in policies, full in (false, true), resource in (nothing, team)
                req=scratch_requirements(
                    bidiagonalize,
                    A;
                    full,
                    team = resource,
                    par_threshold = p,
                )
                s=FaerScratch(bidiagonalize, A; full, team = resource, par_threshold = p)
                try
                    F=bidiagonalize(
                        A;
                        full,
                        scratch = s,
                        team = resource,
                        par_threshold = p,
                    )
                    try
                        U, V=Matrix(F.U), Matrix(F.V)
                        @test norm(A-U[:, 1:k]*Matrix(F.B)*V[:, 1:k]')<=tol*max(
                            norm(A),
                            one(R),
                        )
                        @test U'U≈Matrix{T}(I, size(U, 2), size(U, 2)) atol=tol
                        @test V'V≈Matrix{T}(I, size(V, 2), size(V, 2)) atol=tol
                    finally
                        close(F)
                    end
                    @test scratch_requirements(
                        bidiagonalize!,
                        A;
                        full,
                        team = resource,
                        par_threshold = p,
                    )==req
                    if req.bytes>0
                        short=FaerScratch(req.bytes-1; alignment = req.alignment)
                        try
                            @test_throws ArgumentError bidiagonalize(
                                A;
                                full,
                                scratch = short,
                                team = resource,
                                par_threshold = p,
                            )
                            reserve!(
                                short,
                                bidiagonalize,
                                A;
                                full,
                                team = resource,
                                par_threshold = p,
                            )
                            F=bidiagonalize(
                                A;
                                full,
                                scratch = short,
                                team = resource,
                                par_threshold = p,
                            )
                            @test Matrix(F)≈A rtol=tol atol=tol
                            close(F)
                        finally
                            close(short)
                        end
                    end
                    @test A==snapshot
                finally
                    close(s)
                end
                @test_throws ArgumentError bidiagonalize(
                    A;
                    scratch = s,
                    team = resource,
                    par_threshold = p,
                )
            end
            @test scratch_requirements(
                bidiagonalize,
                A;
                full = true,
                team,
                par_threshold = FaerParThreshold(:never),
            ) == scratch_requirements(
                bidiagonalize,
                A;
                full = true,
                par_threshold = FaerParThreshold(:never),
            )
        end
        for p in policies
            a=FaerMatrix(ComplexF64[1 2 3; 4+im 5 6])
            original=Matrix(a)
            s=FaerScratch(bidiagonalize!, a; team, par_threshold = p)
            try
                F=fetch(@async bidiagonalize!(a; scratch = s, team, par_threshold = p))
                try
                    @test Matrix(F)≈original
                    @test Matrix(a)==hcat(Matrix(F.B), zeros(ComplexF64, 2, 1))
                finally
                    close(F)
                end
            finally
                close(s)
                close(a)
            end
        end
    finally
        close(team)
    end
    @test_throws ArgumentError scratch_requirements(
        bidiagonalize,
        zeros(0, 3);
        team,
        par_threshold = FaerParThreshold(:never),
    )
    @test_throws ArgumentError bidiagonalize(
        zeros(0, 3);
        team,
        par_threshold = FaerParThreshold(:never),
    )
end

struct _ThresholdConversion <: AbstractMatrix{Float64}
    calls::Base.RefValue{Int}
end
Base.size(::_ThresholdConversion) = (2, 3)
Base.getindex(::_ThresholdConversion, i::Int, j::Int) = Float64(i+j)
function Base.Matrix{T}(a::_ThresholdConversion) where {T}
    a.calls[]+=1
    return ones(T, 2, 3)
end
@testset "Bidiagonal policy validation precedes conversion" begin
    # Reuse an observable conversion rather than relying on a native crash.
    calls=Ref(0)
    a=_ThresholdConversion(calls)
    @test_throws ArgumentError scratch_requirements(
        bidiagonalize,
        a;
        par_threshold = :never,
    )
    @test_throws ArgumentError bidiagonalize(a; par_threshold = :never)
    @test calls[]==0
end

@testset "Bidiagonal legacy workspace ABI remains derived" begin
    team=FaerTeam(2)
    try
        for T in (Float32, Float64, ComplexF32, ComplexF64), resource in (nothing, team)
            R=typeof(real(zero(T)))
            tol=4096eps(R)
            A=reshape(T.(1:15), 5, 3)
            T<:Complex && (A .+= T(0.2, -0.1))
            info=zeros(UInt, 2)
            GC.@preserve info begin
                args=(UInt(5), UInt(3), Colmat._tag(T), false, UInt(0), pointer(info))
                isnothing(resource) ?
                Colmat._native_call(Colmat.RustCrate.bidiagonalize_requirements, args...) :
                Colmat._native_call(
                    Colmat.RustCrate.bidiagonalize_team_requirements,
                    resource,
                    args...,
                )
            end
            req=scratch_requirements(bidiagonalize, A; team = resource)
            @test (bytes = Int(info[1]), alignment = Int(info[2]))==req
            s=FaerScratch(req.bytes; alignment = req.alignment)
            U=Matrix{T}(undef, 5, 3)
            V=Matrix{T}(undef, 3, 3)
            d=Vector{T}(undef, 3)
            e=Vector{T}(undef, 2)
            try
                GC.@preserve A U V d e begin
                    args=(
                        s,
                        Ptr{UInt8}(pointer(A)),
                        UInt(5),
                        UInt(3),
                        Ptr{UInt8}(pointer(U)),
                        Ptr{UInt8}(pointer(V)),
                        Ptr{UInt8}(pointer(d)),
                        Ptr{UInt8}(pointer(e)),
                        Colmat._tag(T),
                        false,
                        UInt(0),
                    )
                    isnothing(resource) ?
                    Colmat._native_call(Colmat.RustCrate.bidiagonalize_workspace, args...) :
                    Colmat._native_call(
                        Colmat.RustCrate.bidiagonalize_workspace_team,
                        args...,
                        resource,
                    )
                end
                @test A≈U*Matrix(Bidiagonal(d, e, :U))*V' rtol=tol atol=tol
            finally
                close(s)
            end
        end
    finally
        close(team)
    end
end
