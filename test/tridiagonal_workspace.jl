using Test, Colmat, LinearAlgebra
@testset "Tridiagonal eigen workspace" begin
    team=FaerTeam(2)
    try
        for T in (Float32, Float64, ComplexF32, ComplexF64)
            R=typeof(real(zero(T)))
            for n in (0, 1, 4, 17, 65), threshold in (2, 128), resource in (nothing, team)
                d=T.(1:n)
                e=fill(T<:Complex ? T(0.25, 0.125) : T(0.25), max(0, n-1))
                original=(copy(d), copy(e))
                A=Matrix(Tridiagonal(e, d, conj.(e)))
                req=scratch_requirements(
                    tridiagonal_eigen,
                    d,
                    e;
                    team = resource,
                    recursion_threshold = threshold,
                )
                s=FaerScratch(
                    tridiagonal_eigen,
                    d,
                    e;
                    team = resource,
                    recursion_threshold = threshold,
                )
                try
                    @test scratch_capacity(s).bytes>=req.bytes
                    for rep = 1:2
                        F=tridiagonal_eigen(
                            d,
                            e;
                            scratch = s,
                            team = resource,
                            recursion_threshold = threshold,
                        )
                        X=Matrix(F.vectors)
                        @test norm(A*X-X*Diagonal(F.values)) <=
                              1024eps(R)*max(norm(A), one(R))
                        @test X'X ≈ Matrix{T}(I, n, n) atol=1024eps(R)
                        @test F.values ≈ eigvals(Hermitian(A)) rtol=1024eps(R) atol=1024eps(
                            R,
                        )
                        close(F.vectors)
                    end
                    reserve!(
                        s,
                        tridiagonal_eigvals,
                        d,
                        e;
                        team = resource,
                        recursion_threshold = threshold,
                    )
                    @test tridiagonal_eigvals(
                        d,
                        e;
                        scratch = s,
                        team = resource,
                        recursion_threshold = threshold,
                    ) ≈ eigvals(Hermitian(A)) rtol=1024eps(R) atol=1024eps(R)
                    @test (d, e)==original
                    if req.bytes>0
                        short=FaerScratch(req.bytes-1; alignment = req.alignment)
                        try
                            @test_throws ArgumentError tridiagonal_eigen(
                                d,
                                e;
                                scratch = short,
                                team = resource,
                                recursion_threshold = threshold,
                            )
                            reserve!(
                                short,
                                tridiagonal_eigen,
                                d,
                                e;
                                team = resource,
                                recursion_threshold = threshold,
                            )
                            @test tridiagonal_eigvals(
                                d,
                                e;
                                scratch = short,
                                team = resource,
                                recursion_threshold = threshold,
                            ) ≈ eigvals(Hermitian(A)) rtol=1024eps(R) atol=1024eps(R)
                        finally
                            close(short)
                        end
                    end
                finally
                    close(s)
                end
                @test_throws ArgumentError tridiagonal_eigen(
                    d,
                    e;
                    scratch = s,
                    team = resource,
                )
            end
            exponent=R===Float32 ? 100 : 1000
            for scale in (R(2)^(-exponent), R(2)^exponent)
                d=T[1, 2, 3] .* scale
                e=fill(T<:Complex ? T(0.25, 0.125)*scale : T(0.25)*scale, 2)
                A=Matrix(Tridiagonal(e, d, conj.(e))) ./ scale
                F=tridiagonal_eigen(d, e; team)
                X=Matrix(F.vectors)
                @test norm(A*X-X*Diagonal(F.values ./ scale)) <= 1024eps(R)*norm(A)
                close(F.vectors)
            end
            d=FaerVector(T[1, 2, 3])
            e=FaerVector(T[0.25, 0.25])
            try
                s=FaerScratch(tridiagonal_eigen, d, e; team)
                @test fetch(@async tridiagonal_eigvals(d, e; scratch = s, team)) ≈
                      eigvals(SymTridiagonal(R[1, 2, 3], R[0.25, 0.25]))
                old=d.len
                d.len+=1
                try
                    @test_throws Exception scratch_requirements(
                        tridiagonal_eigen,
                        d,
                        e;
                        team,
                    )
                    @test_throws Exception tridiagonal_eigen(d, e; scratch = s, team)
                finally
                    d.len=old
                end
                close(s)
            finally
                close(d)
                close(e)
            end
            @test_throws ArgumentError scratch_requirements(tridiagonal_eigen, d, e)
        end
    finally
        close(team)
    end
end
@testset "Saved tridiagonal resources" begin
    team=FaerTeam(2)
    try
        for T in (Float32, Float64, ComplexF32, ComplexF64)
            R=typeof(real(zero(T)))
            a=Matrix{T}(I, 4, 4)
            a[2, 1]=T<:Complex ? T(0.2, 0.3) : T(0.2)
            a[1, 2]=conj(a[2, 1])
            f=tridiagonalize(a)
            s=FaerScratch(eigen, f; team)
            try
                @test scratch_requirements(eigen, f; team)==scratch_requirements(
                    tridiagonal_eigen,
                    f.H.dv,
                    f.H.ev;
                    team,
                )
                E=eigen(f; scratch = s, team)
                X=Matrix(E.vectors)
                @test a*X ≈ X*Diagonal(E.values) rtol=1024eps(R) atol=1024eps(R)
                close(E.vectors)
                reserve!(s, eigvals, f; team)
                @test eigvals(f; scratch = s, team) ≈ eigvals(Hermitian(a)) rtol=1024eps(R) atol=1024eps(
                    R,
                )
                old=f.Q.dims
                f.Q.dims=(3, 3)
                try
                    @test_throws Exception eigen(f; scratch = s, team)
                    @test_throws Exception scratch_requirements(eigen, f; team)
                finally
                    f.Q.dims=old
                end
                @test_throws ArgumentError tridiagonal_eigen(
                    f.H.dv,
                    f.H.ev;
                    scratch = "invalid",
                    team,
                )
                req=scratch_requirements(eigen, f; team)
                bad=FaerScratch(req.bytes; alignment = 1)
                try
                    @test_throws ArgumentError eigen(f; scratch = bad, team)
                finally
                    close(bad)
                end
            finally
                close(s)
                close(f)
            end
            @test_throws ArgumentError scratch_requirements(eigen, f; team)
            if T<:Real
                d=FaerVector(T[1, 2, 3])
                e=FaerVector(T[0.25, 0.25])
                h=SymTridiagonal(d, e)
                work=FaerScratch(eigen, h; team)
                try
                    E=eigen(h; scratch = work, team)
                    @test E.values ≈ eigvals(SymTridiagonal(T[1, 2, 3], T[0.25, 0.25])) rtol=1024eps(
                        R,
                    ) atol=1024eps(R)
                    close(E.vectors)
                finally
                    close(work)
                    close(d)
                    close(e)
                end
            end
        end
    finally
        close(team)
    end
    @test_throws ArgumentError scratch_requirements(
        tridiagonal_eigen,
        [1.0],
        Float64[];
        team,
    )
end
