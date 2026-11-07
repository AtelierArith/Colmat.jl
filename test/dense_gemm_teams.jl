using Test, LinearAlgebra, Colmat

_gemm_team_bytes(f) = (f(); @allocated f())

@testset "Dense mul! borrows a caller-owned team in place" begin
    for width in (1, 2)
        FaerTeam(width) do team
            for T in (Float32, Float64, ComplexF32, ComplexF64)
                tol=sqrt(eps(real(T)))*100
                a, b, c0=rand(T, 70, 40), rand(T, 40, 50), rand(T, 70, 50)
                alpha, beta=T(2), T(-0.5)
                expected=alpha*a*b+beta*c0
                A, B=FaerMatrix(a), FaerMatrix(b)
                At, Bt=FaerMatrix(copy(transpose(a))), FaerMatrix(copy(transpose(b)))
                # Native, borrowed, Julia and transposed operands and destinations.
                for (x, y) in ((A, B), (faerview(a), B), (A, b), (transpose(At), transpose(Bt)))
                    c=copy(c0)
                    @test mul!(c, x, y, alpha, beta; team)===c
                    @test c≈expected rtol=tol
                    f=FaerMatrix(c0)
                    @test mul!(f, x, y, alpha, beta; team)===f
                    @test Matrix(f)≈expected rtol=tol
                    g=FaerMatrix(c0)
                    @test mul!(g, x, y; team)===g
                    @test Matrix(g)≈a*b rtol=tol
                    # The positional team form writes in place too.
                    h=FaerMatrix(c0)
                    @test mul!(h, team, x, y, alpha, beta)===h
                    @test Matrix(h)≈expected rtol=tol
                end
                # Aliased operands are snapshot natively.
                s=FaerMatrix(rand(T, 40, 40))
                snapshot=Matrix(s)
                @test mul!(s, s, s; team)===s
                @test Matrix(s)≈snapshot*snapshot rtol=tol
                # Same-scalar products allocate only small Julia descriptors
                # (a 200x200 result alone would be at least 160 kB).
                P, Q, F=(FaerMatrix(rand(T, 200, 200)) for _ = 1:3)
                @test _gemm_team_bytes(() -> mul!(F, P, Q, alpha, beta; team))<10_000
                @test _gemm_team_bytes(() -> mul!(F, team, P, Q, alpha, beta))<10_000
                @test _gemm_team_bytes(() -> mul!(faerview(c0), A, B; team))<10_000
            end
            # Mixed scalar types promote through a staged destination.
            a=FaerMatrix(Float32[1 2; 3 4])
            b=FaerMatrix(ComplexF64[1 2im; 3 4])
            c=zeros(ComplexF64, 2, 2)
            @test mul!(c, a, b; team)===c
            @test c≈Matrix(a)*Matrix(b)
            # alpha == 0 and beta == 0 never read operands or the destination.
            a[1, 1]=NaN
            d=fill(NaN, 2, 2)
            @test mul!(d, a, a, 0, 0; team)==zeros(2, 2)
        end
    end
    team=FaerTeam(2)
    close(team)
    A=FaerMatrix(ones(3, 3))
    @test_throws ArgumentError mul!(FaerMatrix(zeros(3, 3)), A, A; team)
    @test_throws ArgumentError mul!(FaerMatrix(zeros(3, 3)), team, A, A)
end

@testset "Team gemm parallelizes large products" begin
    n=512
    a, b=rand(n, n), rand(n, n)
    FaerTeam(2) do team
        A, B, C=FaerMatrix(a), FaerMatrix(b), FaerMatrix(zeros(n, n))
        @test Matrix(mul!(C, A, B; team))≈a*b
    end
end

@testset "Dense factor reconstruction borrows a caller-owned team" begin
    rng_matrix(T, n)=(a=rand(T, n, n); a*a'+n*I)
    for width in (1, 2), T in (Float32, Float64, ComplexF32, ComplexF64)
        tol=sqrt(eps(real(T)))*100
        FaerTeam(width) do team
            a=rng_matrix(T, 60)
            for F in (
                lu(FaerMatrix(a)),
                full_piv_lu(FaerMatrix(a)),
                qr(FaerMatrix(a)),
                qr(FaerMatrix(a), ColumnNorm()),
                cholesky(Hermitian(FaerMatrix(a))),
                ldlt(Hermitian(FaerMatrix(a))),
                bunchkaufman(Hermitian(FaerMatrix(a))),
                (A=FaerMatrix(a); lu!(A; scratch = FaerScratch(lu!, A))),
                (A=FaerMatrix(a); qr!(A; scratch = FaerScratch(qr!, A))),
            )
                r=reconstruct(F; team)
                @test r isa FaerMatrix{T}
                @test Matrix(r)≈a rtol=tol
                @test Matrix(FaerMatrix(F; team))≈a rtol=tol
                @test Matrix(F; team)≈a rtol=tol
                @test Matrix{ComplexF64}(F; team)≈a rtol=tol
            end
        end
    end
    closed=FaerTeam(2)
    close(closed)
    F=lu(FaerMatrix(rand(4, 4)+4I))
    @test_throws ArgumentError reconstruct(F; team = closed)
    @test_throws ArgumentError Matrix(F; team = closed)
end
