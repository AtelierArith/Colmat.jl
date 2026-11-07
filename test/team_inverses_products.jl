using Test, LinearAlgebra, Colmat

_sweep_team_bytes(f) = (f(); @allocated f())
_sweep_spd(T, n) = (a=rand(T, n, n); a*a'+n*I)

@testset "Factor inverses borrow a caller-owned team" begin
    for width in (1, 2), T in (Float32, Float64, ComplexF32, ComplexF64)
        tol=sqrt(eps(real(T)))*1000
        FaerTeam(width) do team
            a=_sweep_spd(T, 50)
            expected=inv(a)
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
                (A=FaerMatrix(a); cholesky!(A; scratch = FaerScratch(cholesky!, A))),
            )
                X=inv(F; team)
                @test X isa FaerMatrix{T}
                @test Matrix(X)≈expected rtol=tol
            end
            @test Matrix(inv(FaerMatrix(a); team))≈expected rtol=tol
        end
    end
    closed=FaerTeam(2)
    close(closed)
    a=_sweep_spd(Float64, 4)
    for F in (
        lu(FaerMatrix(a)),
        lu(FaerMatrix(ComplexF32.(a))),
        (A=FaerMatrix(a); lu!(A; scratch = FaerScratch(lu!, A))),
        (A=FaerMatrix(a); qr!(A; scratch = FaerScratch(qr!, A))),
    )
        @test_throws ArgumentError inv(F; team = closed)
    end
    @test_throws ArgumentError inv(FaerMatrix(a); team = closed)
end

@testset "Explicit QR Q borrows a caller-owned team" begin
    for width in (1, 2), T in (Float32, Float64, ComplexF32, ComplexF64)
        FaerTeam(width) do team
            a=rand(T, 30, 20)
            for F in (
                qr(FaerMatrix(a)),
                qr(FaerMatrix(a), ColumnNorm()),
                (A=FaerMatrix(a); qr!(A; scratch = FaerScratch(qr!, A))),
            )
                q=qr_q(F; team)
                @test q isa FaerMatrix{T}
                @test size(q)==(30, 30)
                @test Matrix(q)==Matrix(F.Q)
                @test Matrix(qr_q(F))==Matrix(F.Q)
                thin=qr_q(F; thin = true, team)
                @test size(thin)==(30, 20)
                @test Matrix(thin)==Matrix(F.Qthin)
            end
        end
    end
    closed=FaerTeam(2)
    close(closed)
    @test_throws ArgumentError qr_q(qr(FaerMatrix(rand(4, 3))); team = closed)
    @test_throws MethodError qr_q(lu(FaerMatrix(rand(3, 3))))
end

@testset "Dense pseudoinverse borrows a caller-owned team" begin
    for width in (1, 2), T in (Float32, Float64, ComplexF32, ComplexF64)
        tol=sqrt(eps(real(T)))*1000
        FaerTeam(width) do team
            a=rand(T, 40, 25)
            A=FaerMatrix(a)
            @test Matrix(pinv(A; team))≈pinv(a) rtol=tol
            @test Matrix(pinv(A; team, rtol = 1e-3))≈Matrix(pinv(A; rtol = 1e-3)) rtol=tol
            @test Matrix(pinv(transpose(A); team))≈pinv(transpose(a)) rtol=tol
            @test Matrix(pinv(view(A, 1:30, 1:20); team))≈pinv(a[1:30, 1:20]) rtol=tol
            h=_sweep_spd(T, 20)
            @test Matrix(pinv(Hermitian(FaerMatrix(h)); team))≈inv(h) rtol=tol
        end
    end
    closed=FaerTeam(2)
    close(closed)
    @test_throws ArgumentError pinv(FaerMatrix(rand(3, 2)); team = closed)
    @test_throws ArgumentError pinv(Hermitian(FaerMatrix(_sweep_spd(Float64, 3))); team = closed)
end

@testset "Triangular structured mul! borrows a caller-owned team in place" begin
    for width in (1, 2), T in (Float32, Float64, ComplexF32, ComplexF64)
        tol=sqrt(eps(real(T)))*100
        FaerTeam(width) do team
            a, b, c0=rand(T, 60, 60), rand(T, 60, 60), rand(T, 60, 60)
            alpha, beta=T(2), T(-0.5)
            A, B=FaerMatrix(a), FaerMatrix(b)
            for (x, y, ex) in (
                (UpperTriangular(A), B, triu(a)*b),
                (A, LowerTriangular(B), a*tril(b)),
                (UnitLowerTriangular(A), UpperTriangular(B), (tril(a, -1)+I)*triu(b)),
            )
                C=FaerMatrix(c0)
                @test mul!(C, x, y, alpha, beta; team)===C
                @test Matrix(C)≈alpha*ex+beta*c0 rtol=tol
                D=FaerMatrix(c0)
                @test mul!(D, x, y; team)===D
                @test Matrix(D)≈ex rtol=tol
            end
            # structured_mul! writes only the selected destination block.
            C=FaerMatrix(c0)
            @test structured_mul!(C, UpperTriangular(A), B; dest = :upper, team)===C
            @test triu(Matrix(C))≈triu(triu(a)*b) rtol=tol
            @test tril(Matrix(C), -1)==tril(c0, -1)
            # Aliased operands are snapshot; vectors and diagonals accept team.
            S=FaerMatrix(a)
            @test mul!(S, UpperTriangular(S), S; team)===S
            @test Matrix(S)≈triu(a)*a rtol=tol
            v=FaerVector(zeros(T, 60))
            x=rand(T, 60)
            @test mul!(v, UpperTriangular(A), FaerVector(x); team)===v
            @test Vector(v)≈triu(a)*x rtol=tol
            E=FaerMatrix(c0)
            d=rand(T, 60)
            @test mul!(E, Diagonal(d), B; team)===E
            @test Matrix(E)≈Diagonal(d)*b rtol=tol
            # The native in-place path allocates only small Julia descriptors
            # (a 60x60 result alone would be at least 14 kB).
            F=FaerMatrix(c0)
            U=UpperTriangular(A)
            @test _sweep_team_bytes(() -> mul!(F, U, B, alpha, beta; team))<10_000
            @test _sweep_team_bytes(() -> structured_mul!(F, U, B; team))<10_000
        end
    end
    closed=FaerTeam(2)
    close(closed)
    A=FaerMatrix(rand(3, 3))
    C=FaerMatrix(zeros(3, 3))
    @test_throws ArgumentError mul!(C, UpperTriangular(A), A; team = closed)
    @test_throws ArgumentError structured_mul!(C, UpperTriangular(A), A; team = closed)
    @test_throws ArgumentError mul!(C, Diagonal(ones(3)), A; team = closed)
    @test Matrix(C)==zeros(3, 3)
end
