using Test, LinearAlgebra, Random, Colmat

@testset "Reusable native factorizations" begin
    rng = MersenneTwister(18)
    a = [0.0 2 1; 3 1 2; 1 0 4]
    b = randn(rng, 3, 2)
    A, B = FaerMatrix(a), FaerMatrix(b)
    @testset "LU and full-pivot LU" begin
        for factor in (lu, full_piv_lu)
            f = factor(A)
            @test f isa FaerFactorization
            @test size(f)==size(a)
            @test issuccess(f)
            @test Matrix(f) ≈ a
            if factor === lu
                @test Matrix(f.L*f.U) ≈ a[f.p, :]
                @test Matrix(f.P)*a ≈ Matrix(f.L*f.U)
            else
                @test Matrix(f.L*f.U) ≈ a[f.p, f.q]
            end
            @test Matrix(f\B) ≈ a\b
            @test f\b[:, 1] ≈ a\b[:, 1]
            @test Matrix(f'\B) ≈ a'\b
            @test Matrix(transpose(f)\B) ≈ transpose(a)\b
            @test Matrix(inv(f)) ≈ inv(a)
            for g in (copy(f), deepcopy(f))
                @test Matrix(g) ≈ a
                close(g)
                @test !isopen(g)
                @test Matrix(f) ≈ a
            end
            C = copy(B)
            @test ldiv!(f, C) === C
            @test Matrix(C) ≈ a\b
            C=copy(B)
            @test ldiv!(C, f, C) === C
            @test Matrix(C) ≈ a\b
            c=zeros(3, 2)
            @test ldiv!(c, f, b) === c
            @test c≈a\b
            close(f)
            @test !isopen(f)
            @test_throws ArgumentError f\B
            @test_throws ArgumentError Matrix(f)
            @test isnothing(close(f))
        end
        singular = FaerMatrix([1.0 2; 2 4])
        @test_throws SingularException lu(singular)
        f = lu(singular; check = false)
        @test !issuccess(f)
        @test Matrix(f) ≈ Matrix(singular)
        @test_throws SingularException f\ones(2)
        @test !issuccess(lu(singular; allowsingular = true))
        @test lu(A, RowMaximum()) isa FaerFactorization
        @test Matrix(inv(A))≈inv(a)
    end

    @testset "QR" begin
        for (m, n) in ((5, 3), (3, 5), (3, 3), (0, 3), (3, 0), (0, 0))
            a = randn(rng, m, n)
            A = FaerMatrix(a)
            b = randn(rng, m, 2)
            for pivot in (NoPivot(), ColumnNorm())
                f=qr(A, pivot)
                @test f isa FaerFactorization
                @test Matrix(f)≈a
                @test Matrix(f.Qthin*f.R)≈(pivot isa ColumnNorm ? a[:, f.p] : a)
                @test Matrix(f.Q)'*Matrix(f.Q)≈Matrix{Float64}(I, m, m)
                @test size(f.Qthin)==(m, min(m, n))
                @test size(f.R)==(min(m, n), n)
                @test Matrix(f\FaerMatrix(b)) ≈ pinv(a)*b
                @test Matrix(A\FaerMatrix(b)) ≈ pinv(a)*b
            end
        end
    end

    @testset "Cholesky, LDLT, block LDLT" begin
        a=[4.0 1 2; 1 5 1; 2 1 6]
        b=randn(rng, 3, 2)
        for (kind, ctor) in ((:cholesky, cholesky), (:ldlt, ldlt), (:block, bunchkaufman))
            for input in
                (FaerMatrix(a), Symmetric(FaerMatrix(a), :L), Hermitian(FaerMatrix(a), :U))
                f=ctor(input)
                @test f isa FaerFactorization
                @test Matrix(f)≈a
                @test Matrix(f\FaerMatrix(b))≈a\b
                if kind===:cholesky
                    @test Matrix(f.L)*Matrix(f.U)≈a
                elseif kind===:ldlt
                    @test Matrix(f.L*f.D*copy(f.L'))≈a
                else
                    @test Matrix(f.L*f.D*copy(f.L'))≈a[f.p, f.p]
                end
            end
        end
        @test_throws PosDefException cholesky(FaerMatrix([1.0 2; 2 1]))
        @test !issuccess(cholesky(FaerMatrix([1.0 2; 2 1]); check = false))
        @test_throws PosDefException cholesky(FaerMatrix([1.0 2; 0 1]))
        A=FaerMatrix([4.0 1; 1 3])
        A[1, 2]=NaN
        f=cholesky(Symmetric(A, :L))
        @test Matrix(f)≈[4.0 1; 1 3]
        a=[0.0 1; 1 0]
        f=bunchkaufman(Symmetric(FaerMatrix(a)))
        @test Matrix(f.L*f.D*copy(f.L'))≈a[f.p, f.p]
        @test f\[1.0, 2]≈a\[1.0, 2]
        f=lu(FaerMatrix([2.0 0; 0 3]))
        GC.gc()
        @test f\[2.0, 3]≈ones(2)
    end
    @test FaerMatrix(ones(3, 2)) \ ones(3) ≈ fill(0.5, 2)
    @test isempty(Test.detect_ambiguities(Colmat; recursive = true))
end

@testset "Determinants of native factors" begin
    a=[0.0 2 1; 3 1 2; 1 0 4]
    for ctor in (lu, full_piv_lu, qr)
        f=ctor(FaerMatrix(a))
        @test det(f)≈det(a)
        value, sign=logabsdet(f)
        expected, esign=logabsdet(a)
        @test value≈expected
        @test sign==esign
    end
    @test det(FaerMatrix(zeros(0, 0)))==1
    @test logabsdet(FaerMatrix([1e200 0.0; 0 1e200]))[1]≈400log(10)
    @test logdet(FaerMatrix([2.0 0; 0 3]))≈log(6)
    @test_throws DomainError logdet(FaerMatrix([-2.0 0; 0 3]))
    @test logabsdet(FaerMatrix([0.0 0; 0 3]))==(-Inf, 0.0)
    for ctor in (cholesky, ldlt, bunchkaufman)
        a=[4.0 1; 1 3]
        f=ctor(FaerMatrix(a))
        @test det(f)≈det(a)
        @test logdet(f)≈logdet(a)
    end
    f=bunchkaufman(FaerMatrix([0.0 1; 1 0]))
    @test det(f)≈-1
    @test isposdef(FaerMatrix([4.0 1; 1 3]))
    @test !isposdef(FaerMatrix([1.0 2; 2 1]))
    A=FaerMatrix(a)
    f=lu(A)
    fill!(A, 0)
    close(A)
    GC.gc()
    @test f\[1.0, 2]≈a\[1.0, 2]
end
