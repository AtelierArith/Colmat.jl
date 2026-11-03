using Test, Colmat, LinearAlgebra, SparseArrays
@testset "Caller-owned native Pfaffian workspace" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), n in (0, 2, 4, 8)
        A=zeros(T, n, n)
        expected=one(T)
        for i = 1:2:n
            x=T(i+1)+(T<:Complex ? T(0.5im) : zero(T))
            A[i, i+1]=x
            A[i+1, i]=-x
            expected*=x
        end
        a=FaerMatrix(A)
        req=scratch_requirements(pfaffian, a)
        s=FaerScratch(pfaffian, a)
        @test scratch_capacity(s)==req
        for _ = 1:3
            @test pfaffian(a; scratch = s)≈expected
            @test pfaffian(a; scratch = s)≈pfaffian(a)
            @test scratch_capacity(s)==req
        end
        @test Matrix(a)==A
        copy_s=deepcopy(s)
        @test copy_s.handle!==s.handle
        @test scratch_capacity(copy_s)==req
        close(s)
        close(s)
        @test !isopen(s)
        @test isopen(copy_s)
        @test_throws ArgumentError pfaffian(a; scratch = s)
        @test pfaffian(a; scratch = copy_s)≈expected
        tiny=FaerScratch()
        if req.bytes>0
            @test_throws ArgumentError pfaffian(a; scratch = tiny)
        end
        @test reserve!(tiny, pfaffian, a)===tiny
        @test pfaffian(a; scratch = tiny)≈expected
        previous=scratch_capacity(tiny)
        reserve!(tiny, 0; alignment = 1)
        @test scratch_capacity(tiny)==previous
        close(copy_s)
        close(tiny)
    end
    @test_throws ArgumentError FaerScratch(-1)
    @test_throws ArgumentError FaerScratch(1; alignment = 3)
    @test_throws ArgumentError FaerScratch(1; alignment = 0)
    s=FaerScratch(64; alignment = 1)
    @test scratch_capacity(s)==(bytes = 64, alignment = 1)
    @test reserve!(s, 128; alignment = 64)===s
    @test scratch_capacity(s)==(bytes = 128, alignment = 64)
    a=FaerMatrix([0.0 1; -1 0])
    @test_throws ArgumentError pfaffian(a; scratch = 1)
    @test_throws DimensionMismatch scratch_requirements(pfaffian, FaerMatrix(ones(2, 3)))
    reserve!(s, pfaffian, a)
    @test_throws Colmat.RustCall.RustError pfaffian(FaerMatrix(ones(2, 2)); scratch = s)
    @test pfaffian(a; scratch = s)==1.0
    close(a)
    @test_throws ArgumentError scratch_requirements(pfaffian, a)
    @test isempty(
        Test.detect_ambiguities(
            Colmat,
            Base,
            LinearAlgebra,
            SparseArrays;
            recursive = true,
        ),
    )
end
@testset "Pfaffian scratch with caller-owned teams" begin
    for width in (1, 2)
        FaerTeam(width) do team
            for T in (Float32, Float64, ComplexF32, ComplexF64)
                A=zeros(T, 8, 8)
                for i = 1:2:8
                    A[i, i+1]=T(2)+(T<:Complex ? T(0.25im) : zero(T))
                    A[i+1, i]=-A[i, i+1]
                end
                a=FaerMatrix(A)
                scratch=FaerScratch(pfaffian, a)
                expected=pfaffian(a)
                for _ = 1:2
                    @test pfaffian(a; scratch, team)≈expected
                end
                @test pfaffian(a; team)≈expected
                @test Matrix(a)==A
                req=scratch_requirements(pfaffian, a)
                if req.alignment>1
                    bad=FaerScratch(req.bytes; alignment = 1)
                    @test_throws ArgumentError pfaffian(a; scratch = bad, team)
                    close(bad)
                end
                close(scratch)
            end
        end
    end
    a=FaerMatrix([0.0 1; -1 0])
    @test_throws ArgumentError pfaffian(a; team = 1)
    team=FaerTeam(1)
    close(team)
    @test_throws ArgumentError pfaffian(a; team)
    odd=FaerMatrix(zeros(3, 3))
    scratch=FaerScratch(pfaffian, odd)
    @test_throws Colmat.RustCall.RustError pfaffian(odd; scratch)
    close(scratch)
end
@testset "Workspace reuse across scalar types and blocked Pfaffian" begin
    n=64
    L=Matrix{Float64}(I, n, n)+tril(reshape(sin.(collect(1:(n*n))), n, n), -1)/20
    J=zeros(n, n)
    for i = 1:2:n
        J[i, i+1]=1
        J[i+1, i]=-1
    end
    raw=L*J*transpose(L)
    lower=tril(raw, -1)
    A=FaerMatrix(lower-transpose(lower))
    scratch=FaerScratch(pfaffian, FaerMatrix(zeros(ComplexF64, n, n)))
    before=scratch_capacity(scratch)
    FaerTeam(2) do team
        @test pfaffian(A; scratch, team)≈1.0 rtol=1e-9
        @test pfaffian(A; scratch)≈1.0 rtol=1e-9
        for T in (Float32, Float64, ComplexF32, ComplexF64)
            x=T(2)+(T<:Complex ? T(0.5im) : zero(T))
            a=FaerMatrix(T[0 x; -x 0])
            @test pfaffian(a; scratch, team)≈x
            @test scratch_capacity(scratch)==before
        end
    end
    @test_throws ArgumentError reserve!(scratch, 1; alignment = 3)
    @test scratch_capacity(scratch)==before
    close(scratch)
end
