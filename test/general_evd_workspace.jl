using Test, Colmat, LinearAlgebra, SparseArrays
@testset "General EVD workspace and teams" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), n in (0, 1, 4), width in (1, 2)
        FaerTeam(width) do team
            A=reshape(T.(1:(n*n)), n, n)+4Matrix{T}(I, n, n)
            if T<:Complex
                for j = 1:n, i = 1:n
                    A[i, j]+=T((i+2j)*0.05im)
                end
            end
            a=FaerMatrix(A)
            s=FaerScratch(eigen, a; team)
            req=scratch_requirements(eigen, a; team)
            @test scratch_capacity(s)==req
            F=eigen(a; scratch = s, team)
            @test A*F.vectors≈F.vectors*Diagonal(F.values) atol=2e-3
            @test F.values≈eigvals(A; sortby = x->(real(x), imag(x))) atol=2e-3
            @test eltype(F.values)==Complex{typeof(real(zero(T)))}
            @test Matrix(a)==A
            for j = 1:n
                @test norm(F.vectors[:, j])≈1 atol=2e-4
            end
            reserve!(s, eigvals, a; team)
            @test eigvals(a; scratch = s, team)≈F.values atol=2e-3
            @test eigen(a; team).values≈F.values atol=2e-3
            @test scratch_requirements(eigen!, a; team)==req
            @test scratch_requirements(eigvals!, a; team)==scratch_requirements(
                eigvals,
                a;
                team,
            )
            b=FaerMatrix(A)
            G=eigen!(b; scratch = s, team)
            @test A*G.vectors≈G.vectors*Diagonal(G.values) atol=2e-3
            @test istriu(Matrix(b), T<:Real ? -1 : 0)
            b=FaerMatrix(A)
            vals=eigvals!(b; scratch = s, team)
            @test vals≈F.values atol=2e-3
            close(s)
        end
    end
end
@testset "Conjugate pairs and real workspace conversion" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        A=T[0 -1 1 0; 1 0 0 1; 0 0 0 -2; 0 0 2 0]
        a=FaerMatrix(A)
        s=FaerScratch(eigen, a)
        F=eigen(a; scratch = s, sortby = imag)
        @test imag.(F.values)≈[-2, -1, 1, 2] atol=1e-4
        @test A*F.vectors≈F.vectors*Diagonal(F.values) atol=2e-4
        @test eigvals(a; scratch = s, sortby = imag)≈F.values atol=2e-4
        z=FaerMatrix(zeros(T, 4, 4))
        G=eigen(z; scratch = s)
        @test G.values==zeros(Complex{typeof(real(zero(T)))}, 4)
        @test norm(G.vectors)>0
        close(s)
    end
end
@testset "General EVD failure transactions and shape guards" begin
    a=FaerMatrix([1.0 2; 3 4])
    A=Matrix(a)
    s=FaerScratch(eigen, a)
    @test_throws ErrorException eigen!(a; scratch = s, sortby = x->error("sort failed"))
    @test Matrix(a)==A
    tiny=FaerScratch()
    @test_throws ArgumentError eigvals!(a; scratch = tiny)
    @test Matrix(a)==A
    @test_throws ArgumentError eigen(a; scratch = 1)
    @test_throws ArgumentError eigen(a; team = 1)
    a[1, 1]=NaN
    @test_throws Colmat.RustCall.RustError eigen!(a; scratch = s)
    @test isnan(a[1, 1])
    @test a[2, 2]==4
    close(s)
    close(tiny)
    bad=FaerMatrix([1.0 2; 3 4])
    bad.dims=(3, 3)
    s=FaerScratch(eigen, bad)
    @test_throws Colmat.RustCall.RustError eigen(bad; scratch = s)
    close(s)
    close(bad)
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
@testset "Blocked general EVD with known cyclic spectrum" begin
    n=48
    for T in (Float64, ComplexF64)
        rho=T<:Complex ? T(0.9+0.2im) : one(T)
        A=zeros(T, n, n)
        for i = 1:n
            A[i, mod1(i+1, n)]=rho
        end
        a=FaerMatrix(A)
        FaerTeam(2) do team
            s=FaerScratch(eigen, a; team)
            F=eigen(a; scratch = s, team, sortby = nothing)
            @test A*F.vectors≈F.vectors*Diagonal(F.values) atol=1e-9
            @test abs.(F.values)≈fill(abs(rho), n) atol=1e-9
            @test (F.values ./ rho) .^ n≈ones(ComplexF64, n) atol=1e-8
            @test F.vectors'*F.vectors≈I atol=1e-8
            close(s)
        end
    end
end
