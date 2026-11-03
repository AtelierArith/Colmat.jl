using Test, Colmat, LinearAlgebra

@testset "Definite generalized self-adjoint workspace" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        n in (0, 1, 4),
        ua in (:L, :U),
        ub in (:L, :U)

        X=reshape(T.(1:(n*n)), n, n)/max(n*n, 1)
        T<:Complex && (X .+= T(0.1im))
        A=X+X'+Matrix{T}(I, n, n)
        B=X*X'+2Matrix{T}(I, n, n)
        a, b=FaerMatrix(A), FaerMatrix(B)
        if n>1
            a[ua===:L ? 1 : 2, ua===:L ? 2 : 1]=T(NaN)
            b[ub===:L ? 1 : 2, ub===:L ? 2 : 1]=T(NaN)
        end
        H, K=Hermitian(a, ua), Hermitian(b, ub)
        FaerTeam(2) do team
            s=FaerScratch(eigen, H, K; team)
            @test scratch_capacity(s)==scratch_requirements(eigen!, H, K; team)
            F=eigen(H, K; scratch = s, team)
            @test A*F.vectors≈B*F.vectors*Diagonal(F.values) atol=5e-4
            @test F.vectors'*B*F.vectors≈Matrix{T}(I, n, n) atol=5e-4
            @test F.values≈eigvals(Hermitian(A), Hermitian(B)) atol=5e-4
            @test eltype(F.values)==typeof(real(zero(T)))
            reserve!(s, eigvals!, H, K; team)
            @test eigvals(H, K; scratch = s, team)≈F.values atol=5e-4
            @test eigen(H, K; team).values≈F.values atol=5e-4
            x, y=FaerMatrix(A), FaerMatrix(B)
            G=eigen!(Hermitian(x, ua), Hermitian(y, ub); scratch = s, team)
            @test Matrix(x)==G.vectors
            @test Matrix(y)==Matrix{T}(Diagonal(G.values))
            x, y=FaerMatrix(A), FaerMatrix(B)
            v=eigvals!(Hermitian(x, ua), Hermitian(y, ub); scratch = s, team)
            @test Matrix(x)==Matrix{T}(Diagonal(v))
            @test Matrix(y)==Matrix{T}(Diagonal(v))
            close(s)
        end
    end
end
@testset "Generalized workspace errors and promotion" begin
    a, b=FaerMatrix([4.0 1; 1 3]), FaerMatrix([2.0 0; 0 1])
    H, K=Hermitian(a), Hermitian(b)
    s=FaerScratch(eigen, H, K)
    @test_throws ArgumentError eigen(H, K; scratch = FaerScratch())
    @test_throws ArgumentError eigen(H, K; team = 1)
    @test_throws ErrorException eigen!(H, K; scratch = s, sortby = x->error("sort failed"))
    @test Matrix(a)==[4 1; 1 3]
    @test Matrix(b)==[2 0; 0 1]
    @test_throws ArgumentError eigen!(H, H; scratch = s)
    b[1, 1]=-1
    @test_throws Colmat.RustCall.RustError eigen!(H, K; scratch = s)
    @test Matrix(a)==[4 1; 1 3]
    @test Matrix(b)==[-1 0; 0 1]
    close(s)
    for T in (ComplexF32, ComplexF64)
        a=FaerMatrix(T[4 1im; -1im 3])
        b=FaerMatrix(T[2 0; 0 1])
        a[1, 1]=T(4, NaN)
        b[2, 2]=T(1, NaN)
        H, K=Hermitian(a), Hermitian(b)
        s=FaerScratch(eigen, H, K)
        F=eigen(H, K; scratch = s)
        @test T[4 1im; -1im 3]*F.vectors≈T[2 0; 0 1]*F.vectors*Diagonal(F.values) atol=1e-4
        close(s)
    end
    H=Hermitian(FaerMatrix(Float32[4 1; 1 3]))
    K=Hermitian(FaerMatrix([2.0 0; 0 1]))
    s=FaerScratch(eigen, H, K)
    @test eltype(eigen(H, K; scratch = s).values)==Float64
    close(s)
end
@testset "Generalized blocked and metadata paths" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        n=64
        A=Matrix{T}(Diagonal(T.(1:n)))
        B=Matrix{T}(Diagonal(T.(2:(n+1))))
        H, K=Hermitian(FaerMatrix(A)), Hermitian(FaerMatrix(B))
        FaerTeam(2) do team
            s=FaerScratch(eigen, H, K; team)
            F=eigen(H, K; scratch = s, team)
            @test F.values≈[i/(i+1) for i = 1:n] atol=1e-4
            @test A*F.vectors≈B*F.vectors*Diagonal(F.values) atol=1e-4
            @test F.vectors'*B*F.vectors≈I atol=1e-4
            close(s)
        end
    end
    H=Hermitian(FaerMatrix([2.0 1; 1 3]))
    K=Hermitian(FaerMatrix([2.0 0; 0 1]))
    parent(K).dims=(3, 3)
    @test_throws DimensionMismatch scratch_requirements(eigen, H, K)
    parent(H).dims=(3, 3)
    s=FaerScratch(eigen, H, K)
    @test_throws Colmat.RustCall.RustError eigen(H, K; scratch = s)
    close(s)
end
