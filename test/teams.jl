using Test, Colmat, LinearAlgebra, SparseArrays
@testset "Caller-owned reusable native teams" begin
    for width in (1, 2)
        FaerTeam(width) do team
            @test team.width==width
            @test isopen(team)
            for T in (Float32, Float64, ComplexF32, ComplexF64),
                dims in ((0, 0, 0), (3, 0, 4), (0, 3, 4), (4, 3, 0), (4, 3, 5))

                m, k, n=dims
                A=reshape(T.(1:(m*k)), m, k)
                B=reshape(T.(1:(k*n)), k, n)
                T<:Complex && (A .+= T(0.5im); B .+= T(0.25im))
                a, b=FaerMatrix(A), FaerMatrix(B)
                for _ = 1:2
                    C=team_mul(team, a, b)
                    @test C isa FaerMatrix{T}
                    @test Matrix(C)≈A*B
                end
                C=FaerMatrix(ones(T, m, n))
                @test mul!(C, team, a, b, 2, 3)===C
                @test Matrix(C)≈2A*B .+3
                @test Matrix(team_mul(team, A, B))≈A*B
            end
            A=FaerMatrix([1.0 2; 3 4])
            snapshot=Matrix(A)
            @test mul!(A, team, A, A)===A
            @test Matrix(A)≈snapshot*snapshot
            C=FaerMatrix(zeros(ComplexF64, 2, 2))
            @test mul!(C, team, A, A, 1im, 0)===C
            @test Matrix(C)≈1im*Matrix(A)^2
            @test fetch(@async Matrix(team_mul(team, A, A)))≈Matrix(A)^2
            @test_throws DimensionMismatch team_mul(team, ones(2, 3), ones(2, 4))
        end
    end
    team=FaerTeam(2)
    clone=deepcopy(team)
    close(team)
    close(team)
    @test !isopen(team)
    @test isopen(clone)
    @test_throws ArgumentError team_mul(team, ones(2, 2), ones(2, 2))
    @test Matrix(team_mul(clone, ones(2, 2), ones(2, 2)))==fill(2.0, 2, 2)
    close(clone)
    @test_throws ArgumentError FaerTeam(0)
    @test_throws ArgumentError FaerTeam(-1)
    holder=Ref{Any}()
    @test_throws ErrorException FaerTeam(2) do t
        holder[]=t
        error("exercise cleanup")
    end
    @test !isopen(holder[])
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
@testset "Team blocked products, promotion and views" begin
    FaerTeam(2) do team
        A=FaerMatrix(ones(512, 512))
        @test Matrix(team_mul(team, A, A))==fill(512.0, 512, 512)
        a=FaerMatrix(Float32[1 2; 3 4])
        b=FaerMatrix(ComplexF64[1 2im; 3 4])
        @test eltype(team_mul(team, a, b))==ComplexF64
        @test Matrix(team_mul(team, a, b))≈Matrix(a)*Matrix(b)
        big=FaerMatrix(zeros(ComplexF64, 4, 4))
        dest=view(big, 4:-2:2, 1:2)
        @test mul!(dest, team, a, b)===dest
        @test Matrix(dest)≈Matrix(a)*Matrix(b)
        @test Matrix(team_mul(team, transpose(a), b))≈transpose(Matrix(a))*Matrix(b)
        a[1, 1]=NaN
        @test Matrix(mul!(dest, team, a, b, 0, 0))==zeros(ComplexF64, 2, 2)
        closed=copy(b)
        close(closed)
        @test_throws ArgumentError mul!(dest, team, a, closed, 0, 0)
    end
end
