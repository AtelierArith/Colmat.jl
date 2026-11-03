using Test, LinearAlgebra, Random, Colmat

@testset "Native Hessenberg and Schur factorizations" begin
    rng=MersenneTwister(491)
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        tol=typeof(real(zero(T)))===Float32 ? 8e-4 : 3e-11
        for n in (0, 1, 2, 5, 12)
            data=randn(rng, T, n, n)
            a=FaerMatrix(data)
            h=hessenberg(a)
            @test h isa FaerHessenberg
            @test eltype(h.H)===T
            @test size(h)==(n, n)
            @test Matrix(h.H)≈triu(Matrix(h.H), -1) atol=tol
            @test Matrix(h.Q)'*Matrix(h.Q)≈Matrix{T}(I, n, n) rtol=tol atol=tol
            @test Matrix(h)≈data rtol=tol atol=tol
            @test Matrix(copy(h))≈data rtol=tol atol=tol
            @test Matrix(deepcopy(h))≈data rtol=tol atol=tol
            @test Matrix(a)==data
            f=schur(a)
            @test f isa Schur
            @test f.T isa FaerMatrix{T}
            @test f.Z isa FaerMatrix{T}
            @test size(f)==(n, n)
            @test Matrix(f)≈data rtol=tol atol=tol
            @test Matrix(f.Z)'*Matrix(f.Z)≈Matrix{T}(I, n, n) rtol=tol atol=tol
            @test Matrix(f.T)≈triu(Matrix(f.T), T<:Real ? -1 : 0) atol=tol
            @test Matrix(copy(f))≈data rtol=tol atol=tol
            @test Matrix(deepcopy(f))≈data rtol=tol atol=tol
            @test Matrix(a)==data
            @test Matrix(schur(h))≈data rtol=tol atol=tol
            @test sort(f.values; by = x->(real(x), imag(x)))≈sort(
                eigvals(data);
                by = x->(real(x), imag(x)),
            ) rtol=tol atol=tol
            vals=eigvals(f)
            @test vals==f.values
            n>0 && (vals[1] = 0; @test vals!==f.values)
            vectors=eigvecs(f)
            @test vectors isa FaerMatrix
            @test data*Matrix(vectors)≈Matrix(vectors)*Diagonal(f.values) rtol=tol atol=tol
            for j = 1:n
                @test norm(Matrix(vectors)[:, j])≈1 rtol=tol
            end
            e=eigen(f)
            @test data*Matrix(e.vectors)≈Matrix(e.vectors)*Diagonal(e.values) rtol=tol atol=tol
            for select in (falses(n), trues(n), real.(f.values) .> 0)
                originalT=Matrix(f.T)
                originalZ=Matrix(f.Z)
                originalvalues=copy(f.values)
                g=ordschur(f, select)
                @test Matrix(f.T)==originalT
                @test Matrix(f.Z)==originalZ
                @test f.values==originalvalues
                @test Matrix(g)≈data rtol=tol atol=tol
                k=count(select)
                @test sort(g.values[1:k]; by = x->(real(x), imag(x)))≈sort(
                    f.values[select];
                    by = x->(real(x), imag(x)),
                ) rtol=tol atol=tol
                @test Matrix(g.Z)'*Matrix(g.Z)≈Matrix{T}(I, n, n) rtol=tol atol=tol
                dest=copy(f)
                held=view(dest.T, :, :)
                @test ordschur!(dest, select)===dest
                @test Matrix(dest)≈data rtol=tol atol=tol
                @test Matrix(held)==Matrix(dest.T)
            end
            b=copy(a)
            held=view(b, :, :)
            bf=schur!(b)
            @test bf.T===b
            @test Matrix(bf)≈data rtol=tol atol=tol
            @test Matrix(held)==Matrix(b)
            @test_throws DimensionMismatch ordschur(f, trues(n+1))
            closedcopy=copy(f)
            close(f)
            @test !isopen(f)
            @test isopen(closedcopy)
            @test_throws ArgumentError eigvals(f)
            @test_throws ArgumentError eigvecs(f)
            @test_throws ArgumentError ordschur(f, falses(n))
            close(h)
            @test !isopen(h)
            @test_throws ArgumentError Matrix(h)
            close(a)
            @test_throws ArgumentError schur(a)
        end
        a=FaerMatrix(randn(rng, T, 3, 4))
        @test_throws DimensionMismatch schur(a)
        @test_throws DimensionMismatch hessenberg(a)
        a=FaerMatrix{T}(undef, 2, 2)
        a[1, 1]=T(NaN)
        @test_throws Exception schur(a)
        @test_throws Exception hessenberg(a)
    end
end

@testset "Schur block selection and view inputs" begin
    for T in (Float32, Float64)
        data=T[1 2 0 0; -2 1 0 0; 0 0 -3 0; 0 0 0 4]
        a=FaerMatrix(data)
        f=schur(a)
        @test any(!iszero, imag.(f.values))
        @test_throws ArgumentError ordschur(f, Bool[true, false, false, false])
        t, z, vals=Matrix(f.T), Matrix(f.Z), copy(f.values)
        @test_throws ArgumentError ordschur!(f, Bool[false, true, true, false])
        @test Matrix(f.T)==t && Matrix(f.Z)==z && f.values==vals
        g=ordschur(f, Bool[false, false, true, false])
        @test real(g.values[1])≈-3
        @test Matrix(g)≈data
        g=ordschur(f, Bool[true, true, false, true])
        @test count(x->real(x)>0, g.values[1:3])==3
        @test Matrix(g)≈data
        v=view(a, 4:-1:1, 4:-1:1)
        @test Matrix(schur(v))≈reverse(data; dims = (1, 2))
        @test Matrix(hessenberg(v))≈reverse(data; dims = (1, 2))
    end
    @test isempty(detect_ambiguities(Colmat, Base, LinearAlgebra; recursive = true))
end

@testset "Reusing and mutating native Hessenberg storage" begin
    rng=MersenneTwister(523)
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        tol=typeof(real(zero(T)))===Float32 ? 8e-4 : 3e-11
        data=randn(rng, T, 7, 7)
        a=FaerMatrix(data)
        aview=view(a, :, :)
        h=hessenberg!(a)
        @test h.H===a
        @test Matrix(aview)==Matrix(h.H)
        @test Matrix(h)≈data rtol=tol atol=tol
        originalh, originalq=Matrix(h.H), Matrix(h.Q)
        s=schur(h)
        @test Matrix(h.H)==originalh && Matrix(h.Q)==originalq
        @test Matrix(s)≈data rtol=tol atol=tol
        qview=view(h.Q, :, :)
        s=schur!(h)
        @test s.T===h.H && s.Z===h.Q
        @test Matrix(aview)==Matrix(s.T) && Matrix(qview)==Matrix(s.Z)
        @test Matrix(s)≈data rtol=tol atol=tol
        broken=hessenberg(FaerMatrix(data))
        broken.H[4, 1]=T(1)
        bh, bq=Matrix(broken.H), Matrix(broken.Q)
        @test_throws Exception schur!(broken)
        @test Matrix(broken.H)==bh && Matrix(broken.Q)==bq
    end
end

@testset "Scaled and blocked Schur reductions" begin
    template=[0.1 0.2 -0.3; 0.4 -0.2 0.1; -0.1 0.2 0.3]
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        R=typeof(real(zero(T)))
        tol=R===Float32 ? 8e-4 : 4e-11
        for scale in (R===Float32 ? (R(1e-30), R(1e30)) : (R(1e-300), R(1e300)))
            data=T.(template)
            T<:Complex && (data .+= im .* T.(transpose(template)))
            a=FaerMatrix(data .* scale)
            f=schur(a)
            h=hessenberg(a)
            @test Matrix(f) ./ scale≈data rtol=tol atol=tol
            @test Matrix(h) ./ scale≈data rtol=tol atol=tol
            @test all(isfinite, f.values)
            @test Matrix(f.Z)'*Matrix(f.Z)≈Matrix{T}(I, 3, 3) rtol=tol atol=tol
        end
    end
    rng=MersenneTwister(261)
    for T in (Float64, ComplexF64)
        data=randn(rng, T, 257, 257)
        h=hessenberg(FaerMatrix(data))
        @test Matrix(h)≈data rtol=2e-10 atol=2e-10
        f=schur(h)
        @test Matrix(f)≈data rtol=2e-10 atol=2e-10
        @test Matrix(f.Z)'*Matrix(f.Z)≈Matrix{T}(I, 257, 257) rtol=2e-10 atol=2e-10
    end
end

@testset "Schur constructors and repeated eigenvalues" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        tol=typeof(real(zero(T)))===Float32 ? 8e-4 : 3e-11
        for data in (zeros(T, 3, 3), Matrix{T}(I, 3, 3), T[1 1 0; 0 1 1; 0 0 1])
            f=schur(FaerMatrix(data))
            x=eigvecs(f)
            @test data*Matrix(x)≈Matrix(x)*Diagonal(f.values) rtol=tol atol=tol
        end
        t=FaerMatrix(Matrix(Diagonal(T[1, 2, 3])))
        z=FaerMatrix{T}(I, 3, 3)
        for values in
            (Float32[1, 2, 3], Float64[1, 2, 3], ComplexF32[1, 2, 3], ComplexF64[1, 2, 3])
            f=Schur(copy(t), copy(z), values)
            g=ordschur(f, Bool[false, false, true])
            @test real(g.values[1])≈3
            @test Matrix(g)≈Matrix(t)
            @test ordschur!(f, Bool[false, false, true])===f
            @test real(f.values[1])≈3
            @test Matrix(f)≈Matrix(t)
        end
        f=Schur(copy(t), copy(z), view(T[1, 2, 3], :))
        g=deepcopy(f)
        @test typeof(g)===typeof(f)
        @test g.values==f.values
        ordschur!(f, Bool[false, true, false])
        @test real(f.values[1])≈2
        @test real(g.values[1])≈1
        readonly=Schur(copy(t), copy(z), 1:3)
        @test_throws ArgumentError ordschur!(readonly, Bool[false, true, false])
        @test Matrix(readonly.T)==Matrix(t)
    end
end

@testset "Schur eigenvalue conversion errors preserve inputs" begin
    data=[1.0 2 0; -2 1 0; 0 0 4]
    f=Schur(FaerMatrix(data), FaerMatrix{Float64}(I, 3, 3), [1.0, 1, 4])
    originalt, originalz, originalv=Matrix(f.T), Matrix(f.Z), copy(f.values)
    @test_throws InexactError ordschur!(f, Bool[false, false, true])
    @test Matrix(f.T)==originalt && Matrix(f.Z)==originalz && f.values==originalv
    h=hessenberg(FaerMatrix(data))
    hh, qq=h
    @test hh===h.H && qq===h.Q
end
