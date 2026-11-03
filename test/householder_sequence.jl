using Colmat, Test, LinearAlgebra, SparseArrays, Random

function sequence_reference(v, tau)
    T=eltype(v)
    m, k=size(v)
    result=Matrix{T}(I, m, m)
    for j = 1:k
        column=v[:, j]
        result=result*(Matrix{T}(I, m, m)-column*column'/tau[j])
    end
    return result
end
@testset "Native compact block and sequence products" begin
    rng=MersenneTwister(914)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        (m, k) in ((0, 0), (4, 0), (1, 1), (4, 4), (6, 3), (11, 7)),
        b in (1, 2, 3, k+2),
        identity in (false, true)

        v=zeros(T, m, k)
        tau=zeros(real(T), k)
        for j = 1:k
            v[j, j]=1
            v[(j+1):end, j]=randn(rng, T, m-j)
            tau[j]=sum(abs2, v[:, j])/2
            if identity && isodd(j)
                tau[j]=Inf
                v[(j+1):end, j].=0
            end
        end
        reference=sequence_reference(v, tau)
        q=FaerHouseholderSequence(v, tau; blocksize = b)
        @test q isa FaerHouseholderSequence{T}
        @test size(q)==(m, m)
        @test q.nreflections==k && q.blocksize==b
        @test Matrix(q.basis)==v
        @test q.tau==tau
        @test size(q.factor)==(b, k)
        @test Matrix(q) ≈ reference
        @test reference'*reference ≈ Matrix{T}(I, m, m)
        reconstructed=FaerHouseholderSequence(q.basis, q.factor)
        @test Matrix(reconstructed) ≈ reference
        @test Matrix(inv(q)) ≈ reference'
        @test Matrix(q*q') ≈ Matrix{T}(I, m, m)
        @test hasproperty(q, :basis) && hasproperty(q, :factor)
        a=randn(rng, T, m, 3)
        r=randn(rng, T, 4, m)
        for (op, expected) in (
            (q, reference),
            (q', reference'),
            (transpose(q), transpose(reference)),
            (conj(q), conj(reference)),
        )
            @test Matrix(op) ≈ expected
            @test Matrix(op*FaerMatrix(a)) ≈ expected*a
            @test Matrix(FaerMatrix(r)*op) ≈ r*expected
            @test op*a[:, 1] ≈ expected*a[:, 1]
            left=FaerMatrix(a)
            @test lmul!(op, left)===left
            @test Matrix(left) ≈ expected*a
            right=FaerMatrix(r)
            @test rmul!(right, op)===right
            @test Matrix(right) ≈ r*expected
            @test Matrix(op\a) ≈ expected'*a
        end
        for width in (1, max(1, k), k+3)
            changed=reblock(q, width)
            @test changed.blocksize==width
            @test Matrix(changed) ≈ reference
            @test Matrix(reblock(q', width)) ≈ reference'
        end
        exported=q.factor
        @test reblock!(q, b+1)===q
        @test q.blocksize==b+1
        @test Matrix(q) ≈ reference
        @test size(exported)==(b, k)
        copied=copy(q)
        deep=deepcopy(q)
        adj=q'
        close(q)
        @test !isopen(q)
        @test Matrix(copied) ≈ reference
        @test Matrix(deep) ≈ reference
        @test Matrix(adj) ≈ reference'
        @test_throws ArgumentError q.basis
        @test_throws ArgumentError q.factor
        @test_throws ArgumentError q.tau
        @test_throws ArgumentError q*a
    end
end
@testset "Sequence ignored storage, validation and snapshots" begin
    v=[1.0 0 0; 0.2 1 0; 0.3 0.4 1; 0.1 -0.2 0.3]
    tau=[sum(abs2, v[:, j])/2 for j = 1:3]
    q=FaerHouseholderSequence(v, tau; blocksize = 2)
    reference=Matrix(q)
    ignored=copy(v)
    for j = 1:3, i = 1:j
        ignored[i, j]=NaN
    end
    @test Matrix(FaerHouseholderSequence(ignored, tau; blocksize = 2)) ≈ reference
    factor=Matrix(q.factor)
    for j = 1:3, i = (mod1(j, 2)+1):2
        factor[i, j]=NaN
    end
    @test Matrix(FaerHouseholderSequence(v, factor)) ≈ reference
    @test_throws ArgumentError FaerHouseholderSequence(v, tau; blocksize = 0)
    @test_throws DimensionMismatch FaerHouseholderSequence(v, tau[1:2])
    @test_throws DimensionMismatch FaerHouseholderSequence(ones(2, 3), ones(3))
    for t in (-1.0, 0.0, NaN, 0.01)
        bad=copy(tau)
        bad[1]=t
        @test_throws ArgumentError FaerHouseholderSequence(v, bad)
    end
    @test_throws ArgumentError FaerHouseholderSequence(v, ComplexF64[1+im, 1, 1])
    bad=Matrix(q.factor)
    bad[1, 2]+=10
    @test_throws ArgumentError FaerHouseholderSequence(v, bad)
    before=Matrix(q)
    @test_throws ArgumentError reblock!(q, 0)
    @test Matrix(q)==before
    basis=q.basis
    basis[4, 1]=99
    factor=q.factor
    factor[1, 1]=99
    @test Matrix(q)==before
    @test_throws BoundsError q[0, 1]
    @test q[2, 3]≈reference[2, 3]
    @test_throws DimensionMismatch q*ones(3)
    close(q)
    @test_throws ArgumentError reblock(q, 2)
end
@testset "Sequence promotion, views and mul!" begin
    rng=MersenneTwister(396)
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        v=zeros(T, 6, 3)
        for j = 1:3
            v[j, j]=1
            v[(j+1):end, j]=randn(rng, T, 6-j)
        end
        tau=[sum(abs2, v[:, j])/2 for j = 1:3]
        q=FaerHouseholderSequence(v, tau; blocksize = 2)
        m=Matrix(q)
        a=randn(rng, T, 6, 4)
        dest=FaerMatrix(a)
        @test mul!(dest, q, dest, 2, 3)===dest
        @test Matrix(dest) ≈ 2m*a+3a
        right=FaerMatrix(a')
        @test mul!(right, right, q, 2, 3)===right
        @test Matrix(right) ≈ 2a'*m+3a'
        vector=FaerVector(a[:, 1])
        @test mul!(vector, q, vector)===vector
        @test Vector(vector) ≈ m*a[:, 1]
        parent=FaerMatrix(randn(rng, T, 12, 8))
        viewmatrix=view(parent, 12:-2:2, 8:-2:2)
        expected=m*Matrix(viewmatrix)
        @test lmul!(q, viewmatrix)===viewmatrix
        @test Matrix(viewmatrix) ≈ expected
        for S in (Float32, Float64, ComplexF32, ComplexF64)
            P=promote_type(T, S)
            x=randn(rng, S, 6, 2)
            reference=sequence_reference(P.(v), tau)
            @test eltype(q*FaerMatrix(x))===P
            @test Matrix(q*FaerMatrix(x)) ≈ reference*P.(x)
            @test Matrix(FaerMatrix(x')*q') ≈ P.(x')*reference'
        end
    end
    @test isempty(
        detect_ambiguities(Colmat, Base, LinearAlgebra, SparseArrays; recursive = true),
    )
end
@testset "Sequence application avoids dense coefficient storage" begin
    n=100_000
    k=3
    v=zeros(n, k)
    for j = 1:k
        v[j, j]=1
        v[(j+1):end, j].=1/sqrt(n)
    end
    tau=[sum(abs2, v[:, j])/2 for j = 1:k]
    q=FaerHouseholderSequence(v, tau; blocksize = 2)
    x=FaerVector(ones(n))
    result=q*x
    @test result isa FaerVector
    @test norm(Vector(result))≈sqrt(n)
    @test Vector(q' * result)≈ones(n)
    @test size(q.factor)==(2, k)
end

@testset "Unitary operator left/right solves" begin
    rng=MersenneTwister(91)
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        v=T[1 0; 0.2 1; 0.3 0.4; 0.1 -0.2]
        tau=[sum(abs2, v[:, j])/2 for j = 1:2]
        for q in
            (FaerHouseholderSequence(v, tau; blocksize = 2), householder(randn(rng, T, 4)))
            m=Matrix(q)
            a=randn(rng, T, 4, 3)
            b=randn(rng, T, 3, 4)
            target=FaerMatrix(a)
            @test ldiv!(q, target)===target
            @test Matrix(target) ≈ m'*a
            target=FaerMatrix(a)
            @test ldiv!(target, q, target)===target
            @test Matrix(target) ≈ m'*a
            target=FaerMatrix(b)
            @test rdiv!(target, q)===target
            @test Matrix(target) ≈ b*m'
            @test Matrix(FaerMatrix(b)/q) ≈ b*m'
            @test Matrix(b/q) ≈ b*m'
            target=FaerVector(a[:, 1])
            @test ldiv!(target, q, target)===target
            @test Vector(target) ≈ m'*a[:, 1]
        end
    end
    @test isempty(
        detect_ambiguities(Colmat, Base, LinearAlgebra, SparseArrays; recursive = true),
    )
end
