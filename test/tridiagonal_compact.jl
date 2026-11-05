using Test, LinearAlgebra, Random
using Colmat

@testset "Compact tridiagonal Q reconstruction and actions" begin
    rng=MersenneTwister(8022)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        n in (0, 1, 2, 4, 9),
        bs in unique([0, 1, min(3, max(1, n-1))])

        R=real(T)
        tol=R(500)*eps(R)
        x=randn(rng, T, n, n)
        a=x+x'
        f=tridiagonalize(a; compact = true, blocksize = bs)
        q=f.Q
        dense=Matrix(q)
        @test q isa FaerTridiagonalQ{T}
        @test f isa FaerTridiagonalization{T}
        @test size(f)==(n, n)
        @test dense*Matrix(f.H)*dense' ≈ a atol=tol rtol=tol
        @test dense'dense ≈ Matrix{T}(I, n, n) atol=tol rtol=tol
        @test Matrix(f) ≈ a atol=tol rtol=tol
        basis=q.basis
        factor=q.factor
        phases=q.phases
        seq=FaerHouseholderSequence(basis, factor)
        rebuilt=Matrix{T}(I, n, n)
        n>1 && (rebuilt[2:end, 2:end]=Matrix(seq))
        @test rebuilt*Diagonal(phases) ≈ dense atol=tol rtol=tol
        @test size(basis)==(max(0, n-1), max(0, n-1))
        @test size(factor)==(q.blocksize, max(0, n-1))
        @test q.nreflections==max(0, n-1)
        @test all(isfinite, phases)
        fill!(basis, zero(T))
        fill!(factor, zero(T))
        fill!(phases, zero(T))
        @test Matrix(q) ≈ dense atol=tol rtol=tol
        rhs=randn(rng, T, n, 3)
        left=randn(rng, T, 3, n)
        for (h, d) in (
            (q, dense),
            (transpose(q), transpose(dense)),
            (q', dense'),
            (conj(q), conj(dense)),
        )
            @test Matrix(h*rhs) ≈ d*rhs atol=tol rtol=tol
            @test Matrix(left*h) ≈ left*d atol=tol rtol=tol
            @test Matrix(h\rhs) ≈ d'rhs atol=tol rtol=tol
            @test Matrix(left/h) ≈ left*d' atol=tol rtol=tol
            dest=copy(rhs)
            lmul!(h, dest)
            @test dest ≈ d*rhs atol=tol rtol=tol
            dest=copy(left)
            rmul!(dest, h)
            @test dest ≈ left*d atol=tol rtol=tol
            dest=copy(rhs)
            mul!(dest, h, dest, T(2), T(-1))
            @test dest ≈ T(2)*d*rhs-rhs atol=tol rtol=tol
        end
        v=randn(rng, T, n)
        @test q*v ≈ dense*v atol=tol rtol=tol
        owned=FaerVector(v)
        lmul!(q, owned)
        @test Vector(owned) ≈ dense*v atol=tol rtol=tol
        row=FaerRow(v)
        rmul!(row, q)
        @test Matrix(row) ≈ reshape(v, 1, :)*dense atol=tol rtol=tol
        close(owned)
        close(row)
        # Promotion converts the reflector data, rather than a materialized Q.
        promoted=ComplexF64.(rhs)
        @test Matrix(q*promoted) ≈ ComplexF64.(dense)*promoted atol=max(tol, 1e-12) rtol=max(
            tol,
            1e-12,
        )
        @test Matrix(copy(q)) ≈ dense atol=tol rtol=tol
        @test typeof(deepcopy(q))==typeof(q)
        @test Matrix(deepcopy(f)) ≈ a atol=tol rtol=tol
        e=eigen(f)
        @test a*Matrix(e.vectors) ≈ Matrix(e.vectors)*Diagonal(e.values) atol=tol rtol=tol
        @test eigvals(f) ≈ eigvals(Hermitian(a)) atol=tol rtol=tol
        held=q'
        independent=copy(q)
        close(f)
        @test !isopen(q)
        @test isopen(held)
        @test Matrix(held) ≈ dense' atol=tol rtol=tol
        @test Matrix(independent) ≈ dense atol=tol rtol=tol
        @test_throws ArgumentError Matrix(q)
        close(q)
        close(held)
        close(independent)
        close(seq)
    end
end

@testset "Compact tridiagonal resources and ownership" begin
    rng=MersenneTwister(8023)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        policy in (FaerParThreshold(), FaerParThreshold(0), FaerParThreshold(:never))

        R=real(T)
        tol=R(1000)*eps(R)
        x=randn(rng, T, 8, 8)
        a=x+x'
        team=FaerTeam(2)
        req=scratch_requirements(
            tridiagonalize,
            a;
            compact = true,
            blocksize = 3,
            team,
            par_threshold = policy,
        )
        scratch=FaerScratch(req.bytes; alignment = req.alignment)
        source=FaerMatrix(a)
        f=tridiagonalize(
            source;
            compact = true,
            blocksize = 3,
            scratch,
            team,
            par_threshold = policy,
        )
        close(source)
        GC.gc()
        @test Matrix(f) ≈ a atol=tol rtol=tol
        @test scratch_capacity(scratch)==(bytes = req.bytes, alignment = req.alignment)
        q=f.Q
        dense=Matrix(q)
        rhs=randn(rng, T, 8, 5)
        backing=zeros(T, 10, 7)
        v=view(backing, 2:9, 2:6)
        v.=rhs
        reserve!(scratch, lmul!, q, v; team)
        before=scratch_capacity(scratch)
        lmul!(q, v; scratch, team)
        @test v ≈ dense*rhs atol=tol rtol=tol
        @test scratch_capacity(scratch)==before
        @test all(iszero, backing[1, :]) && all(iszero, backing[end, :])
        @test all(iszero, backing[:, 1]) && all(iszero, backing[:, end])
        reserve!(scratch, eigen, f; team)
        before=scratch_capacity(scratch)
        e=eigen(f; scratch, team, sortby = x->-x)
        @test issorted(e.values; rev = true)
        @test a*Matrix(e.vectors) ≈ Matrix(e.vectors)*Diagonal(e.values) atol=tol rtol=tol
        @test scratch_capacity(scratch)==before
        tiny=FaerScratch(0)
        applyreq=scratch_requirements(lmul!, q, rhs; team)
        if applyreq.bytes>0
            dest=copy(rhs)
            @test_throws Exception lmul!(q, dest; scratch = tiny, team)
            @test dest==rhs
        end
        if req.bytes>0
            @test_throws Exception tridiagonalize(
                a;
                compact = true,
                blocksize = 3,
                scratch = tiny,
                team,
                par_threshold = policy,
            )
        end
        # Forged cached metadata must fail against the actual native owner.
        for field in (:n, :k, :blocksize)
            old=getfield(q, field)
            setfield!(q, field, old+1)
            @test_throws Exception q.phases
            @test_throws Exception scratch_requirements(eigen, f; team)
            setfield!(q, field, old)
        end
        wrong=tridiagonalize(zeros(R, 8, 8); compact = true).Q
        forged=FaerTridiagonalQ{T,typeof(Colmat._handle(wrong))}(
            Colmat._handle(wrong),
            q.n,
            q.k,
            wrong.blocksize,
            false,
            false,
        )
        if T!==R
            @test_throws Exception Matrix(forged)
        end
        close(team)
        dest=copy(rhs)
        @test_throws ArgumentError lmul!(q, dest; team)
        @test dest==rhs
        @test_throws ArgumentError tridiagonalize(
            a;
            compact = true,
            team,
            par_threshold = FaerParThreshold(:never),
        )
        close(tiny)
        close(scratch)
        close(f)
        close(wrong)
        close(forged)
    end
end

@testset "Compact tridiagonal staged aliases and tiny complex phases" begin
    for T in (ComplexF32, ComplexF64)
        R=real(T)
        small=T===ComplexF32 ? R(1e-30) : R(1e-300)
        a=T[1 -small*im; small*im 2]
        f=tridiagonalize(a; compact = true)
        q=Matrix(f.Q)
        r=q*Matrix(f.H)*q'
        @test abs(imag(r[2, 1])/small-one(R))<=R(100)*eps(R)
        @test abs(real(r[2, 1])/small)<=R(100)*eps(R)
        @test Matrix(f.Q'*f.Q) ≈ Matrix{T}(I, 2, 2) atol=R(100)*eps(R)
        source=FaerMatrix(a)
        staged=tridiagonalize!(source; compact = true)
        @test Matrix(source)==Matrix(staged.H)
        @test Matrix(staged) ≈ a
        wrapper=Hermitian(FaerMatrix(a), :L)
        alias=hessenberg!(wrapper; compact = true)
        @test alias.Q isa FaerTridiagonalQ
        @test Matrix(alias) ≈ a
        close(f)
        close(staged)
        close(source)
        close(alias)
        close(parent(wrapper))
    end
end

@testset "Compact Q casts only reflector data" begin
    a=Float64[1e300 2e299 0; 2e299 3e300 1e299; 0 1e299 2e300]
    f=tridiagonalize(a; compact = true)
    rhs=Float32[1 2; 3 4; 5 6]
    expected=Float32.(Matrix(f.Q))*rhs
    actual=copy(rhs)
    lmul!(f.Q, actual)
    @test actual ≈ expected rtol=100eps(Float32)
    close(f)
end
