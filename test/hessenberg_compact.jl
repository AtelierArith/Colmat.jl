using Colmat, Test, LinearAlgebra, Random

@testset "Compact Hessenberg reconstruction and implicit actions" begin
    rng=MersenneTwister(8030)
    for T in (Float32, Float64, ComplexF32, ComplexF64), n in (0, 1, 2, 4, 17, 65)
        R=real(T)
        tol=R(1000)*max(n, 1)*eps(R)
        a=randn(rng, T, n, n)
        source=FaerMatrix(a)
        f=hessenberg(source; compact = true)
        @test f isa FaerCompactHessenberg{T}
        @test f.Q isa FaerHessenbergQ{T}
        @test Matrix(source)==a
        close(source)
        q=f.Q
        dense=Matrix(q)
        @test Matrix(f) ≈ a atol=tol rtol=tol
        @test Matrix{ComplexF64}(f) ≈ ComplexF64.(a) atol=tol rtol=tol
        converted=FaerMatrix{ComplexF64}(f)
        @test Matrix(converted) ≈ ComplexF64.(a) atol=tol rtol=tol
        close(converted)
        reconstructed=reconstruct(f)
        hpart, qpart=f
        @test hpart===f.H && qpart===f.Q
        @test istriu(Matrix(f.H), -1)
        @test dense'dense ≈ Matrix{T}(I, n, n) atol=tol rtol=tol
        @test q.phases==ones(T, n)
        @test q.nreflections==max(n-1, 0)
        basis, factor=q.basis, q.factor
        sequence=FaerHouseholderSequence(basis, factor)
        rebuilt=Matrix{T}(I, n, n)
        n>1 && (rebuilt[2:end, 2:end]=Matrix(sequence))
        @test rebuilt ≈ dense atol=tol rtol=tol
        fill!(basis, zero(T))
        fill!(factor, zero(T))
        @test Matrix(q) ≈ dense atol=tol rtol=tol
        rhs=randn(rng, T, n, 3)
        lhs=randn(rng, T, 3, n)
        for (oriented, reference) in (
            (q, dense),
            (q', dense'),
            (transpose(q), transpose(dense)),
            (conj(q), conj(dense)),
        )
            @test oriented isa FaerHessenbergQ
            @test Matrix(oriented*rhs) ≈ reference*rhs atol=tol rtol=tol
            @test Matrix(lhs*oriented) ≈ lhs*reference atol=tol rtol=tol
            @test Matrix(oriented\rhs) ≈ reference'rhs atol=tol rtol=tol
            @test Matrix(lhs/oriented) ≈ lhs*reference' atol=tol rtol=tol
            dest=copy(rhs)
            mul!(dest, oriented, dest, T(2), T(-1))
            @test dest ≈ T(2)*reference*rhs-rhs atol=tol rtol=tol
        end
        vector=randn(rng, T, n)
        @test q*vector ≈ dense*vector atol=tol rtol=tol
        promoted=ComplexF64.(rhs)
        @test Matrix(q*promoted) ≈ ComplexF64.(dense)*promoted atol=tol rtol=tol
        @test typeof(deepcopy(q))==typeof(q)
        @test typeof(deepcopy(f))==typeof(f)
        @test Matrix(copy(f)) ≈ a atol=tol rtol=tol
        snapshot=q'
        independent=copy(q)
        close(f)
        @test Matrix(reconstructed) ≈ a atol=tol rtol=tol
        close(reconstructed)
        @test !isopen(q)
        @test Matrix(snapshot) ≈ dense' atol=tol rtol=tol
        @test Matrix(independent) ≈ dense atol=tol rtol=tol
        @test_throws ArgumentError Matrix(q)
        close(snapshot)
        close(independent)
        close(sequence)
    end
end

@testset "Compact Hessenberg saved Schur and staged sharing" begin
    rng=MersenneTwister(8031)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        n in (0, 1, 2, 9, 33),
        use_team in (false, true)

        tol=real(T)(2000)*max(n, 1)*eps(real(T))
        a=randn(rng, T, n, n)
        source=FaerMatrix(a)
        team=use_team ? FaerTeam(2) : nothing
        work=FaerScratch(hessenberg, source; compact = true, team)
        f=hessenberg(source; compact = true, scratch = work, team)
        cap=scratch_capacity(work)
        @test Matrix(source)==a
        @test scratch_requirements(hessenberg!, source; compact = true, team) ==
              scratch_requirements(hessenberg, source; compact = true, team)
        held=f.Q
        q=Matrix(held)
        band=Matrix(f.H)
        reserve!(work, schur, f; team)
        cap=scratch_capacity(work)
        result=schur(f; scratch = work, team)
        @test Matrix(result.Z)*Matrix(result.T)*Matrix(result.Z)' ≈ a atol=tol rtol=tol
        @test Matrix(result.Z)'*Matrix(result.Z) ≈ Matrix{T}(I, n, n) atol=tol rtol=tol
        @test Matrix(f.H)==band && Matrix(f.Q)==q
        @test scratch_capacity(work)==cap
        @test scratch_requirements(schur!, f; team)==scratch_requirements(schur, f; team)
        hview=view(f.H, :, :)
        staged=schur!(f; scratch = work, team)
        @test staged.T===f.H && staged.Z===f.Q
        @test f.Q isa FaerMatrix{T}
        @test Matrix(hview)==Matrix(staged.T)
        @test Matrix(f) ≈ a atol=tol rtol=tol
        @test Matrix(held)==q
        @test typeof(deepcopy(f))==typeof(f)
        reserve!(work, schur!, f; team)
        old_q=f.Q
        repeated=schur!(f; scratch = work, team)
        @test repeated.T===f.H && repeated.Z===old_q
        @test Matrix(f) ≈ a atol=tol rtol=tol
        before_alias=Matrix(f.H)
        f.Q=f.H
        @test_throws ArgumentError schur!(f; scratch = work, team)
        @test Matrix(f.H)==before_alias
        f.Q=old_q
        # A bang reduction owns the packed reflectors separately from H=A.
        reserve!(work, hessenberg!, source; compact = true, team)
        bang=hessenberg!(source; compact = true, scratch = work, team)
        @test bang.H===source
        @test Matrix(bang) ≈ a atol=tol rtol=tol
        close(result)
        close(f)
        @test isopen(held)
        close(held)
        close(bang)
        close(work)
        isnothing(team)||close(team)
    end
end

@testset "Compact Hessenberg resources and transactional failures" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        a=T[4 1 2 0; 2 3 0 1; 1 0 5 0; 0 2 0 6]
        source=FaerMatrix(a)
        team=FaerTeam(2)
        req=scratch_requirements(hessenberg, source; compact = true, team)
        tiny=FaerScratch(0)
        if req.bytes>0
            for op in (hessenberg, hessenberg!)
                @test_throws ArgumentError op(source; compact = true, scratch = tiny, team)
                @test Matrix(source)==a
            end
        end
        reserve!(tiny, hessenberg, source; compact = true, team)
        if req.alignment>1
            unaligned=FaerScratch(req.bytes; alignment = 1)
            @test_throws ArgumentError hessenberg!(
                source;
                compact = true,
                scratch = unaligned,
                team,
            )
            @test Matrix(source)==a
            close(unaligned)
        end
        f=hessenberg(source; compact = true, scratch = tiny, team)
        q=Matrix(f.Q)
        band=Matrix(f.H)
        short=FaerScratch(0)
        for op in (schur, schur!)
            @test_throws ArgumentError op(f; scratch = short, team)
            @test f.Q isa FaerHessenbergQ
            @test Matrix(f.H)==band && Matrix(f.Q)==q
        end
        reserve!(tiny, schur, f; team)
        f.H[4, 1]=one(T)
        invalid=Matrix(f.H)
        for op in (schur, schur!)
            @test_throws Colmat.RustCall.RustError op(f; scratch = tiny, team)
            @test Matrix(f.H)==invalid && Matrix(f.Q)==q
        end
        f.H[4, 1]=zero(T)
        f.H[1, 1]=T(NaN)
        invalid=Matrix(f.H)
        @test_throws Colmat.RustCall.RustError schur!(f; scratch = tiny, team)
        @test isequal(Matrix(f.H), invalid)
        @test Matrix(f.Q)==q
        for field in (:n, :k, :blocksize)
            old=getfield(f.Q, field)
            setfield!(f.Q, field, old+1)
            @test_throws Exception scratch_requirements(schur, f; team)
            @test_throws Exception f.Q.phases
            setfield!(f.Q, field, old)
        end
        old_dims=f.H.dims
        f.H.dims=(5, 5)
        @test_throws r"current native dimensions" schur(f)
        f.H.dims=old_dims
        source[1, 1]=T(NaN)
        invalid_input=Matrix(source)
        for op in (hessenberg, hessenberg!)
            @test_throws Colmat.RustCall.RustError op(
                source;
                compact = true,
                scratch = tiny,
                team,
            )
            @test isequal(Matrix(source), invalid_input)
        end
        overflowing=FaerMatrix(fill(T(floatmax(real(T))), 4, 4))
        original_overflowing=Matrix(overflowing)
        @test_throws Colmat.RustCall.RustError hessenberg!(
            overflowing;
            compact = true,
            scratch = tiny,
            team,
        )
        @test Matrix(overflowing)==original_overflowing
        close(overflowing)
        close(team)
        @test_throws ArgumentError scratch_requirements(schur, f; team)
        @test_throws ArgumentError hessenberg(source; compact = true, team)
        close(tiny)
        @test_throws ArgumentError hessenberg(source; compact = true, scratch = tiny)
        close(short)
        close(f.Q)
        @test_throws ArgumentError schur(f)
        close(f)
        close(source)
    end
end

@testset "Compact Hessenberg views, scaling and legacy contract" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        R=real(T)
        base=T[4 1 2 0; 2 3 0 1; 1 0 5 0; 0 2 0 6]
        for exponent in (
            0,
            T<:Union{Float32,ComplexF32} ? 100 : 900,
            T<:Union{Float32,ComplexF32} ? -100 : -900,
        )
            scale=ldexp(one(R), exponent)
            source=FaerMatrix(base*scale)
            f=hessenberg(source; compact = true)
            @test Matrix(f) ./ scale ≈ base rtol=500eps(R) atol=500eps(R)
            @test all(isfinite, Matrix(f.H))
            close(f)
            close(source)
        end
        source=FaerMatrix(base)
        @test hessenberg(source) isa FaerHessenberg{T,typeof(source)}
        for v in
            (view(source, :, :), view(source, 4:-1:1, 4:-1:1), view(source, 1:2:4, 1:2:4))
            expected=Matrix(v)
            work=FaerScratch(hessenberg, v; compact = true)
            for task in
                [Threads.@spawn hessenberg(v; compact = true, scratch = work) for _ = 1:2]
                f=fetch(task)
                @test Matrix(f) ≈ expected rtol=1000eps(R) atol=1000eps(R)
                close(f)
            end
            close(work)
        end
        close(source)
    end
    @test isempty(detect_ambiguities(Colmat, Base, LinearAlgebra; recursive = true))
end
