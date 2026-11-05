using Colmat, LinearAlgebra, Test, Random
function hct_fixture(T, m, k; identity = false)
    v=zeros(T, m, k)
    tau=zeros(typeof(real(zero(T))), k)
    for j = 1:k
        v[j, j]=1
        for i = (j+1):m
            v[i, j]=T(0.02sin(2i+j))
            T<:Complex && (v[i, j]+=T(0.01im*cos(i+3j)))
        end
        tau[j]=sum(abs2, v[:, j])/2
        if identity && isodd(j)
            tau[j]=Inf
            v[(j+1):end, j].=0
        end
    end
    v, tau
end
function hct_apply(v, tau, B; reverse = false)
    out=copy(B)
    # Q=H1*...*Hk applies the rightmost reflection first.
    order=reverse ? (1:length(tau)) : (length(tau):-1:1)
    for j in order
        isinf(tau[j])&&continue
        w=view(v, :, j)
        out .-= w*((w'*out)/tau[j])
    end
    out
end
@testset "Householder construction teams and saved block upgrades" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        width in (0, 1, 2),
        (m, k) in ((0, 0), (4, 0), (1, 1), (7, 5), (96, 65), (1024, 32)),
        identity in (false, true)

        team=width==0 ? nothing : FaerTeam(width)
        v, tau=hct_fixture(T, m, k; identity)
        b=8
        q=FaerHouseholderSequence(v, tau; blocksize = b, team)
        B=T[sin(i+j) for i = 1:m, j = 1:2]
        gold=hct_apply(v, tau, B)
        tol=T in (Float32, ComplexF32) ? 2e-3 : 1e-10
        @test Matrix(q*B)≈gold atol=tol rtol=tol
        @test q.tau==tau
        factor=q.factor
        basis=q.basis
        reconstructed=FaerHouseholderSequence(
            basis,
            factor;
            blocksize = 16,
            prev_block_size = 8,
            team,
        )
        close(basis)
        close(factor)
        @test reconstructed.blocksize==16
        @test Matrix(reconstructed*B)≈gold atol=tol rtol=tol
        for transform in (Base.identity, adjoint, transpose, conj),
            newb in (1, 3, 8, 16, 24),
            prev in (nothing, 1, 8)

            !isnothing(prev) && (newb%prev!=0 || b%prev!=0) && continue
            src=transform(q)
            expected=if transform===adjoint
                hct_apply(v, tau, B; reverse = true)
            elseif transform===transpose
                conj.(hct_apply(v, tau, conj.(B); reverse = true))
            elseif transform===conj
                conj.(hct_apply(v, tau, conj.(B)))
            else
                gold
            end
            changed=reblock(src, newb; prev_block_size = prev, team)
            @test changed.blocksize==newb
            @test changed.transposed==src.transposed && changed.conjugated==src.conjugated
            @test Matrix(changed*B)≈expected atol=tol rtol=tol
            exported=changed.factor
            @test reblock!(changed, 8; team)===changed
            @test Matrix(changed*B)≈expected atol=tol rtol=tol
            @test size(exported)==(newb, k)
            close(exported)
            close(changed)
            src===q||close(src)
        end
        close(q)
        @test Matrix(reconstructed*B)≈gold atol=tol rtol=tol
        close(reconstructed)
        isnothing(team)||close(team)
    end
end
@testset "Householder construction validation, atomic reblock and shared teams" begin
    v, tau=hct_fixture(Float64, 20, 10)
    team=FaerTeam(2)
    q=FaerHouseholderSequence(v, tau; blocksize = 8, team)
    before=Matrix(q)
    handle=q.handle
    for prev in (0, -1, 3, 16)
        @test_throws ArgumentError reblock!(q, 16; prev_block_size = prev, team)
        @test q.handle===handle
        @test Matrix(q)==before
    end
    @test_throws ArgumentError FaerHouseholderSequence(v, tau; team = :bad)
    @test_throws ArgumentError reblock(q, 16; team = :bad)
    bad=FaerMatrix(v)
    bad.dims=(21, 10)
    @test_throws Colmat.RustCall.RustError FaerHouseholderSequence(bad, tau; team)
    bad.dims=size(v)
    close(bad)
    old=q.n
    q.n=21
    @test_throws Colmat.RustCall.RustError reblock!(q, 16; team)
    @test q.handle===handle
    q.n=old
    wrong=copy(q.factor)
    wrong[1, 2]+=0.1
    @test_throws ArgumentError FaerHouseholderSequence(v, wrong; team)
    close(wrong)
    jobs=[@async reblock(q, b; team) for b in (3, 8, 16)]
    for result in fetch.(jobs)
        @test Matrix(result)≈before
        close(result)
    end
    close(team)
    @test_throws ArgumentError FaerHouseholderSequence(v, tau; team)
    @test_throws ArgumentError reblock!(q, 16; team)
    @test q.handle===handle
    @test Matrix(q)==before
    close(q)
    @test_throws ArgumentError reblock(q, 16)
end
