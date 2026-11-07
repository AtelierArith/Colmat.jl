using Colmat, Test, LinearAlgebra, SparseArrays, Random

_bytes(f) = (f(); @allocated f())
# Same pattern, new values (stored zeros included).
function _revalue(S::SparseMatrixCSC{T}, rng) where {T}
    R=copy(S)
    R.nzval .= randn(rng, T, nnz(R))
    return R
end

@testset "Reusable sparse product plans" begin
    rng=MersenneTwister(114)
    for T in (Float32, Float64, ComplexF32, ComplexF64), format in (:csc, :csr)
        a=sprand(rng, T, 7, 5, 0.4)
        b=sprand(rng, T, 5, 6, 0.4)
        a.nzval[1]=zero(T) # Stored zeros are part of the planned pattern.
        A=FaerSparseMatrix(a; format)
        B=FaerSparseMatrix(b; format)
        plan=FaerSparseProductPlan(A, B)
        @test plan isa FaerSparseProductPlan
        @test isopen(plan)
        @test size(plan)==(7, 6)
        @test plan.format===format
        ref=A*B
        @test nnz(plan)==nnz(ref)
        C=FaerSparseMatrix{T}(plan)
        @test C.format===format
        @test compressed_storage(C).indices==compressed_storage(ref).indices
        @test iszero(Matrix(C))
        values=nonzeros(C)
        epoch=C.epoch
        @test mul!(C, plan, A, B)===C
        @test Matrix(C) ≈ Matrix(a)*Matrix(b)
        @test C.epoch==epoch
        # Reuse after the values (not the patterns) of A and B change.
        a2, b2=_revalue(a, rng), _revalue(b, rng)
        copyto!(A, FaerSparseMatrix(a2; format))
        copyto!(B, FaerSparseMatrix(b2; format))
        α, β=T(2), T(1)/3
        C0=Matrix(C)
        mul!(C, plan, A, B, α, β)
        @test Matrix(C) ≈ α*Matrix(a2)*Matrix(b2)+β*C0
        @test C.epoch==epoch
        @test Vector(values)==compressed_storage(C).values # live view stays valid
        mul!(C, plan, A, B, α, 0)
        @test Matrix(C) ≈ α*Matrix(a2)*Matrix(b2)
        mul!(C, plan, A, B, 0, 1)
        @test Matrix(C) ≈ α*Matrix(a2)*Matrix(b2)
        # Caller scratch and team.
        req=scratch_requirements(mul!, C, plan, A, B)
        @test req.bytes>0
        work=FaerScratch(mul!, C, plan, A, B)
        @test scratch_capacity(work).bytes>=req.bytes
        mul!(C, plan, A, B; scratch = work)
        @test Matrix(C) ≈ Matrix(a2)*Matrix(b2)
        team=FaerTeam(2)
        treq=scratch_requirements(mul!, C, plan, A, B; team)
        reserve!(work, mul!, C, plan, A, B; team)
        @test scratch_capacity(work).bytes>=treq.bytes
        mul!(C, plan, A, B, 1, 0; scratch = work, team)
        @test Matrix(C) ≈ Matrix(a2)*Matrix(b2)
        close(team)
        # Independent copies.
        q=copy(plan)
        close(plan)
        @test !isopen(plan)
        @test_throws ArgumentError mul!(C, plan, A, B)
        mul!(C, q, A, B)
        @test Matrix(C) ≈ Matrix(a2)*Matrix(b2)
        @test isopen(deepcopy(q))
        close(work)
    end
end

@testset "Sparse product plans validate patterns before writing" begin
    rng=MersenneTwister(51)
    a=sprand(rng, 8, 6, 0.4)
    b=sprand(rng, 6, 5, 0.4)
    A, B=FaerSparseMatrix(a), FaerSparseMatrix(b)
    plan=FaerSparseProductPlan(A, B)
    C=A*B
    snapshot=Matrix(C)
    # A different A pattern (dropzeros! on a stored zero shrinks it).
    a0=copy(a)
    a0.nzval[1]=0
    A0=FaerSparseMatrix(a0)
    dropzeros!(A0)
    @test_throws ArgumentError mul!(C, plan, A0, B)
    @test Matrix(C)==snapshot
    B1=FaerSparseMatrix(sprand(rng, 6, 5, 0.6))
    @test_throws ArgumentError mul!(C, plan, A, B1)
    # A destination without the planned pattern.
    D=FaerSparseMatrix(sprand(rng, 8, 5, 0.3))
    Dsnap=Matrix(D)
    @test_throws ArgumentError mul!(D, plan, A, B)
    @test Matrix(D)==Dsnap
    @test_throws DimensionMismatch mul!(FaerSparseMatrix(spzeros(8, 4)), plan, A, B)
    # Storage formats must match the plan.
    @test_throws ArgumentError mul!(C, plan, FaerSparseMatrix(a; format = :csr), B)
    @test_throws ArgumentError mul!(copy(C; format = :csr), plan, A, B)
    @test_throws ArgumentError FaerSparseProductPlan(A, FaerSparseMatrix(b; format = :csr))
    @test_throws DimensionMismatch FaerSparseProductPlan(B, B)
    # The destination scalar must hold the promoted operand scalar.
    @test_throws ArgumentError mul!(FaerSparseMatrix{Float32}(plan), plan, A, B)
    # Promoted operands are converted (allocating), keeping their patterns.
    Cc=FaerSparseMatrix{ComplexF64}(plan)
    mul!(Cc, plan, FaerSparseMatrix{Float32}(a), B)
    @test Matrix(Cc) ≈ Matrix(a)*Matrix(b) rtol=1e-5
    # Destination aliases are rejected before writing.
    S=FaerSparseMatrix(sparse(Diagonal(rand(rng, 4))))
    splan=FaerSparseProductPlan(S, S)
    Ssnap=Matrix(S)
    @test_throws ArgumentError mul!(S, splan, S, S)
    @test Matrix(S)==Ssnap
    # Insufficient scratch.
    small=FaerScratch(0; alignment = 1)
    @test_throws ArgumentError mul!(C, plan, A, B; scratch = small)
    @test Matrix(C)==snapshot
    @test_throws ArgumentError mul!(C, plan, A, B; scratch = 1)
    @test_throws ArgumentError mul!(C, plan, A, B; team = 1)
    # Closed handles.
    close(A0)
    @test_throws ArgumentError mul!(C, plan, A0, B)
    @test_throws ArgumentError FaerSparseProductPlan(A0, B)
end

@testset "Sparse product plans over borrowed Julia CSC operands" begin
    rng=MersenneTwister(156)
    for T in (Float64, ComplexF32), Ti in (Int32, Int64)
        a=SparseMatrixCSC{T,Ti}(sprand(rng, T, 9, 7, 0.35))
        b=SparseMatrixCSC{T,Ti}(sprand(rng, T, 7, 8, 0.35))
        plan=FaerSparseProductPlan(faerview(a), faerview(b))
        @test plan.format===:csc
        C=FaerSparseMatrix{T}(plan)
        mul!(C, plan, faerview(a), faerview(b))
        @test Matrix(C) ≈ Matrix(a)*Matrix(b)
        # Values of the Julia parents change in place; the plan is reused.
        a.nzval .= randn(rng, T, nnz(a))
        b.nzval .*= 2
        work=FaerScratch(mul!, C, plan, faerview(a), faerview(b))
        mul!(C, plan, faerview(a), faerview(b), 1, 0; scratch = work)
        @test Matrix(C) ≈ Matrix(a)*Matrix(b)
        # Mixed borrowed/owned operands are accepted (the borrowed one is copied).
        mul!(C, plan, FaerSparseMatrix(a), faerview(b))
        @test Matrix(C) ≈ Matrix(a)*Matrix(b)
        # A structural change of a parent is detected.
        a2=copy(a)
        a2[1, 1]=iszero(a2[1, 1]) ? one(T) : zero(T)
        dropzeros!(a2)
        nnz(a2)==nnz(a) || @test_throws ArgumentError mul!(C, plan, faerview(a2), faerview(b))
    end
end

@testset "Sparse product plan allocations" begin
    rng=MersenneTwister(7)
    a=sprand(rng, 400, 400, 0.02)
    b=sprand(rng, 400, 400, 0.02)
    A, B=FaerSparseMatrix(a), FaerSparseMatrix(b)
    plan=FaerSparseProductPlan(A, B)
    C=FaerSparseMatrix{Float64}(plan)
    work=FaerScratch(mul!, C, plan, A, B)
    fa, fb=faerview(a), faerview(b) # Reused 0-based index buffers.
    # Julia-heap bytes only; native buffers are covered by the Rust probe tests.
    old=_bytes(() -> mul!(C, A, B))
    planned=_bytes(() -> mul!(C, plan, A, B; scratch = work))
    borrowed=_bytes(() -> mul!(C, plan, fa, fb; scratch = work))
    @info "sparse product mul! Julia bytes" old planned borrowed
    @test planned<old
    @test Matrix(C) ≈ Matrix(a)*Matrix(b)
end
