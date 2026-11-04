using Test, LinearAlgebra, SparseArrays, Random, Colmat
ow_tol(T, n = 4) = 300eps(typeof(real(zero(T))))*max(1, n)
function ow_check(op, ref, team, T)
    m, n=size(ref)
    b=reshape(T.(1:(n*3)), n, 3)/T(max(1, n))
    req=scratch_requirements(*, op, b; team)
    work=FaerScratch(*, op, b; team)
    @test scratch_capacity(work)==req
    expected=ref*b
    for _ = 1:2
        x=*(op, b; scratch = work, team)
        @test Matrix(x)≈expected atol=ow_tol(T, max(m, n)) rtol=ow_tol(T, max(m, n))
        @test scratch_capacity(work)==req
        close(x)
    end
    out=FaerMatrix(zeros(T, m, 3))
    old=Matrix(out)
    v=view(out, :, :)
    reqmul=scratch_requirements(mul!, out, op, b, T(2), T(0.5); team)
    reserve!(work, mul!, out, op, b, T(2), T(0.5); team)
    @test mul!(out, op, b, T(2), T(0.5); scratch = work, team)===out
    @test Matrix(out)≈2expected+0.5old atol=ow_tol(T, max(m, n)) rtol=ow_tol(T, max(m, n))
    @test Matrix(v)==Matrix(out)
    vector=ones(T, n)
    vout=zeros(T, m)
    vwork=FaerScratch(mul!, vout, op, vector; team)
    @test mul!(vout, op, vector; scratch = vwork, team)===vout
    @test vout≈ref*vector atol=ow_tol(T, max(m, n)) rtol=ow_tol(T, max(m, n))
    @test *(op, vector; scratch = vwork, team)≈ref*vector atol=ow_tol(T, max(m, n)) rtol=ow_tol(
        T,
        max(m, n),
    )
    close(vwork)
    right=ones(T, 2, m)
    rdest=zeros(T, 2, n)
    rwork=FaerScratch(mul!, rdest, right, op; team)
    @test mul!(rdest, right, op; scratch = rwork, team)===rdest
    @test rdest≈right*ref atol=ow_tol(T, max(m, n)) rtol=ow_tol(T, max(m, n))
    @test Matrix(*(right, op; scratch = rwork, team))≈right*ref atol=ow_tol(T, max(m, n)) rtol=ow_tol(
        T,
        max(m, n),
    )
    @test scratch_requirements(*, right, op; team)==scratch_requirements(
        mul!,
        rdest,
        right,
        op;
        team,
    )
    close(rwork)
    close(out)
    close(work)
end
@testset "Reusable operator workspace and team application" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), width in (1, 2)
        FaerTeam(width) do team
            phase=T<:Complex ? T(0.3im) : zero(T)
            a=T[4 1; 2 3]+phase*I
            for source in (
                FaerMatrix(a),
                FaerSparseMatrix(sparse(a)),
                FaerSparseMatrix(sparse(a); format = :csr),
                Diagonal(T[4, 3]),
            )
                ref=Matrix(source)
                op=FaerLinearOperator(source)
                source isa Union{FaerMatrix,FaerSparseMatrix} && close(source)
                for (variant, reference) in (
                    (op, ref),
                    (transpose(op), transpose(ref)),
                    (op', ref'),
                    (conj(op), conj(ref)),
                )
                    ow_check(variant, reference, team, T)
                    variant===op || close(variant)
                end
                close(op)
            end
            for n in (0, 1, 5)
                op=FaerLinearOperator(I, n; T)
                ow_check(op, Matrix{T}(I, n, n), team, T)
                close(op)
            end
            left=FaerLinearOperator(FaerMatrix(a))
            right=FaerLinearOperator(Diagonal(T[2, 3]))
            op=left*right
            close(left)
            close(right)
            ow_check(op, a*Diagonal(T[2, 3]), team, T)
            close(op)
            for fun in (lu, full_piv_lu, qr, (a->qr(a, ColumnNorm())))
                owner=FaerMatrix(a)
                f=fun(owner)
                op=FaerLinearOperator(f)
                close(owner)
                close(f)
                ow_check(op, inv(a), team, T)
                ow_check(op', inv(a)', team, T)
                close(op)
            end
            positive=T[4 1; 1 3]
            for fun in (cholesky, ldlt, bunchkaufman)
                owner=FaerMatrix(positive)
                f=fun(Hermitian(owner))
                op=FaerLinearOperator(f)
                close(owner)
                close(f)
                ow_check(op, inv(positive), team, T)
                close(op)
            end
        end
    end
end
@testset "Sparse retained actions, composition and scalar promotion" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        FaerTeam(2) do team
            for format in (:csc, :csr),
                algorithm in (:simplicial, :supernodal),
                kind in (:lu, :qr, :cholesky)

                a=T[4 1; 1 3]
                source=FaerSparseMatrix(sparse(a); format)
                f=kind===:lu ? lu(source; algorithm) :
                  kind===:qr ? qr(source; algorithm) :
                  cholesky(Hermitian(source); algorithm)
                op=FaerLinearOperator(f)
                close(f)
                close(source)
                ow_check(op, inv(a), team, T)
                close(op)
            end
            a=T[3 0; 1 4; 0 2]
            source=FaerMatrix(a)
            # Retained SVD inverse actions support rectangular factors. Dense
            # QR preconditioners currently require square factors by contract.
            f=svd(source; native = true)
            op=FaerLinearOperator(f)
            close(f)
            close(source)
            ow_check(op, pinv(a), team, T)
            ow_check(op', pinv(a)', team, T)
            close(op)
            for R in (Float32, Float64, ComplexF32, ComplexF64)
                a=T[4 1; 1 3]
                f=lu(FaerMatrix(a))
                op=FaerLinearOperator(f)
                close(f)
                b=R[1 3; 2 4]
                P=promote_type(T, R)
                out=zeros(P, 2, 2)
                work=FaerScratch(mul!, out, op, b; team)
                mul!(out, op, b; scratch = work, team)
                @test out≈P.(a)\P.(b) atol=ow_tol(P) rtol=ow_tol(P)
                close(work)
                # alpha/beta and destination types participate in the query.
                dest=zeros(ComplexF64, 2, 2)
                alpha=2.0+im
                beta=0.5im
                work=FaerScratch(mul!, dest, op, b, alpha, beta; team)
                mul!(dest, op, b, alpha, beta; scratch = work, team)
                @test dest≈alpha .* (ComplexF64.(a)\ComplexF64.(b)) atol=ow_tol(T) rtol=ow_tol(
                    T,
                )
                close(work)
                close(op)
            end
        end
    end
end
@testset "Operator workspace alias, shortage, failures and lifecycle" begin
    FaerTeam(2) do team
        a=[4.0 1; 1 3]
        left=FaerLinearOperator(FaerMatrix(a))
        right=FaerLinearOperator(Diagonal([2.0, 3]))
        op=left*right
        close(left)
        close(right)
        b=[1.0 3; 2 4]
        expected=a*Diagonal([2.0, 3])*b
        out=FaerMatrix(b)
        work=FaerScratch(mul!, out, op, out; team)
        mul!(out, op, out; scratch = work, team)
        @test Matrix(out)≈expected
        req=scratch_requirements(mul!, out, op, out; team)
        @test req.bytes>0
        saved=Matrix(out)
        for short in (
            FaerScratch(req.bytes-1; alignment = req.alignment),
            FaerScratch(req.bytes; alignment = 1),
        )
            @test_throws ArgumentError mul!(out, op, out; scratch = short, team)
            @test Matrix(out)==saved
            @test_throws ArgumentError mul!(out, op, out, 0, 0; scratch = short, team)
            @test Matrix(out)==saved
            reserve!(short, mul!, out, op, out; team)
            mul!(out, op, b; scratch = short, team)
            @test Matrix(out)≈expected
            copyto!(out, saved)
            close(short)
        end
        results=fetch.([@async *(op, b; scratch = work, team) for _ = 1:3])
        for x in results
            @test Matrix(x)≈expected
            close(x)
        end
        @test_throws ArgumentError mul!(out, op, b; scratch = UInt8[], team)
        @test Matrix(out)==saved
        @test_throws ArgumentError mul!(out, op, b; scratch = work, team = 2)
        @test Matrix(out)==saved
        # Valid zero-alpha does not inspect NaN RHS elements or previous output.
        mul!(out, op, fill(NaN, 2, 2), 0, 0; scratch = work, team)
        @test iszero(Matrix(out))
        close(work)
        @test_throws ArgumentError mul!(out, op, b; scratch = work, team)
        close(out)
        close(op)
    end
    op=FaerLinearOperator(I, 2)
    team=FaerTeam(2)
    work=FaerScratch(*, op, ones(2); team)
    close(team)
    out=fill(42.0, 2)
    @test_throws ArgumentError mul!(out, op, ones(2); scratch = work, team)
    @test out==fill(42.0, 2)
    close(op)
    @test_throws ArgumentError scratch_requirements(*, op, ones(2))
    close(work)
end
@testset "Operator workspace current native dtype/shape guards" begin
    FaerTeam(2) do team
        op=FaerLinearOperator(I, 2; T = Float32)
        b=FaerMatrix(ones(Float32, 2, 2))
        out=FaerMatrix(fill(42.0f0, 2, 2))
        work=FaerScratch(mul!, out, op, b; team)
        saved=Matrix(out)
        Colmat._native_call(
            Colmat.RustCrate.zeroes,
            b,
            UInt(2),
            UInt(3),
            Colmat._tag(Float32),
        )
        @test_throws Exception mul!(out, op, b; scratch = work, team)
        @test Matrix(out)==saved
        @test_throws Exception op*b
        @test_throws Exception mul!(out, op, b)
        @test Matrix(out)==saved
        Colmat._native_call(
            Colmat.RustCrate.zeroes,
            b,
            UInt(2),
            UInt(2),
            Colmat._tag(ComplexF64),
        )
        @test_throws Exception mul!(out, op, b; scratch = work, team)
        @test Matrix(out)==saved
        Colmat._native_call(
            Colmat.RustCrate.operator_identity,
            op,
            UInt(2),
            Colmat._tag(ComplexF64),
        )
        @test_throws Exception scratch_requirements(*, op, ones(Float32, 2))
        @test_throws Exception op*ones(Float32, 2)
        @test_throws Exception op*ones(ComplexF64, 2)
        @test_throws Exception scratch_requirements(*, op, ones(ComplexF64, 2))
        @test_throws Exception mul!(out, op, ones(Float32, 2, 2); scratch = work, team)
        @test Matrix(out)==saved
        close(work)
        close(op)
        close(b)
        close(out)
    end
end

@testset "Native vector and view workspace application" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        FaerTeam(2) do team
            a=T[4 1; 2 3]
            op=FaerLinearOperator(FaerMatrix(a))
            x=FaerVector(T[1, 2])
            y=FaerVector(zeros(T, 2))
            work=FaerScratch(mul!, y, op, x; team)
            @test mul!(y, op, x; scratch = work, team)===y
            @test Vector(y)≈a*T[1, 2]
            z=*(op, x; scratch = work, team)
            @test z isa FaerVector
            @test Vector(z)≈a*T[1, 2]
            close(z)
            close(x)
            close(y)
            close(work)
            source=reshape(T.(1:12), 4, 3)
            b=view(source, 1:2:4, 3:-1:1)
            parentout=FaerMatrix(zeros(T, 4, 6))
            out=view(parentout, 1:2:4, 2:2:6)
            work=FaerScratch(mul!, out, op, b, T(2), T(3); team)
            mul!(out, op, b, T(2), T(3); scratch = work, team)
            @test Matrix(out)≈2a*Matrix(b)
            @test iszero(Matrix(parentout)[2:2:4, :])
            close(parentout)
            close(work)
            close(op)
        end
    end
    FaerTeam(2) do team
        op=FaerLinearOperator(FaerMatrix(ComplexF64[2im 0; 0 1]))
        out=FaerMatrix(ones(2, 2))
        saved=Matrix(out)
        b=ones(2, 2)
        work=FaerScratch(mul!, out, op, b; team)
        @test_throws InexactError mul!(out, op, b; scratch = work, team)
        @test Matrix(out)==saved
        close(out)
        close(op)
        close(work)
    end
end
