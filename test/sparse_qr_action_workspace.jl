using Test, LinearAlgebra, SparseArrays, Colmat
sqa_tol(T, n) = 500eps(typeof(real(zero(T))))*max(n, 1)
function sqa_fixture(T, m, n, empty_head)
    a=zeros(T, m, n)
    for j = 1:n, i = 1:m
        empty_head && j<=m && continue
        a[i, j]=T(mod(j-1, max(m, 1))+1==i ? 4+j : 0.04)
        T<:Complex && (a[i, j]+=T(0.02im*mod(i+2j, 5)))
    end
    a
end
function sqa_check(op, reference, team, T; shortage = false)
    b=ones(T, size(reference, 2), 3)
    x=fill(T(42), size(reference, 1), 3)
    req=scratch_requirements(mul!, x, op, b; team)
    work=FaerScratch(mul!, x, op, b; team)
    cap=scratch_capacity(work)
    for _ = 1:2
        @test mul!(x, op, b; scratch = work, team)===x
        @test x≈reference*b atol=sqa_tol(T, max(size(reference)...)) rtol=sqa_tol(
            T,
            max(size(reference)...),
        )
        @test scratch_capacity(work)==cap
    end
    if shortage && req.bytes>0
        short=FaerScratch(req.bytes-1; alignment = req.alignment)
        saved=copy(x)
        @test_throws ArgumentError mul!(x, op, b; scratch = short, team)
        @test x==saved
        reserve!(short, mul!, x, op, b; team)
        mul!(x, op, b; scratch = short, team)
        @test x≈reference*b atol=sqa_tol(T, max(size(reference)...)) rtol=sqa_tol(
            T,
            max(size(reference)...),
        )
        close(short)
    end
    close(work)
    req
end
@testset "Sparse QR inverse actions with caller scratch/team" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        format in (:csc, :csr),
        algorithm in (:auto, :supernodal)

        FaerTeam(2) do team
            for (m, n, empty) in (
                    (0, 0, false),
                    (4, 0, false),
                    (5, 3, false),
                    (3, 3, false),
                    (0, 3, true),
                    (3, 6, false),
                    (3, 6, true),
                ),
                reversed in (false, true)

                a=sqa_fixture(T, m, n, empty)
                source=FaerSparseMatrix(sparse(a); format)
                f=qr(
                    source;
                    algorithm,
                    ordering = reversed ? reverse(collect(1:n)) : :default,
                )
                close(source)
                close(f.source)
                op=FaerLinearOperator(f)
                close(f)
                reference=pinv(a)
                for (variant, ref) in (
                    (op, reference),
                    (conj(op), conj(reference)),
                    (transpose(op), transpose(reference)),
                    (op', reference'),
                )
                    req=sqa_check(variant, ref, team, T; shortage = (m, n)==(3, 6))
                    if m>0 && n>0
                        @test req.bytes>0
                    end
                    variant===op || close(variant)
                end
                g=deepcopy(op)
                close(op)
                @test Matrix(g)≈reference atol=sqa_tol(T, max(m, n)) rtol=sqa_tol(
                    T,
                    max(m, n),
                )
                close(g)
            end
        end
    end
end
@testset "Sparse blocked QR projections and promoted cached rank" begin
    FaerTeam(2) do team
        for (m, n, empty) in ((129, 65, false), (33, 70, false), (33, 70, true)),
            algorithm in (:auto, :supernodal)

            a=sqa_fixture(ComplexF64, m, n, empty)
            f=qr(FaerSparseMatrix(sparse(a)); algorithm)
            op=FaerLinearOperator(f)
            close(f)
            close(f.source)
            reference=pinv(a)
            for (variant, ref) in (
                (op, reference),
                (op', reference'),
                (transpose(op), transpose(reference)),
                (conj(op), conj(reference)),
            )
                sqa_check(variant, ref, team, ComplexF64; shortage = true)
                variant===op || close(variant)
            end
            close(op)
        end
        for algorithm in (:auto, :supernodal)
            a=Float32[1 0 0; 0 1e-9 0]
            f=qr(FaerSparseMatrix(sparse(a)); algorithm)
            op=FaerLinearOperator(f)
            close(f)
            b=ComplexF64[1, 1]
            expected=pinv(ComplexF64.(a); rtol = 2eps(Float32))*b
            work=FaerScratch(*, op, b; team)
            x=*(op, b; scratch = work, team)
            @test x≈expected atol=1e-6 rtol=1e-6
            @test x[2]==0
            close(work)
            close(op)
        end
        a=sqa_fixture(ComplexF64, 3, 6, true)
        f=qr(FaerSparseMatrix(sparse(a)); algorithm = :supernodal)
        op=FaerLinearOperator(f)
        close(f)
        b=ones(ComplexF64, 3, 2)
        work=FaerScratch(*, op, b; team)
        results=fetch.([@async *(op, b; scratch = work, team) for _ = 1:3])
        for result in results
            @test Matrix(result)≈pinv(a)*b atol=1e-11 rtol=1e-11
            close(result)
        end
        close(op)
        close(work)
    end
end
