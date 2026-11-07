using Test, LinearAlgebra, SparseArrays, Colmat

_bytes(f) = (f(); @allocated f())

@testset "Borrowed Julia sparse matrices" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), Ti in (Int32, Int64)
        dense=T[4 0 1 0; 0 0 0 2; 3 5 0 0; 0 0 6 7; 1 0 0 0]
        T<:Complex && (dense .+= im .* (dense .!= 0))
        S=SparseMatrixCSC{T,Ti}(sparse(dense))
        S.nzval[2]=zero(T) # A stored zero stays stored.
        b=faerview(S)
        @test b isa FaerBorrowedSparseMatrix{T,Ti}
        @test parent(b)===S
        @test size(b)==size(S)
        @test nnz(b)==nnz(S)
        @test b==S
        @test b[3, 2]==S[3, 2]
        @test SparseArrays.nonzeros(b)===SparseArrays.nonzeros(S)
        @test SparseArrays.rowvals(b)===SparseArrays.rowvals(S)
        @test SparseArrays.getcolptr(b)===SparseArrays.getcolptr(S)
        @test faerview(b)===b

        owned=FaerSparseMatrix(b)
        @test owned isa FaerSparseMatrix{T}
        @test nnz(owned)==nnz(S)
        @test SparseMatrixCSC(owned)==S
        @test SparseMatrixCSC(copy(b))==S
        @test FaerSparsity(b)==FaerSparsity(FaerSparseMatrix(S))
        @test Matrix(FaerMatrix(b))==Matrix(S)

        X=T.(reshape(1:12, 4, 3))
        Y=T.(reshape(1:15, 3, 5))
        # Sparse × dense / dense × sparse with native, view and borrowed operands.
        for B in (FaerMatrix(X), view(FaerMatrix(T.(reshape(1:16, 4, 4))), :, 1:3), faerview(X), X)
            @test b*B isa FaerMatrix{T}
            @test Matrix(b*B)≈Matrix(S)*Matrix(B)
        end
        for A in (FaerMatrix(Y), faerview(Y), Y)
            @test Matrix(A*b)≈Matrix(A)*Matrix(S)
        end
        @test Matrix(b*transpose(FaerMatrix(permutedims(X))))≈Matrix(S)*X
        @test Matrix(b*FaerMatrix(conj(permutedims(X)))')≈Matrix(S)*X
        v=T.(1:4)
        @test b*v isa Vector{T}
        @test b*v≈Matrix(S)*v
        @test Vector(b*FaerVector(v))≈Matrix(S)*v

        # Native mul! into a FaerMatrix destination with alpha/beta.
        C0=T.(reshape(1:15, 5, 3))
        for (α, β) in ((1, 0), (2, 0), (1, 1), (2, -3))
            C=FaerMatrix(C0)
            @test mul!(C, b, FaerMatrix(X), α, β)===C
            @test Matrix(C)≈α*Matrix(S)*X+β*C0
            C=FaerMatrix(C0)
            mul!(C, b, faerview(X), α, β)
            @test Matrix(C)≈α*Matrix(S)*X+β*C0
        end
        C=FaerMatrix(C0)
        mul!(C, b, FaerMatrix(X))
        @test Matrix(C)≈Matrix(S)*X
        D=FaerMatrix(T.(reshape(1:12, 3, 4)))
        mul!(D, FaerMatrix(Y), b, 1, 2)
        @test Matrix(D)≈Y*Matrix(S)+2*T.(reshape(1:12, 3, 4))
        @test_throws DimensionMismatch mul!(FaerMatrix(zeros(T, 4, 3)), b, FaerMatrix(X))
        @test_throws DimensionMismatch b*FaerMatrix(zeros(T, 3, 2))

        # Zero-copy values: later Julia value writes are visible.
        S.nzval[1]=T(-9)
        @test Matrix(b*FaerMatrix(X))≈Matrix(S)*X
        # Structural changes are seen at the next call.
        S[2, 1]=T(8)
        @test nnz(b)==nnz(S)
        @test Matrix(b*FaerMatrix(X))≈Matrix(S)*X
        @test SparseMatrixCSC(FaerSparseMatrix(b))==S
    end

    @testset "empty and capacity" begin
        E=spzeros(Float64, 3, 2)
        e=faerview(E)
        @test Matrix(e*FaerMatrix(ones(2, 2)))==zeros(3, 2)
        @test nnz(FaerSparseMatrix(e))==0
        @test size(faerview(spzeros(0, 0))*FaerMatrix(zeros(0, 1)))==(0, 1)
        S=sparse([1, 2], [1, 2], [1.0, 2.0], 2, 2)
        # Extra rowval/nzval capacity beyond nnz is ignored.
        push!(S.rowval, 7)
        push!(S.nzval, 99.0)
        @test Matrix(faerview(S)*FaerMatrix([1.0 0; 0 1]))==[1.0 0; 0 2]
    end

    @testset "invalid structure" begin
        @test_throws ArgumentError faerview(sparse([true false; false true]))
        bad=SparseMatrixCSC(2, 2, [1, 3, 3], [2, 1], [1.0, 2.0]) # unsorted rows
        @test_throws ArgumentError faerview(bad)
        dup=SparseMatrixCSC(2, 2, [1, 3, 3], [1, 1], [1.0, 2.0])
        @test_throws ArgumentError faerview(dup)
        oob=SparseMatrixCSC(2, 2, [1, 2, 2], [3], [1.0])
        @test_throws ArgumentError faerview(oob)
        S=sparse([1.0 0; 0 1])
        b=faerview(S)
        S.rowval[1]=3 # Corrupt after borrowing: rejected before Rust reads it.
        @test_throws ArgumentError b*FaerMatrix(ones(2, 1))
        @test_throws ArgumentError FaerSparseMatrix(b)
    end

    @testset "aliasing, failures and closed handles" begin
        S=sparse([4.0 0 1 0; 0 2 0 3; 1 0 5 0; 0 0 0 6])
        b=faerview(S)
        C0=reshape(collect(1.0:12.0), 4, 3)
        # A source viewing the destination is staged before Rust writes.
        C=FaerMatrix(C0)
        mul!(C, b, view(C, 1:4, :), 2, 3)
        @test Matrix(C)≈2*Matrix(S)*C0+3*C0
        D0=reshape(collect(1.0:16.0), 4, 4)
        D=FaerMatrix(D0)
        mul!(D, b, transpose(view(D, 1:4, 1:4)), 1, 1)
        @test Matrix(D)≈Matrix(S)*transpose(D0)+D0
        D=FaerMatrix(D0)
        mul!(D, view(D, :, :), b, 1, -1)
        @test Matrix(D)≈D0*Matrix(S)-D0
        # Failures leave the destination unchanged.
        C=FaerMatrix(C0)
        @test_throws DimensionMismatch mul!(C, b, FaerMatrix(ones(3, 3)), 2, 3)
        @test Matrix(C)==C0
        S.rowval[1]=9
        @test_throws ArgumentError mul!(C, b, FaerMatrix(ones(4, 3)), 2, 3)
        @test Matrix(C)==C0
        S.rowval[1]=1
        X=FaerMatrix(ones(4, 3))
        close(X)
        @test_throws ArgumentError mul!(C, b, X, 2, 3)
        @test Matrix(C)==C0
        E=FaerMatrix(C0)
        close(E)
        Y=FaerMatrix(ones(4, 3))
        @test_throws ArgumentError mul!(E, b, Y)
        @test Matrix(Y)==ones(4, 3)
        @test S==sparse([4.0 0 1 0; 0 2 0 3; 1 0 5 0; 0 0 0 6])
        # Mixed scalar types use the promoted product.
        b32=faerview(SparseMatrixCSC{Float32,Int}(S))
        C=FaerMatrix(C0)
        mul!(C, b32, FaerMatrix(ones(4, 3)), 2, 1)
        @test Matrix(C)≈2*Matrix(S)*ones(4, 3)+C0
        @test Matrix(b32*FaerMatrix(ones(4, 3))) isa Matrix{Float64}
    end

    @testset "construction resources" begin
        A=[4.0 1 0; 1 5 2; 0 2 6]
        S=sparse(A)
        b=faerview(S)
        team=FaerTeam(2)
        p=FaerSparseSymbolic(b; kind = :lu)
        work=FaerScratch(factorize, p, FaerSparseMatrix(b); team)
        F=factorize(p, b; scratch = work, team)
        @test F\[1.0, 2, 3]≈A\[1.0, 2, 3]
        @test lu(b; scratch = work, team)\[1.0, 2, 3]≈A\[1.0, 2, 3]
        S.nzval .*= 3
        update=FaerScratch(lu!, F, FaerSparseMatrix(b); team)
        @test lu!(F, b; scratch = update, team)===F
        @test F\[1.0, 2, 3]≈(3A)\[1.0, 2, 3]
    end

    @testset "factorization inputs" begin
        for T in (Float32, Float64, ComplexF32, ComplexF64)
            A=T[4 1 0; 1 5 2; 0 2 6]
            T<:Complex && (A=A+im*T[0 1 0; -1 0 1; 0 -1 0])
            S=sparse(A)
            b=faerview(S)
            rhs=T[1, 2, 3]
            F=lu(b)
            @test F isa FaerSparseFactorization{T,:lu}
            @test F\rhs≈A\rhs
            @test b\rhs≈A\rhs
            Q=qr(b)
            @test Q\rhs≈A\rhs
            H=cholesky(Hermitian(b, :L))
            @test H\rhs≈A\rhs
            @test cholesky(b)\rhs≈A\rhs
            P=FaerSparseSymbolic(b; kind = :lu)
            @test factorize(P, b)\rhs≈A\rhs
            # Factors keep an owned snapshot: later writes to S do not change F.
            S.nzval .*= 2
            @test F\rhs≈A\rhs
            # Refactorization reads the current borrowed values.
            @test lu!(F, b)===F
            @test F\rhs≈(2A)\rhs
            qr!(Q, b)
            @test Q\rhs≈(2A)\rhs
            cholesky!(H, b)
            @test H\rhs≈(2A)\rhs
        end
    end

    @testset "allocations" begin
        # Julia-side bytes of the borrowed paths do not grow with nnz: indices
        # are converted into reused buffers and values are read in place.
        sizes=map((200, 2000)) do n
            S=sprandn(n, n, 0.005)+10I
            B=FaerMatrix(randn(n, 4))
            J=randn(n, 4)
            b=faerview(S)
            C=FaerMatrix(zeros(n, 4))
            F=lu(b)
            (
                mul = _bytes(() -> mul!(C, b, B)),
                product = _bytes(() -> b*B),
                julia = _bytes(() -> b*J),
                owned = _bytes(() -> FaerSparseMatrix(b)),
                refactor = _bytes(() -> lu!(F, b)),
                values = sizeof(SparseArrays.nonzeros(S)),
                copied = _bytes(() -> lu!(F, FaerSparseMatrix(S))),
            )
        end
        small, large=sizes
        for key in (:mul, :product, :julia, :owned)
            @test getproperty(large, key)<8192
            @test getproperty(large, key)-getproperty(small, key)<1024
        end
        @test large.refactor<large.values
        @test large.refactor<large.copied÷2
    end
end
