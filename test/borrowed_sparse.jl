using Test, LinearAlgebra, SparseArrays, Random, Colmat

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

    @testset "transposed and adjoint borrowed operands" begin
        for T in (Float32, Float64, ComplexF32, ComplexF64), Ti in (Int32, Int64)
            dense=T[4 0 1 0; 0 0 0 2; 3 5 0 0; 0 0 6 7; 1 0 0 0]
            T<:Complex && (dense .+= im .* (dense .!= 0))
            S=SparseMatrixCSC{T,Ti}(sparse(dense))
            b=faerview(S)
            @test faerview(transpose(S)) isa Transpose{T,FaerBorrowedSparseMatrix{T,Ti}}
            @test parent(faerview(transpose(S))).parent===S
            @test faerview(S') isa Adjoint{T,FaerBorrowedSparseMatrix{T,Ti}}
            X=T.(reshape(1:15, 5, 3))
            T<:Complex && (X=X .+ im)
            Y=T.(reshape(1:8, 2, 4))
            for (t, M) in ((transpose(b), transpose(dense)), (b', dense'))
                for B in (FaerMatrix(X), faerview(X), X, view(FaerMatrix(X), :, 1:2))
                    @test t*B isa FaerMatrix{T}
                    @test Matrix(t*B)≈M*Matrix(B)
                end
                @test Matrix(t*FaerMatrix(permutedims(X))')≈M*permutedims(X)'
                for A in (FaerMatrix(Y), faerview(Y), Y)
                    @test Matrix(A*t)≈Matrix(A)*M
                end
                v=T.(1:5)
                @test t*v isa Vector{T}
                @test t*v≈M*v
                C0=T.(reshape(1:12, 4, 3))
                C=FaerMatrix(C0)
                @test mul!(C, t, FaerMatrix(X), 2, -1)===C
                @test Matrix(C)≈2*M*X-C0
                C=FaerMatrix(C0)
                mul!(C, t, faerview(X))
                @test Matrix(C)≈M*X
                D=FaerMatrix(zeros(T, 2, 5))
                mul!(D, faerview(Y), t, 1, 0)
                @test Matrix(D)≈Y*M
                @test_throws DimensionMismatch t*FaerMatrix(X[1:4, :])
            end
            # Values are read in place at every call.
            S.nzval[1]=T(-3)
            @test Matrix(transpose(b)*FaerMatrix(X))≈transpose(Matrix(S))*X
            # Mixed scalar types promote.
            @test Matrix(transpose(faerview(sparse(Float32[1 2; 0 3])))*[1.0 0; 0 1]) isa Matrix{Float64}
        end
    end

    @testset "borrowed value destinations" begin
        for T in (Float64, ComplexF32), Ti in (Int32, Int64)
            S=SparseMatrixCSC{T,Ti}(sparse(T[1 0 2; 0 3 0; 4 0 5]))
            S.nzval[2]=zero(T) # stored zero stays stored
            S0=copy(S)
            nz=SparseArrays.nonzeros(S)
            d=faerview(S)
            src=sparse(T[1 0 0; 0 2 0; 1 0 1])
            @test add_assign!(d, src)===d
            @test SparseArrays.nonzeros(S)===nz
            @test S==S0+src
            @test SparseArrays.rowvals(S)==SparseArrays.rowvals(S0)
            sub_assign!(d, FaerSparseMatrix(src))
            @test S==S0
            add_into!(d, src, faerview(S0))
            @test S==src+S0
            sub_into!(d, faerview(S0), transpose(FaerSparseMatrix(sparse(transpose(src)))))
            @test S==S0-src
            # Aliased sources are snapshotted first.
            copyto!(S.nzval, S0.nzval)
            add_assign!(d, d)
            @test S==2*S0
            # A source outside the pattern is rejected before any write.
            copyto!(S.nzval, S0.nzval)
            @test_throws ArgumentError add_assign!(d, sparse(T[0 1 0; 0 0 0; 0 0 0]))
            @test S==S0
            @test_throws DimensionMismatch add_assign!(d, spzeros(T, 2, 3))
            # Scaling and broadcasting update the parent in place.
            rmul!(d, 2)
            @test S==2*S0
            lmul!(T(1//2), d)
            @test S==S0
            d .*= 3
            @test S==3*S0 && SparseArrays.nonzeros(S)===nz
            d .= d .+ d
            @test S==6*S0
            fill!(d, zero(T))
            @test iszero(S)
        end
    end

    @testset "planned products into and from borrowed matrices" begin
        rng=Random.MersenneTwister(113)
        for T in (Float64, ComplexF64), Ti in (Int32, Int64)
            a=SparseMatrixCSC{T,Ti}(sprand(rng, T, 9, 7, 0.35))
            b=SparseMatrixCSC{T,Ti}(sprand(rng, T, 7, 8, 0.35))
            A, B=FaerSparseMatrix(a), FaerSparseMatrix(b)
            for (x, y) in ((faerview(a), faerview(b)), (faerview(a), B), (A, faerview(b)))
                plan=FaerSparseProductPlan(x, y)
                @test plan.format===:csc
                C=SparseMatrixCSC(FaerSparseMatrix{T}(plan))
                c=faerview(C)
                nz=SparseArrays.nonzeros(C)
                @test mul!(c, plan, x, y)===c
                @test C≈a*b
                @test SparseArrays.nonzeros(C)===nz
                C0=copy(C)
                work=FaerScratch(mul!, c, plan, x, y)
                mul!(c, plan, x, y, 2, -1; scratch = work)
                @test C≈2*a*b-C0
                team=FaerTeam(2)
                mul!(c, plan, x, y; team)
                @test C≈a*b
                # Owned destinations accept the mixed operands without copies.
                D=FaerSparseMatrix{T}(plan)
                mul!(D, plan, x, y)
                @test SparseMatrixCSC(D)≈a*b
            end
            plan=FaerSparseProductPlan(faerview(a), faerview(b))
            C=SparseMatrixCSC(FaerSparseMatrix{T}(plan))
            # Destination pattern mismatch: rejected before any write.
            bad=copy(C)
            bad.nzval .= 1
            bi, bj, _=findnz(bad)
            bad[bi[1], bj[1]]=0
            dropzeros!(bad)
            vals=copy(bad.nzval)
            @test_throws ArgumentError mul!(faerview(bad), plan, faerview(a), faerview(b))
            @test bad.nzval==vals
            # A destination sharing values with an operand is rejected.
            sq=SparseMatrixCSC{T,Ti}(sprand(rng, T, 6, 6, 0.4)+I)
            sp=FaerSparseProductPlan(faerview(sq), faerview(sq))
            @test_throws ArgumentError mul!(faerview(sq), sp, faerview(sq), faerview(sq))
            # Owned CSR operands cannot mix with borrowed CSC ones.
            @test_throws ArgumentError FaerSparseProductPlan(faerview(a), FaerSparseMatrix(b; format = :csr))
        end
    end

    @testset "allocations of transposed and destination paths" begin
        sizes=map((200, 2000)) do n
            S=sprandn(n, n, 0.005)+10I
            B=FaerMatrix(randn(n, 4))
            b=faerview(S)
            C=FaerMatrix(zeros(n, 4))
            plan=FaerSparseProductPlan(b, b)
            P=SparseMatrixCSC(FaerSparseMatrix{Float64}(plan))
            p=faerview(P)
            work=FaerScratch(mul!, p, plan, b, b)
            (
                transposed = _bytes(() -> transpose(b)*B),
                adjoint_mul = _bytes(() -> mul!(C, b', B)),
                planned = _bytes(() -> mul!(p, plan, b, b; scratch = work)),
                fallback = _bytes(() -> Matrix(transpose(S))*Matrix(B)),
            )
        end
        small, large=sizes
        @info "borrowed transposed/destination Julia bytes" small large
        for key in (:transposed, :adjoint_mul, :planned)
            @test getproperty(large, key)<16384
            @test getproperty(large, key)-getproperty(small, key)<1024
        end
        @test large.transposed<large.fallback÷100
    end
end
