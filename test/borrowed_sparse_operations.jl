using Test, LinearAlgebra, SparseArrays, Random, Colmat

_bytes(f) = (f(); @allocated f())
# Julia-side bytes at two sizes: constant when no owned copy is made in Julia
# (Rust-owned results and the reused 0-based buffers are not counted).
function _flat(f, small, big)
    x, y = _bytes(() -> f(small)), _bytes(() -> f(big))
    return y < 16384 && abs(y - x) < 1024
end
const _Owner = Union{
    FaerSparseMatrix,Transpose{<:Any,<:FaerSparseMatrix},Adjoint{<:Any,<:FaerSparseMatrix}
}
function _sample(::Type{T}, n, seed; density=0.3) where {T}
    return sprand(MersenneTwister(seed), T, n, n, density) + sparse(T(4) * I, n, n)
end

@testset "Borrowed CSR to owned CSR" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), Ti in (Int32, Int64)
        S = SparseMatrixCSC{T,Ti}(_sample(T, 6, 1))
        S.nzval[3] = zero(T) # stored zeros stay stored
        b = faerview(S)
        for (w, want) in ((transpose(b), transpose(S)), (adjoint(b), adjoint(S)))
            o = FaerSparseMatrix(w)
            @test o isa FaerSparseMatrix{T}
            @test o.format === :csr
            @test nnz(o) == nnz(S)
            @test Matrix(o) == Matrix(want)
            @test copy(w).format === :csr
            @test Matrix(copy(w)) == Matrix(want)
            c = FaerSparseMatrix(w; format=:csc)
            @test c.format === :csc && Matrix(c) == Matrix(want)
            @test FaerSparseMatrix(faerview(w isa Transpose ? transpose(S) : S')).format ===
                :csr
            # Used to recurse forever through SparseArrays' generic conversion.
            @test sparse(w) == sparse(want)
            @test SparseMatrixCSC(w) == SparseMatrixCSC(want)
            @test Matrix(FaerSparseMatrix{ComplexF64}(w)) == ComplexF64.(Matrix(want))
        end
    end
    small = faerview(_sample(Float64, 400, 2; density=0.005))
    big = faerview(_sample(Float64, 400, 2; density=0.05))
    @test _flat(b -> FaerSparseMatrix(transpose(b)), small, big)
    @test _flat(b -> FaerSparseMatrix(adjoint(b)), small, big)
end

@testset "Linear operators from borrowed sparse matrices" begin
    for T in (Float64, ComplexF64)
        S = _sample(T, 5, 3)
        b = faerview(S)
        x = rand(MersenneTwister(4), T, 5)
        op = FaerLinearOperator(b)
        @test op.storage === :csc
        @test Vector(op * FaerVector(x)) ≈ S * x
        # A snapshot: later writes to the parent do not change the operator.
        S.nzval[1] += one(T)
        @test Vector(op * FaerVector(x)) ≈ (S - sparse([1], [1], [one(T)], 5, 5)) * x
        for (w, want) in ((transpose(b), transpose(S)), (adjoint(b), adjoint(S)))
            t = FaerLinearOperator(w)
            @test t.storage !== :dense
            @test Vector(t * FaerVector(x)) ≈ want * x
        end
        h = FaerLinearOperator(Hermitian(faerview(S + S'), :L))
        @test h.storage !== :dense
        @test Vector(h * FaerVector(x)) ≈ Hermitian(S + S', :L) * x
        bad = copy(S)
        bad.nzval[1] = T(NaN)
        @test_throws ArgumentError FaerLinearOperator(faerview(bad))
        P = S * S' + 5I
        y = cg(faerview(P), x; maxiter=50)
        @test P * Vector(y.x) ≈ x
    end
end

@testset "Plan-less sparse products and sums with borrowed operands" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        A = _sample(T, 6, 5)
        B = _sample(T, 6, 6)
        a, b = faerview(A), faerview(B)
        oa, ob = FaerSparseMatrix(A), FaerSparseMatrix(B; format=:csr)
        ops = (
            (a, A),
            (transpose(a), transpose(A)),
            (adjoint(a), adjoint(A)),
            (oa, A),
            (ob, B),
            (transpose(ob), transpose(B)),
        )
        for (x, X) in ops, (y, Y) in ops
            x isa _Owner && y isa _Owner && continue
            for (f, ref) in (
                (*, Matrix(X) * Matrix(Y)),
                (+, Matrix(X) + Matrix(Y)),
                (-, Matrix(X) - Matrix(Y)),
            )
                r = f(x, y)
                @test r isa FaerSparseMatrix{T}
                @test Matrix(r) ≈ ref
            end
        end
        # Owners produced by permutation and kron are read in place too.
        p = FaerPermutation([3, 1, 6, 2, 5, 4])
        for o in (
            FaerSparseMatrix(B)[p, p],
            kron(
                FaerSparseMatrix(sparse(T[1 0; 2 3])),
                FaerSparseMatrix(sparse(Matrix{T}(I, 3, 3))),
            ),
        )
            O = Matrix(o)
            @test Matrix(a * o) ≈ Matrix(A) * O
            @test Matrix(o * a) ≈ O * Matrix(A)
            @test Matrix(a + o) ≈ Matrix(A) + O
            @test Matrix(transpose(a) - o) ≈ transpose(Matrix(A)) - O
        end
        @test_throws DimensionMismatch faerview(sparse(ones(T, 2, 3))) * a
        @test_throws DimensionMismatch faerview(sparse(ones(T, 2, 3))) + a
        @test faerview(A) * FaerSparseMatrix(sparse(ones(Float32, 6, 2))) isa
            FaerSparseMatrix{promote_type(T, Float32)}
    end
    small = faerview(_sample(Float64, 300, 7; density=0.005))
    big = faerview(_sample(Float64, 300, 7; density=0.05))
    owner = FaerSparseMatrix(_sample(Float64, 300, 8; density=0.001))
    for f in (*, +, -)
        @test _flat(b -> f(b, b), small, big)
        @test _flat(b -> f(transpose(b), b), small, big)
        @test _flat(b -> f(b, owner), small, big)
    end
end

@testset "Sparse triangular solves with borrowed matrices" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        L = sparse(LowerTriangular(Matrix(_sample(T, 6, 8))))
        Full = _sample(T, 6, 9)
        for S in (L, Full)
            b = faerview(S)
            x = rand(MersenneTwister(10), T, 6)
            X = rand(MersenneTwister(11), T, 6, 2)
            for W in
                (LowerTriangular, UpperTriangular, UnitLowerTriangular, UnitUpperTriangular)
                for (w, M) in
                    ((b, S), (transpose(b), transpose(S)), (adjoint(b), adjoint(S)))
                    ref = W(Matrix(M))
                    W === UpperTriangular && S === L && w === b && continue # singular upper part
                    @test (W(w) \ x) ≈ ref \ x
                    @test (W(w) \ X) ≈ ref \ X
                    y = copy(x)
                    ldiv!(W(w), y)
                    @test y ≈ ref \ x
                    z = FaerMatrix(X)
                    ldiv!(W(w), z)
                    @test Matrix(z) ≈ ref \ X
                    d = zeros(T, 6)
                    ldiv!(d, W(w), x)
                    @test d ≈ ref \ x
                end
            end
            @test (LowerTriangular(b) \ Float32.(real(x))) ≈
                LowerTriangular(Matrix(S)) \ real(x) rtol = 1e-4
        end
    end
    # Zero diagonal and aliasing.
    S = sparse([1.0 0; 1 0])
    @test_throws SingularException LowerTriangular(faerview(S)) \ ones(2)
    A = sparse(LowerTriangular(Matrix(_sample(Float64, 4, 12))))
    @test_throws ArgumentError ldiv!(LowerTriangular(faerview(A)), view(nonzeros(A), 1:4))
    # With caller scratch the Julia-side cost does not depend on nnz.
    lower(d) =
        faerview(sparse(LowerTriangular(Matrix(_sample(Float64, 200, 13; density=d)))))
    small, big = lower(0.005), lower(0.1)
    y = rand(200)
    scratch(b) = FaerScratch(ldiv!, LowerTriangular(b), y)
    ss, sb = scratch(small), scratch(big)
    @test _flat(
        b -> ldiv!(LowerTriangular(b), y; scratch=b === small ? ss : sb), small, big
    )
end

@testset "Symbolic analysis reads borrowed and CSC owner patterns in place" begin
    S = _sample(Float64, 12, 14)
    H = S + S'
    for (kind, A) in ((:lu, S), (:qr, S), (:cholesky, H))
        for opts in (
            (;),
            (; ordering=:identity),
            (; algorithm=:simplicial),
            (; algorithm=:supernodal),
            (; ordering=collect(12:-1:1)),
        )
            kind === :cholesky &&
                haskey(opts, :ordering) &&
                opts.ordering isa Vector &&
                continue
            want = FaerSparseSymbolic(FaerSparsity(FaerSparseMatrix(A)); kind, opts...)
            for src in (faerview(A), FaerSparseMatrix(A), FaerSparseMatrix(A; format=:csr))
                p = FaerSparseSymbolic(src; kind, opts...)
                @test size(p) == size(want)
                f = factorize(p, FaerSparseMatrix(A))
                g = factorize(want, FaerSparseMatrix(A))
                x = rand(MersenneTwister(15), 12)
                @test f \ x ≈ g \ x
                @test f \ x ≈ Matrix(A) \ x
            end
            s = FaerScratch(FaerSparseSymbolic, faerview(A); kind, opts...)
            p = FaerSparseSymbolic(faerview(A); kind, scratch=s, opts...)
            @test factorize(p, faerview(A)) \ ones(12) ≈ Matrix(A) \ ones(12)
        end
    end
    p = FaerSparseSymbolic(faerview(S); kind=:lu, algorithm=:simplicial, ordering=:colamd)
    @test factorize(p, faerview(S)) \ ones(12) ≈ Matrix(S) \ ones(12)
    @test lu(faerview(S)) \ ones(12) ≈ Matrix(S) \ ones(12)
    @test cholesky(Hermitian(faerview(H), :L)) \ ones(12) ≈ Matrix(H) \ ones(12)
    @test_throws DimensionMismatch FaerSparseSymbolic(
        faerview(sparse(ones(2, 3))); kind=:lu
    )
    bad = SparseMatrixCSC(2, 2, [1, 3, 3], [2, 1], [1.0, 1.0]) # unsorted rows
    @test_throws ArgumentError FaerSparseSymbolic(faerview(bad))
end
