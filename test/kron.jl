using Test, LinearAlgebra, SparseArrays, Colmat

_kron_data(::Type{T}, m, n, shift = 0) where {T} =
    T<:Complex ? T[complex(i+2j+shift, i-j) for i in 1:m, j in 1:n] :
    T[i+2j+shift for i in 1:m, j in 1:n]
_kron_vector_data(::Type{T}, n, shift = 0) where {T} = vec(_kron_data(T, n, 1, shift))

@testset "Dense Kronecker products of views, vectors and borrowed arrays" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        a=_kron_data(T, 4, 5)
        b=_kron_data(T, 3, 2, 1)
        A=FaerMatrix(a)
        B=FaerMatrix(b)
        r=kron(A, B)
        @test r isa FaerMatrix{T}
        @test Matrix(r)==kron(a, b)
        # Strided, reversed and transposed operands are read in place.
        operands=(
            (view(A, 1:2:4, 5:-1:1), a[1:2:4, 5:-1:1]),
            (view(B, 3:-1:1, :), b[3:-1:1, :]),
            (A', a'),
            (transpose(B), transpose(b)),
            (view(A, :, 2:4)', a[:, 2:4]'),
            (faerview(b), b),
            (faerview(view(copy(a), 4:-1:1, 1:2:5)), a[4:-1:1, 1:2:5]),
            (faerview(copy(b))', b'),
        )
        for (x, xx) in operands, (y, yy) in operands
            r=kron(x, y)
            @test r isa FaerMatrix{T}
            @test Matrix(r)==kron(xx, yy)
        end
        # Mixed native and Julia operands give a native owner.
        for (x, xx) in operands
            @test kron(x, b) isa FaerMatrix{T}
            @test Matrix(kron(x, b))==kron(xx, b)
            @test Matrix(kron(a', x))==kron(a', xx)
            @test Matrix(kron(x, view(b, 3:-1:1, :)))==kron(xx, b[3:-1:1, :])
        end
        # Vectors follow LinearAlgebra's shapes.
        v=_kron_vector_data(T, 3)
        w=_kron_vector_data(T, 4, 2)
        V=FaerVector(v)
        W=FaerVector(w)
        @test kron(V, W) isa FaerVector{T}
        @test Vector(kron(V, W))==kron(v, w)
        @test Vector(kron(view(V, 3:-1:1), W))==kron(v[3:-1:1], w)
        @test Vector(kron(V, w))==kron(v, w)
        @test Vector(kron(v, view(W, 1:2:4)))==kron(v, w[1:2:4])
        @test Vector(kron(faerview(v), faerview(w)))==kron(v, w)
        @test kron(V', W') isa LinearAlgebra.Adjoint{T,<:FaerVector{T}}
        @test Matrix(kron(V', W'))==kron(v', w')
        @test kron(transpose(V), transpose(W)) isa
              LinearAlgebra.Transpose{T,<:FaerVector{T}}
        @test Matrix(kron(transpose(V), transpose(W)))==kron(transpose(v), transpose(w))
        @test Matrix(kron(V', transpose(W)))==kron(collect(v'), collect(transpose(w)))
        @test Matrix(kron(V', w'))==kron(v', w')
        @test Matrix(kron(v', W'))==kron(v', w')
        @test kron(V, W') isa FaerMatrix{T}
        @test Matrix(kron(V, W'))==kron(v, w')
        @test Matrix(kron(V', W))==kron(v', w)
        @test Matrix(kron(A, V))==kron(a, v)
        @test Matrix(kron(V, A))==kron(v, a)
        @test Matrix(kron(view(A, 2, :), B))==kron(a[2, :], b)
        R=FaerRow(v)
        @test kron(R, FaerRow(w)) isa FaerRow{T}
        @test Matrix(kron(R, FaerRow(w)))==kron(reshape(v, 1, :), reshape(w, 1, :))
        @test Matrix(kron(R, B))==kron(reshape(v, 1, :), b)
        # Empty operands.
        @test size(kron(view(A, 1:0, :), B))==(0, 10)
        @test size(kron(FaerVector(T[]), W))==(0,)
    end
    # Promotion converts only the operand of the other scalar type.
    A=FaerMatrix(Float32[1 2; 3 4])
    B=FaerMatrix(ComplexF64[1+im 2; 3 4-im])
    @test kron(A, B) isa FaerMatrix{ComplexF64}
    @test Matrix(kron(A, B))≈kron(Float32[1 2; 3 4], ComplexF64[1+im 2; 3 4-im])
    @test Matrix(kron(view(A, :, 2:-1:1), [1 2; 3 4]))==kron(Float32[2 1; 4 3], [1 2; 3 4])
    @test kron(view(A, :, 1), FaerVector([1.0, 2.0])) isa FaerVector{Float64}
    @test_throws ArgumentError kron(A, BigFloat[1 2; 3 4])
    # Closed owners and invalidated views throw.
    C=FaerMatrix(rand(2, 2))
    v=view(C, :, 1:2)
    close(C)
    @test_throws ArgumentError kron(C, A)
    @test_throws ArgumentError kron(v, A)
end

@testset "kron! into native destinations" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        a=_kron_data(T, 2, 3)
        b=_kron_data(T, 3, 2, 1)
        A=FaerMatrix(a)
        B=FaerMatrix(b)
        C=FaerMatrix(zeros(T, 6, 6))
        @test kron!(C, A, B)===C
        @test Matrix(C)==kron(a, b)
        C=FaerMatrix(zeros(T, 9, 4))
        kron!(C, A', view(B, 3:-1:1, :))
        @test Matrix(C)==kron(a', b[3:-1:1, :])
        C=FaerMatrix(zeros(T, 4, 9))
        kron!(C, faerview(a), b')
        @test Matrix(C)==kron(a, b')
        # A destination view, including a reversed and a conjugated layout.
        D=FaerMatrix(zeros(T, 8, 9))
        kron!(view(D, 2:7, 7:-1:2), A, B)
        expected=zeros(T, 8, 9)
        expected[2:7, 7:-1:2]=kron(a, b)
        @test Matrix(D)==expected
        # Operands sharing the destination's owner are snapshot first.
        E=FaerMatrix(zeros(T, 4, 4))
        E[1:2, 1:2]=a[:, 1:2]
        e=Matrix(E)
        kron!(E, view(E, 1:2, 1:2), view(E, 1:2, 1:2))
        @test Matrix(E)==kron(e[1:2, 1:2], e[1:2, 1:2])
        S=FaerMatrix(T[1 2; 3 4])
        F=FaerMatrix(zeros(T, 4, 4))
        kron!(view(F, :, :), S, S')
        @test Matrix(F)==kron(T[1 2; 3 4], T[1 2; 3 4]')
        G=FaerMatrix(T[1 2; 3 4])
        @test_throws DimensionMismatch kron!(G, G, G)
        @test Matrix(G)==T[1 2; 3 4]
        # Vector destinations.
        v=_kron_vector_data(T, 2)
        w=_kron_vector_data(T, 3, 1)
        X=FaerVector(zeros(T, 6))
        kron!(X, FaerVector(v), w)
        @test Vector(X)==kron(v, w)
        Y=FaerVector(zeros(T, 9))
        kron!(view(Y, 8:-1:3), FaerVector(v), faerview(w))
        @test Vector(Y)[8:-1:3]==kron(v, w)
        Z=FaerVector(T[1, 2, 0, 0])
        kron!(Z, view(Z, 1:2), view(Z, 1:2))
        @test Vector(Z)==kron(T[1, 2], T[1, 2])
    end
    # Destination scalar differs from the operands' promotion.
    C=FaerMatrix(zeros(ComplexF64, 4, 4))
    kron!(C, FaerMatrix([1.0 2; 3 4]), FaerMatrix(Float32[1 0; 0 1]))
    @test Matrix(C)==kron([1.0 2; 3 4], [1.0 0; 0 1])
    @test_throws DimensionMismatch kron!(
        FaerMatrix(zeros(3, 3)),
        FaerMatrix(ones(2, 2)),
        [1.0],
    )
end

@testset "Native sparse Kronecker products" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        s1=sparse([1, 3, 2, 4, 1], [1, 1, 2, 3, 3], T[1, 2, 3, 4, 5], 4, 3)
        s2=sparse([2, 1, 3], [1, 2, 2], T[6, 7, 0], 3, 2)  # one stored zero
        T<:Complex && (s1=s1+im*s1)
        expected=kron(s1, s2)
        for f1 in (:csc, :csr), f2 in (:csc, :csr)
            S1=FaerSparseMatrix(s1; format = f1)
            S2=FaerSparseMatrix(s2; format = f2)
            r=kron(S1, S2)
            @test r isa FaerSparseMatrix{T}
            @test r.format===f1
            @test nnz(r)==nnz(s1)*nnz(s2)
            @test SparseMatrixCSC(r)==expected
            @test nnz(SparseMatrixCSC(r))==nnz(expected)
            @test SparseMatrixCSC(kron(S1', S2))==kron(s1', s2)
            @test kron(S1', S2).format===(f1===:csc ? :csr : :csc)
            @test SparseMatrixCSC(kron(transpose(S1), S2'))==kron(transpose(s1), s2')
        end
        S1=FaerSparseMatrix(s1)
        # Borrowed Julia CSC operands, including one view used twice.
        b1=faerview(s1)
        @test SparseMatrixCSC(kron(b1, b1))==kron(s1, s1)
        @test SparseMatrixCSC(kron(b1, FaerSparseMatrix(s2)))==expected
        @test SparseMatrixCSC(kron(S1, faerview(s2)))==expected
        @test SparseMatrixCSC(kron(b1', s2))==kron(s1', s2)
        @test SparseMatrixCSC(kron(s1, FaerSparseMatrix(s2)))==expected
        @test kron(b1, s2) isa FaerSparseMatrix{T}
        # Mixed sparse and dense: SparseArrays sparsifies the dense operand.
        d=T[1 0; 0 2]
        for D in (FaerMatrix(d), d, faerview(d), view(FaerMatrix(T[0 1; 2 0]), :, 2:-1:1))
            r=kron(S1, D)
            @test r isa FaerSparseMatrix{T}
            @test SparseMatrixCSC(r)==kron(s1, Matrix(D))
            @test nnz(r)==nnz(kron(s1, Matrix(D)))
            @test SparseMatrixCSC(kron(D, S1))==kron(Matrix(D), s1)
            @test SparseMatrixCSC(kron(D', b1))==kron(Matrix(D)', s1)
        end
        v=T[1, 0, 3]
        @test SparseMatrixCSC(kron(S1, FaerVector(v)))==kron(s1, v)
        @test SparseMatrixCSC(kron(FaerVector(v)', S1))==kron(v', s1)
        sv=sparsevec(v)
        @test SparseMatrixCSC(kron(S1, sv))==kron(s1, sv)
        @test SparseMatrixCSC(kron(sv', b1))==kron(sv', s1)
        @test size(kron(FaerSparseMatrix(spzeros(T, 0, 2)), S1))==(0, 6)
    end
    # Promotion across sparse scalar types.
    a=FaerSparseMatrix(sparse(Float32[1 0; 2 3]))
    b=FaerSparseMatrix(sparse(ComplexF64[0 1+im; 1 0]))
    @test kron(a, b) isa FaerSparseMatrix{ComplexF64}
    sa=sparse(Float32[1 0; 2 3])
    @test SparseMatrixCSC(kron(a, b))==kron(sa, sparse(ComplexF64[0 1+im; 1 0]))
    @test SparseMatrixCSC(kron(a, [1 2; 0 1]))==kron(sa, sparse([1 2; 0 1]))
    @test_throws ArgumentError kron(a, sparse(BigFloat[1 0; 0 1]))
    c=FaerSparseMatrix(sparse([1.0 0; 0 1]))
    close(c)
    @test_throws ArgumentError kron(c, a)
end

@testset "Kronecker allocations and ambiguities" begin
    a=rand(40, 40)
    A=FaerMatrix(a)
    V=view(A, 1:2:40, 40:-1:1)
    kron(V, V)
    # The product is the only large allocation (the owner lives in Rust).
    @test (@allocated kron(V, V))<10_000
    J=rand(40, 40)
    kron(A, J)
    @test (@allocated kron(A, J))<10_000
    C=FaerMatrix(zeros(1600, 800))
    kron!(C, A, V')
    @test (@allocated kron!(C, A, V'))<10_000
    s=FaerSparseMatrix(sprand(60, 60, 0.05))
    kron(s, s)
    @test (@allocated kron(s, s))<10_000
    @test isempty(
        detect_ambiguities(Colmat, Base, LinearAlgebra, SparseArrays; recursive = true),
    )
end
