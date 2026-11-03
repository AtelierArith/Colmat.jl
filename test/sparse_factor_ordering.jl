using Test, Colmat, LinearAlgebra, SparseArrays

@testset "COLAMD in retained simplicial LU/QR" begin
    changed_order=false
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        format in (:csc, :csr),
        kind in (:lu, :qr),
        n in (0, 1, 7, 23),
        tuned in (false, true)

        m=kind===:qr ? n+3 : n
        A=zeros(T, m, n)
        for j = 1:n
            A[j, j]=T(5+j)
            A[1, j]=T(0.2j)
            A[j, 1]+=T(0.1j)
            j<n && (A[j+1, j]=T(0.3))
            if T<:Complex
                A[j, j]+=T(0.4im)
            end
            m>n && (A[m, j]=T(0.15j))
        end
        a=FaerSparseMatrix(sparse(A); format)
        controls=tuned ? (; dense_row = -1.0, dense_col = 0.0, aggressive = false) : (;)
        expected=colamd(a; controls...)
        p=Vector(expected)
        changed_order |= p!=collect(1:n)
        plan=FaerSparseSymbolic(
            a;
            kind,
            algorithm = :simplicial,
            ordering = :colamd,
            controls...,
        )
        F=factorize(plan, a)
        close(plan)
        close(expected)
        component=kind===:lu ? F.q : F.pcol
        @test Vector(component)==p
        close(component)
        @test Matrix(F)≈A atol=1e-3 rtol=1e-3
        if kind===:lu
            @test Matrix(F.L*F.U)≈A[Vector(F.p), p] atol=1e-3
        else
            @test Matrix(F.Q)*Matrix(F.R)≈A[:, p] atol=1e-3
        end
        B=reshape(T.(1:(m*2)), m, 2)
        @test Matrix(F\B)≈A\B atol=1e-3 rtol=1e-3
        # The public convenience constructor must forward the same controls.
        G=kind===:lu ? lu(a; algorithm = :simplicial, ordering = :colamd, controls...) :
          qr(a; algorithm = :simplicial, ordering = :colamd, controls...)
        component=kind===:lu ? G.q : G.pcol
        @test Vector(component)==p
        close(component)
        close(G)
        # Updates retain the symbolic permutation after the input/plan is closed.
        b=FaerSparseMatrix(sparse(2A); format)
        close(a)
        if kind===:lu
            @test lu!(F, b)===F
        else
            @test qr!(F, b)===F
        end
        close(b)
        component=kind===:lu ? F.q : F.pcol
        @test Vector(component)==p
        close(component)
        @test Matrix(F)≈2A atol=1e-3 rtol=1e-3
        C=reshape(T.(1:(n*2)), n, 2)
        @test Matrix(adjoint(F)\C)≈pinv(adjoint(2A))*C atol=1e-3 rtol=1e-3
        H=copy(F)
        close(F)
        @test Matrix(H)≈2A atol=1e-3 rtol=1e-3
        close(H)
    end
    @test changed_order
end

@testset "COLAMD controls and sparse ownership errors" begin
    A=FaerSparseMatrix(sparse([4.0 1; 1 3]))
    P=FaerSparsity(A)
    for kind in (:lu, :qr)
        for threshold in (NaN, Inf, -Inf)
            @test_throws ArgumentError FaerSparseSymbolic(
                P;
                kind,
                algorithm = :simplicial,
                ordering = :colamd,
                dense_row = threshold,
            )
            @test_throws ArgumentError FaerSparseSymbolic(
                P;
                kind,
                algorithm = :simplicial,
                ordering = :colamd,
                dense_col = threshold,
            )
        end
        if kind===:lu
            @test_throws ArgumentError FaerSparseSymbolic(P; kind, ordering = :colamd)
        else
            @test FaerSparseSymbolic(P; kind, ordering = :colamd) isa FaerSparseSymbolic
        end
        @test_throws ArgumentError FaerSparseSymbolic(
            P;
            kind,
            algorithm = :simplicial,
            ordering = :identity,
            dense_row = 0.0,
        )
        @test_throws ArgumentError FaerSparseSymbolic(
            P;
            kind,
            algorithm = :simplicial,
            ordering = :colamd,
            dense = 0.0,
        )
    end
    @test_throws ArgumentError FaerSparseSymbolic(P; kind = :cholesky, ordering = :colamd)
    @test_throws ArgumentError FaerSparseSymbolic(P; kind = :cholesky, dense_col = 0.0)
    close(P)
    @test_throws ArgumentError FaerSparseSymbolic(
        P;
        kind = :qr,
        algorithm = :simplicial,
        ordering = :colamd,
    )
    close(A)
    @test_throws ArgumentError lu(A; algorithm = :simplicial, ordering = :colamd)
    @test_throws ArgumentError qr(A; algorithm = :simplicial, ordering = :colamd)
end
