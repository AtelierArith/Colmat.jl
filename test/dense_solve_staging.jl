using Colmat, Test, LinearAlgebra

const DSS_TYPES=(Float32, Float64, ComplexF32, ComplexF64)
_dss_matrix(::Type{T}, n) where {T} =
    T.([1/(i+j) + (i==j)*n + (T<:Complex ? im*sin(i-j) : 0) for i = 1:n, j = 1:n])

@testset "Dense factor solves write native destinations without Julia staging" begin
    n=600
    a=_dss_matrix(Float64, n)
    F=lu(FaerMatrix(a))
    B=FaerMatrix(reshape(sin.(1:(4n)), n, 4))
    Bt=FaerMatrix(reshape(cos.(1:(4n)), 4, n))
    Y=FaerMatrix(zeros(n, 4))
    Yt=FaerMatrix(zeros(4, n))
    rhs=n*4*sizeof(Float64)
    for f in (
        ()->ldiv!(F, copy(B)),
        ()->ldiv!(Y, F, B),
        ()->rdiv!(copy(Bt), F),
        ()->rdiv!(Yt, Bt, F),
        ()->ldiv!(Y, F', B),
        ()->rdiv!(Yt, Bt, transpose(F)),
    )
        f()
        # Bounded by the RHS size; the old paths staged several RHS copies
        # (and right solves copied every element generically).
        @test @allocated(f()) < 2rhs
    end
    @test Matrix(ldiv!(Y, F, B)) ≈ a\Matrix(B)
    @test Matrix(rdiv!(Yt, Bt, F)) ≈ Matrix(Bt)/a
    @test Matrix(ldiv!(Y, F', B)) ≈ a'\Matrix(B)
    @test Matrix(rdiv!(Yt, Bt, transpose(F))) ≈ Matrix(Bt)/transpose(a)
    foreach(close, (F, B, Bt, Y, Yt))
end

@testset "Native and Julia operand combinations" begin
    for T in DSS_TYPES, kind in (:lu, :qr, :cholesky)
        n=7
        a=_dss_matrix(T, n)
        kind===:cholesky && (a=a*a'+n*I)
        A=FaerMatrix(a)
        F=kind===:lu ? lu(A) : kind===:qr ? qr(A) : cholesky(Hermitian(A))
        b=T.(reshape(1:(3n), n, 3)) .+ T(0.5)
        tol=1000eps(real(T))
        ref=a\b
        rref=permutedims(b)/a
        for B in (b, FaerMatrix(b), view(FaerMatrix(b), :, :), faerview(copy(b)))
            for D in (
                zeros(T, n, 3),
                FaerMatrix(zeros(T, n, 3)),
                view(FaerMatrix(zeros(T, n + 2, 3)), 2:(n+1), :),
            )
                @test ldiv!(D, F, B) === D
                @test isapprox(Array(D), ref; rtol = tol)
            end
        end
        for B in (permutedims(b), FaerMatrix(permutedims(b)))
            D=FaerMatrix(zeros(T, 3, n))
            @test isapprox(Matrix(rdiv!(D, B, F)), rref; rtol = tol)
            E=copy(B)
            rdiv!(E, F)
            @test isapprox(Array(E), rref; rtol = tol)
        end
        # Two-argument ldiv! overwrites its right-hand side in place.
        X=FaerMatrix(copy(b))
        @test ldiv!(F, X) === X
        @test isapprox(Matrix(X), ref; rtol = tol)
        v=FaerVector(b[:, 1])
        ldiv!(F, v)
        @test isapprox(Vector(v), ref[:, 1]; rtol = tol)
        jv=b[:, 1]
        ldiv!(F, jv)
        @test isapprox(jv, ref[:, 1]; rtol = tol)
        # A narrower destination still converts through the staging path.
        if T===Float64
            D=zeros(Float32, n, 3)
            ldiv!(D, F, FaerMatrix(b))
            @test D ≈ Float32.(ref)
        end
        @test_throws DimensionMismatch ldiv!(FaerMatrix(zeros(T, n, 2)), F, FaerMatrix(b))
        @test Matrix(A) == a
        foreach(close, (A, F))
    end
end
