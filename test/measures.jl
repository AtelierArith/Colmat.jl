using Colmat, Test, LinearAlgebra

const MEASURE_TYPES=(Float32, Float64, ComplexF32, ComplexF64)
_measure_sample(::Type{T}, n) where {T} =
    T<:Complex ?
    T.(reshape(1:(n*n), n, n) ./ n .+ 0.25im .* reshape((n*n):-1:1, n, n) ./ n) + T(n)*I :
    T.(reshape(1:(n*n), n, n) ./ n .- n/3) + T(n)*I
_close(x, y) = isapprox(x, y; rtol = 100eps(real(float(typeof(y)))), atol = 0)

@testset "Native determinants of views and transposes" begin
    team=FaerTeam(2)
    for T in MEASURE_TYPES, n in (0, 1, 4)
        a=_measure_sample(T, n)
        A=FaerMatrix(a)
        ref=det(a)
        @test det(A) isa T
        @test _close(det(A), ref)
        @test _close(det(A; team), ref)
        @test _close(det(view(A, :, :)), ref)
        @test _close(det(transpose(A)), ref)
        @test _close(det(A'), conj(ref))
        @test _close(det(view(A, n:-1:1, :)), det(a[n:-1:1, :]))
        big=FaerMatrix(zeros(T, 2n+1, 2n+1))
        copyto!(view(big, 1:2:2n, 2:2:2n), a)
        V=view(big, 1:2:2n, 2:2:2n)
        @test _close(det(V), ref)
        @test _close(det(V'; team), conj(ref))
        @test _close(tr(V), tr(a))
        @test _close(tr(V'), conj(tr(a)))
        @test Matrix(A) == a
        foreach(close, (A, big))
    end
    @test_throws DimensionMismatch det(FaerMatrix(ones(2, 3)))
    @test_throws DimensionMismatch det(view(FaerMatrix(ones(3, 3)), 1:2, :))
    @test_throws DimensionMismatch tr(view(FaerMatrix(ones(3, 3)), 1:2, :))
    closed=FaerTeam(2)
    close(closed)
    @test_throws ArgumentError det(FaerMatrix(ones(2, 2)); team = closed)
    close(team)
end

@testset "Structured determinants, traces and log-determinants" begin
    W=(LowerTriangular, UpperTriangular, UnitLowerTriangular, UnitUpperTriangular)
    for T in MEASURE_TYPES, n in (0, 1, 4), w in W
        a=_measure_sample(T, n)
        A=FaerMatrix(a)
        for X in (A, view(A, :, :), transpose(A), A')
            ref=w(Matrix(X))
            @test _close(det(w(X)), det(ref))
            @test _close(tr(w(X)), tr(ref))
            l, s=logabsdet(w(X))
            rl, rs=logabsdet(ref)
            @test _close(l, rl)
            @test _close(s, rs)
            if T<:Real && any(<(0), diag(ref))
                isodd(
                    count(
                        <(0),
                        w in (UnitLowerTriangular, UnitUpperTriangular) ? T[] : diag(ref),
                    ),
                ) && @test_throws DomainError logdet(w(X))
            else
                @test _close(logdet(w(X)), logdet(ref))
            end
        end
        close(A)
    end
end

@testset "Native diagonal determinants" begin
    for T in MEASURE_TYPES
        for d in (T[2, 3, 4], T[2, -3, 4], T[1, 0, 2], T[])
            T<:Complex && !isempty(d) && (d=d .+ T(0.5im))
            D=FaerDiagonal(d)
            # LinearAlgebra rejects empty Diagonal reductions; Colmat follows the
            # empty dense convention (det 1, trace 0).
            ref=isempty(d) ? zeros(T, 0, 0) : Diagonal(d)
            @test _close(det(D), det(ref))
            @test _close(tr(D), tr(ref))
            l, s=logabsdet(D)
            rl, rs=logabsdet(ref)
            @test isequal(l, rl) || _close(l, rl)
            @test _close(s, rs)
            if T<:Real && isodd(count(<(0), d))
                @test_throws DomainError logdet(D)
            else
                @test isequal(logdet(D), logdet(ref)) || _close(logdet(D), logdet(ref))
            end
            close(D)
        end
    end
    D=FaerDiagonal([1.0, 2.0])
    close(D)
    @test_throws ArgumentError det(D)
end

@testset "Log-determinants of dense views and transposes" begin
    for T in MEASURE_TYPES
        a=_measure_sample(T, 3)
        A=FaerMatrix(a)
        for (X, x) in ((view(A, :, :), a), (transpose(A), transpose(a)), (A', a'))
            l, s=logabsdet(X)
            rl, rs=logabsdet(x)
            @test _close(l, rl)
            @test _close(s, rs)
        end
        @test Matrix(A) == a
        close(A)
    end
end
