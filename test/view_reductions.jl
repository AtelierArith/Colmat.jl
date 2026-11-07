using Test, Colmat, LinearAlgebra, SparseArrays, Random

_close(x, y, tol) = isequal(x, y) || isapprox(x, y; rtol = tol, atol = tol)
function _same(x, y, tol)
    x isa Number && return y isa Number && _close(x, y, tol)
    return size(x)==size(y) && all(_close(a, b, tol) for (a, b) in zip(x, y))
end
# Every native operand family paired with the Julia array it reads.
function _operands(rng, T, m, n)
    A=randn(rng, T, m, n)
    v=randn(rng, T, m*n)
    native=FaerMatrix(A)
    x=FaerVector(v)
    J=copy(A)
    pairs=Any[
        (native, A),
        (x, v),
        (FaerRow(reshape(v, 1, :)), reshape(v, 1, :)),
        (view(native, m:-1:1, 1:2:n), A[m:-1:1, 1:2:n]),
        (view(native, :, n:-1:1), A[:, n:-1:1]),
        (view(x, length(v):-2:1), v[length(v):-2:1]),
        (faerview(J), J),
        (faerview(view(J, m:-1:1, n:-1:1)), J[m:-1:1, n:-1:1]),
        (faerview(view(J, 1:2:m, :)), J[1:2:m, :]),
    ]
    m>0 && push!(pairs, (view(native, max(1, m-1), :), A[max(1, m-1), :]))
    return pairs, (native, x)
end

@testset "Native view reductions" begin
    rng=MersenneTwister(97)
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        R=real(T)
        tol=512eps(R)
        for (m, n) in ((5, 4), (1, 3), (3, 1), (0, 3), (3, 0), (0, 0))
            pairs, owners=_operands(rng, T, m, n)
            try
                for (a, J) in pairs
                    for dims in (:, 1, 2, (1, 2), 3, (), (2, 3))
                        @test _same(sum(a; dims), sum(J; dims), tol)
                        @test eltype(sum(a; dims))===T
                    end
                    for p in (2, 1, Inf, 0, -Inf, 3, 0.5, -1, -2.5, 1.5, 7//2)
                        @test _same(norm(a, p), norm(J, p), tol)
                        @test norm(a, p) isa R
                    end
                    @test _same(norm(a), norm(J), tol)
                    @test all(isfinite, a)===all(isfinite, J)
                    @test any(isnan, a)===any(isnan, J)
                    @test all(isfinite, a; dims = 1)==all(isfinite, J; dims = 1)
                    if T<:Real
                        for f in (minimum, maximum, extrema), dims in (:, 1, 2, (1, 2), 3, ())
                            expected=try
                                f(J; dims)
                            catch e
                                e
                            end
                            if expected isa Exception
                                @test_throws typeof(expected) f(a; dims)
                            else
                                @test isequal(f(a; dims), expected)
                                @test isequal(f(identity, a; dims), expected)
                            end
                        end
                        @test isequal(minimum(abs, a; init = R(Inf)), minimum(abs, J; init = R(Inf)))
                        @test isequal(extrema(a; init = (R(Inf), R(-Inf))), extrema(J; init = (R(Inf), R(-Inf))))
                        @test isequal(maximum(a; dims = 1, init = R(-Inf)), maximum(J; dims = 1, init = R(-Inf)))
                    else
                        # Julia compares complex entries only when there are two.
                        length(J)>1 && @test_throws MethodError minimum(a)
                    end
                    @test _same(sum(a; init = T(1)), sum(J; init = T(1)), tol)
                end
                for (a, J) in pairs, (b, K) in pairs
                    if length(J)==length(K)
                        @test _same(dot(a, b), dot(vec(J), vec(K)), tol)
                    else
                        @test_throws DimensionMismatch dot(a, b)
                    end
                end
            finally
                foreach(close, owners)
            end
        end
    end

    @testset "promotion, NaN and signed zero" begin
        for (S, U) in ((Float32, Float64), (Float32, ComplexF64), (Float64, ComplexF64), (ComplexF32, ComplexF64))
            x=randn(S, 6)
            y=randn(U, 6)
            nx=FaerVector(x)
            try
                result=dot(view(nx, 6:-1:1), faerview(y))
                @test result isa promote_type(S, U)
                @test result≈dot(reverse(x), y)
                @test dot(faerview(y), nx)≈dot(y, x)
            finally
                close(nx)
            end
        end
        z=ComplexF64[1+2im, 3-1im, -2im]
        w=ComplexF64[2-1im, 1im, 4]
        @test dot(faerview(z), faerview(w))≈sum(conj.(z) .* w)
        for T in (Float32, Float64)
            J=T[0 -0.0 NaN; -0.0 0 1; 2 -Inf Inf]
            # Constructors reject non-finite input; element writes do not.
            native=FaerMatrix(zeros(T, size(J)))
            for i in eachindex(J)
                native[i]=J[i]
            end
            try
                for (a, K) in ((native, J), (view(native, 3:-1:1, :), J[3:-1:1, :]), (faerview(J), J))
                    for f in (minimum, maximum, extrema), dims in (:, 1, 2)
                        @test isequal(f(a; dims), f(K; dims))
                    end
                    for p in (0, -Inf, Inf, 3, -3, 0.5, 1, 2)
                        @test isequal(norm(a, p), norm(K, p))
                    end
                    @test all(isfinite, a)===false
                    @test any(isnan, a)===true
                    @test isequal(sum(a; dims = 2), sum(K; dims = 2))
                end
                clean=T[0 -0.0; 3 -4]
                for p in (0, -Inf, 3, -3, 0.5, -1)
                    @test isequal(norm(faerview(clean), p), norm(clean, p)) ||
                          norm(faerview(clean), p)≈norm(clean, p)
                end
                @test norm(faerview(clean), 0)===T(2)
                big=fill(floatmax(T)/2, 4)
                @test norm(faerview(big), 3)≈norm(big, 3)
                @test isfinite(norm(faerview(big), 3))
                tiny=fill(nextfloat(zero(T)), 4)
                @test norm(faerview(tiny), -3)≈norm(tiny, -3)
            finally
                close(native)
            end
        end
        c=ComplexF64[NaN+0im, 1im, 0]
        @test all(isfinite, faerview(c))===false
        @test any(isnan, faerview(c))===true
        @test norm(faerview(c), 0)===2.0
        @test isnan(norm(faerview(c), -Inf))
    end

    @testset "closed and invalidated inputs" begin
        native=FaerMatrix(rand(4, 3))
        x=FaerVector(rand(5))
        v=view(native, 1:2, :)
        w=view(x, 1:3)
        close(v)
        for f in (sum, minimum, maximum, extrema, norm, a->norm(a, 3), a->all(isfinite, a), a->any(isnan, a))
            @test_throws ArgumentError f(v)
        end
        @test_throws ArgumentError dot(v, v)
        resize!(x, 6)
        @test_throws InvalidStateException sum(w)
        @test_throws InvalidStateException norm(w, 3)
        @test_throws InvalidStateException dot(w, faerview(rand(3)))
        close(native)
        close(x)
        @test_throws ArgumentError norm(native, 3)
        @test_throws ArgumentError minimum(native; dims = 1)
        @test_throws ArgumentError dot(native, faerview(rand(4, 3)))
        @test_throws ArgumentError norm(x, 0)
    end

    @testset "allocations" begin
        n=200
        J=randn(n, n)
        native=FaerMatrix(J)
        x=FaerVector(randn(n*n))
        y=FaerVector(randn(n*n))
        try
            V=view(native, n:-1:1, 1:2:n)
            B=faerview(J)
            xv=view(x, n*n:-1:1)
            calls=(
                ()->norm(native, 3),
                ()->norm(x, -Inf),
                ()->norm(V, 3),
                ()->norm(B, 0),
                ()->dot(xv, y),
                ()->dot(xv, faerview(vec(J))),
                ()->sum(V),
                ()->sum(V; dims = 1),
                ()->minimum(V),
                ()->extrema(native; dims = 2),
                ()->all(isfinite, V),
                ()->any(isnan, B),
            )
            for f in calls
                f()
                # Descriptors, scalar Refs and the reduced output only.
                @test @allocated(f())<16*n+4096
            end
        finally
            close(native)
            close(x)
            close(y)
        end
    end

    @test isempty(
        Test.detect_ambiguities(Colmat, Base, LinearAlgebra, SparseArrays; recursive = true),
    )
end
