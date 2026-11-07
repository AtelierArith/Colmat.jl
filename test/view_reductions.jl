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

    @testset "in-place reductions into native and Julia destinations" begin
        rng=MersenneTwister(970)
        for T in (Float32, Float64, ComplexF64)
            pairs, owners=_operands(rng, T, 5, 4)
            fs=T<:Real ? (sum!, minimum!, maximum!, extrema!) : (sum!,)
            try
                for (a, J) in pairs
                    shapes=ndims(J)==1 ? ((1,), size(J)) :
                           ((1, size(J, 2)), (size(J, 1), 1), (1, 1), size(J), (size(J, 1),))
                    for f! in fs, shape in shapes, init in (true, false)
                        S=f! === extrema! ? Tuple{T,T} : T
                        start=f! === extrema! ? [(T(-1), T(1)) for _ in 1:prod(shape)] :
                              randn(rng, T, prod(shape))
                        start=reshape(start, shape)
                        expected=f!(copy(start), J; init)
                        dests=Any[copy(start)]
                        if S===T
                            push!(dests, faerview(copy(start)))
                            push!(dests, length(shape)==1 ? FaerVector(vec(start)) : FaerMatrix(start))
                        end
                        for r in dests
                            try
                                @test f!(r, a; init)===r
                                @test _same(Array(r), expected, 512eps(real(T)))
                            finally
                                r isa Union{FaerMatrix,FaerVector} && close(r)
                            end
                        end
                    end
                end
                native, x=owners
                # Julia sources into native destinations.
                J=randn(rng, T, 5, 4)
                r=FaerMatrix(zeros(T, 1, 4))
                @test Array(sum!(r, J))≈sum(J; dims = 1)
                @test Array(sum!(r, J; init = false))≈2sum(J; dims = 1)
                close(r)
                # A destination viewing the source sees the reduction of the old values.
                A=Array(native)
                @test Array(sum!(view(native, 1:1, :), native))≈sum(A; dims = 1)
                # Base handles (and reports) shape errors and empty sources.
                @test_throws DimensionMismatch sum!(zeros(T, 3), native)
                @test_throws DimensionMismatch sum!(zeros(T, 1, 1, 1), native)
                @test sum!(zeros(T, 1, 3), faerview(zeros(T, 0, 3)))==zeros(T, 1, 3)
                @test sum!(zeros(ComplexF64, 1, 4), native)≈sum(Array(native); dims = 1)
            finally
                foreach(close, owners)
            end
        end
        for T in (Float32, Float64)
            J=T[0 -0.0 NaN; -0.0 0 1; 2 -Inf Inf]
            native=FaerMatrix(zeros(T, size(J)))
            for i in eachindex(J)
                native[i]=J[i]
            end
            try
                for f! in (minimum!, maximum!), dims in ((1, 3), (3, 1)), init in (true, false)
                    start=reshape(T[NaN, -0.0, 0], dims)
                    @test isequal(f!(copy(start), native; init), f!(copy(start), J; init))
                end
            finally
                close(native)
            end
        end
        @test_throws MethodError minimum!(zeros(ComplexF64, 1, 2), faerview(rand(ComplexF64, 2, 2)))
    end

    @testset "allocations" begin
        n=200
        J=randn(n, n)
        native=FaerMatrix(J)
        x=FaerVector(randn(n*n))
        y=FaerVector(randn(n*n))
        rows=FaerMatrix(zeros(1, n÷2))
        cols=zeros(n)
        pairs=Matrix{Tuple{Float64,Float64}}(undef, 1, n)
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
            # In-place forms also stage the O(length(r)) result once (n*n*8
            # bytes for a copy of the source).
            for f in (
                ()->sum!(rows, V),
                ()->minimum!(rows, V; init = false),
                ()->maximum!(cols, native),
                ()->sum!(cols, B),
                ()->extrema!(pairs, native),
            )
                f()
                @test @allocated(f())<64*n+8192
            end
        finally
            close(rows)
            close(native)
            close(x)
            close(y)
        end
    end

    @test isempty(
        Test.detect_ambiguities(Colmat, Base, LinearAlgebra, SparseArrays; recursive = true),
    )
end
