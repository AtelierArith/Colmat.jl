using Colmat, Test, LinearAlgebra

const CALLBACK_TYPES=(Float32, Float64, ComplexF32, ComplexF64)

@testset "Reused callback buffers" begin
    for T in CALLBACK_TYPES
        n=40
        d=T.(range(2, 3; length = n))
        T<:Complex && (d=d .+ T(0.1im))
        fw!(y, x) = (y .= d .* x; y)
        bw!(y, x) = (y .= conj.(d) .* x; y)
        owned=FaerLinearOperator(fw!, bw!, n, n; T)
        reused=FaerLinearOperator(fw!, bw!, n, n; T, buffers = :reuse)
        x=FaerMatrix(T.(reshape(1:(3n), n, 3)))
        for op in (owned, reused), k in (1, 3)
            X=FaerMatrix(Matrix(x)[:, 1:k])
            @test Matrix(op*X) ≈ Diagonal(d)*Matrix(X)
            @test Matrix(op'*X) ≈ Diagonal(conj.(d))*Matrix(X)
            close(X)
        end
        b=FaerVector(ones(T, n))
        h=FaerLinearOperator(
            (y, x)->(y .= abs.(d) .* x; y),
            (y, x)->(y .= abs.(d) .* x; y),
            n,
            n;
            T,
            buffers = :reuse,
        )
        r=cg(h, b)
        @test r.converged
        @test Vector(r.x) ≈ ones(T, n) ./ abs.(d)
        foreach(close, (owned, reused, x, b, h))
    end
end

@testset "Reused buffers stop per-application allocation" begin
    n=50_000
    d=collect(range(2.0, 3.0; length = n))
    fw!(y, x) = (y .= d .* x; y)
    owned=FaerLinearOperator(fw!, fw!, n, n)
    reused=FaerLinearOperator(fw!, fw!, n, n; buffers = :reuse)
    b=FaerVector(ones(n))
    cg(owned, b)
    cg(reused, b)
    ao=@allocated cg(owned, b)
    ar=@allocated cg(reused, b)
    @test ar < ao ÷ 5
    @test Vector(cg(reused, b).x) ≈ 1 ./ d
    foreach(close, (owned, reused, b))
end

@testset "Recursive application and invalid buffer modes" begin
    n=6
    inner=Ref{Any}(nothing)
    depth=Ref(0)
    function fw!(y, x)
        if depth[]==0
            depth[]+=1
            try
                # Re-enter the same operator; the in-use pair is not shared.
                z=inner[]*FaerMatrix(copy(x))
                y.=2 .* Matrix(z)
                close(z)
            finally
                depth[]-=1
            end
        else
            y.=x
        end
        return y
    end
    op=FaerLinearOperator(fw!, fw!, n, n; buffers = :reuse)
    inner[]=op
    X=FaerMatrix(reshape(1.0:(2n), n, 2))
    @test Matrix(op*X) == 2 .* Matrix(X)
    @test Matrix(op*X) == 2 .* Matrix(X)
    @test_throws ArgumentError FaerLinearOperator(fw!, fw!, n, n; buffers = :borrow)
    # An incomplete write is still rejected with reused buffers.
    bad=FaerLinearOperator((y, x)->y, (y, x)->y, n, n; buffers = :reuse)
    @test_throws ArgumentError bad*X
    @test_throws ArgumentError bad*X
    foreach(close, (op, X, bad))
end
