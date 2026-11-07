using Colmat, Test, LinearAlgebra, SparseArrays, Random

# Native Householder-family, compact-basis and operator products written
# straight into Julia, strided, borrowed (`faerview`) and native destinations.
_hd_bytes(f) = (f(); f(); @allocated f())
function _hd_sequence(rng, T, n, k)
    v=zeros(T, n, k)
    tau=zeros(real(T), k)
    for j = 1:k
        v[j, j]=1
        v[(j+1):end, j]=randn(rng, T, n-j)
        tau[j]=sum(abs2, v[:, j])/2
    end
    return FaerHouseholderSequence(v, tau; blocksize = 3)
end
function _hd_readonly(payload::Matrix{Float64})
    header="{'descr': '<f8', 'fortran_order': True, 'shape': $(size(payload)), }"
    pad=64-(10+length(header)+1)%64
    bytes=vcat(
        UInt8[0x93],
        Vector{UInt8}("NUMPY"),
        UInt8[0x01, 0x00],
        reinterpret(UInt8, [htol(UInt16(length(header)+pad+1))]),
        Vector{UInt8}(header*" "^pad*"\n"),
        reinterpret(UInt8, vec(payload)),
    )
    return bytes, npyborrow(bytes)
end
# Every destination kind receives `f(dest)`; returns the Julia results.
function _hd_destinations(f, c0)
    T=eltype(c0)
    m, n=size(c0)
    out=Matrix{T}[]
    c=copy(c0)
    @test f(c)===c
    push!(out, c)
    big=fill(T(7), m+3, 2n)
    v=view(big, (m+1):-1:2, 1:2:(2n))
    copyto!(v, c0)
    @test f(v)===v
    push!(out, Matrix(v))
    @test big[1, :]==fill(T(7), 2n) && all(==(T(7)), big[:, 2:2:(2n)])
    j=reverse(copy(c0); dims = 1)
    r=faerview(view(j, m:-1:1, :))
    @test f(r)===r
    push!(out, Matrix(r))
    g=FaerMatrix(c0)
    @test f(g)===g
    push!(out, Matrix(g))
    h=FaerMatrix(c0)
    hv=view(h, :, :)
    @test f(hv)===hv
    push!(out, Matrix(h))
    return out
end

@testset "Householder, sequence and compact Q products into destinations" begin
    rng=MersenneTwister(92)
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        tol=sqrt(eps(real(T)))*100
        n=7
        s=randn(rng, T, n, n)
        actions=(
            householder(randn(rng, T, n)),
            _hd_sequence(rng, T, n, 4),
            adjoint(_hd_sequence(rng, T, n, 5)),
            hessenberg(FaerMatrix(randn(rng, T, n, n)); compact = true).Q,
            tridiagonalize(FaerMatrix(s+s'); compact = true).Q,
            adjoint(hessenberg(FaerMatrix(randn(rng, T, n, n)); compact = true).Q),
        )
        for q in actions
            d=Matrix(q)
            b=randn(rng, T, n, 3)
            c0=randn(rng, T, n, 3)
            l=randn(rng, T, 3, n)
            l0=randn(rng, T, 3, n)
            alpha, beta=T(2), T(-0.5)
            for be in (zero(T), beta), x in (b, faerview(b), FaerMatrix(b))
                for got in _hd_destinations(c->mul!(c, q, x, alpha, be), c0)
                    @test got≈alpha*d*b+be*c0 rtol=tol
                end
            end
            for got in _hd_destinations(c->mul!(c, q, b), c0)
                @test got≈d*b rtol=tol
            end
            for got in _hd_destinations(c->mul!(c, l, q, alpha, beta), l0)
                @test got≈alpha*l*d+beta*l0 rtol=tol
            end
            for got in _hd_destinations(c->lmul!(q, c), b)
                @test got≈d*b rtol=tol
            end
            for got in _hd_destinations(c->rmul!(c, q), l)
                @test got≈l*d rtol=tol
            end
            # Vectors and Float64 coefficients for Float32 data.
            x=randn(rng, T, n)
            y=randn(rng, T, n)
            y0=copy(y)
            @test mul!(y, q, x, 2.0, 0.5)===y
            @test y≈2*d*x+y0/2 rtol=tol
            # Aliases: the source is the destination or overlaps it.
            c=copy(b)
            mul!(c, q, c, alpha, beta)
            @test c≈alpha*d*b+beta*b rtol=tol
            wide=randn(rng, T, n, 4)
            w0=copy(wide)
            mul!(view(wide, :, 1:3), q, view(wide, :, 2:4))
            @test wide[:, 1:3]≈d*w0[:, 2:4] rtol=tol
            # beta == 0 never reads C; alpha == 0 never reads B.
            c=fill(T(NaN), n, 3)
            mul!(c, q, b, alpha, 0)
            @test c≈alpha*d*b rtol=tol
            c=copy(c0)
            mul!(c, q, fill(T(NaN), n, 3), 0, beta)
            @test c≈beta*c0 rtol=tol
            # Native application allocates only constant FFI bookkeeping.
            c=copy(c0)
            @test _hd_bytes(()->mul!(c, q, b, alpha, beta)) < 4096
            @test _hd_bytes(()->lmul!(q, c)) < 4096
            r=faerview(c)
            @test _hd_bytes(()->lmul!(q, r)) < 4096
            # A complex product cannot be written into a real destination.
            if T<:Real
                z=householder(randn(rng, ComplexF64, n))
                c=copy(c0)
                @test_throws InexactError mul!(c, z, ComplexF64.(b))
                @test c==c0
            end
        end
    end
    # Destination writes do not depend on the problem size.
    rng=MersenneTwister(93)
    small, large=map((16, 128)) do n
        q=_hd_sequence(rng, Float64, n, n÷2)
        b, c=randn(rng, n, n), randn(rng, n, n)
        (_hd_bytes(()->mul!(c, q, b, 2.0, 3.0)), _hd_bytes(()->lmul!(q, faerview(c))))
    end
    @test large[1]<=small[1]+256 && large[2]<=small[2]+256
    # Read-only NPY destinations are rejected before any write.
    bytes, ro=_hd_readonly([1.0 2; 3 4])
    before=copy(bytes)
    for q in (householder([1.0, 2.0]), _hd_sequence(rng, Float64, 2, 1))
        @test_throws ArgumentError mul!(ro, q, [1.0 0; 0 1])
        @test_throws ArgumentError mul!(ro, q, [1.0 0; 0 1], 2, 3)
        @test_throws ArgumentError lmul!(q, ro)
    end
    @test bytes==before
end

@testset "Compact bidiagonal bases into destinations" begin
    rng=MersenneTwister(94)
    for T in (Float32, Float64, ComplexF32, ComplexF64), (m, n) in ((6, 4), (4, 6))
        tol=sqrt(eps(real(T)))*100
        f=bidiagonalize(FaerMatrix(randn(rng, T, m, n)); compact = true)
        for q in (f.U, f.V, adjoint(f.U), transpose(f.V))
            d=Matrix(q)
            b=randn(rng, T, size(q, 2), 3)
            c0=randn(rng, T, size(q, 1), 3)
            l=randn(rng, T, 2, size(q, 1))
            l0=randn(rng, T, 2, size(q, 2))
            alpha, beta=T(2), T(-0.5)
            for be in (zero(T), beta), x in (b, faerview(b))
                for got in _hd_destinations(c->mul!(c, q, x, alpha, be), c0)
                    @test got≈alpha*d*b+be*c0 rtol=tol
                end
            end
            for got in _hd_destinations(c->mul!(c, l, q, alpha, beta), l0)
                @test got≈alpha*l*d+beta*l0 rtol=tol
            end
            c=fill(T(NaN), size(c0))
            mul!(c, q, b, alpha, 0)
            @test c≈alpha*d*b rtol=tol
            c=copy(c0)
            @test _hd_bytes(()->mul!(c, q, b, alpha, beta)) < 4096
        end
        full=bidiagonalize(FaerMatrix(randn(rng, T, m, n)); compact = true, full = true)
        q=full.U
        d=Matrix(q)
        b=randn(rng, T, m, 2)
        c=copy(b)
        lmul!(q, c)
        @test c≈d*b rtol=tol
        mul!(c, q', c)
        @test c≈b rtol=tol
    end
end

@testset "Operator products into destinations" begin
    rng=MersenneTwister(95)
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        tol=sqrt(eps(real(T)))*100
        a=randn(rng, T, 5, 4)
        sq=randn(rng, T, 4, 4)
        ops=(
            FaerLinearOperator(FaerMatrix(a)),
            FaerLinearOperator(sparse(a)),
            FaerLinearOperator(FaerMatrix(a))',
            FaerLinearOperator((y, x)->mul!(y, a, x), (y, x)->mul!(y, a', x), 5, 4; T),
        )
        for op in ops
            d=Matrix(op)
            b=randn(rng, T, size(op, 2), 3)
            c0=randn(rng, T, size(op, 1), 3)
            l=randn(rng, T, 2, size(op, 1))
            l0=randn(rng, T, 2, size(op, 2))
            alpha, beta=T(2), T(-0.5)
            for be in (zero(T), beta), x in (b, faerview(b), FaerMatrix(b))
                for got in _hd_destinations(c->mul!(c, op, x, alpha, be), c0)
                    @test got≈alpha*d*b+be*c0 rtol=tol
                end
            end
            for got in _hd_destinations(c->mul!(c, op, b), c0)
                @test got≈d*b rtol=tol
            end
            for got in _hd_destinations(c->mul!(c, l, op, alpha, beta), l0)
                @test got≈alpha*l*d+beta*l0 rtol=tol
            end
            x=randn(rng, T, size(op, 2))
            y=randn(rng, T, size(op, 1))
            y0=copy(y)
            @test mul!(y, op, x, 2.0, 0.5)===y
            @test y≈2*d*x+y0/2 rtol=tol
            c=fill(T(NaN), size(c0))
            mul!(c, op, b, alpha, 0)
            @test c≈alpha*d*b rtol=tol
            c=copy(c0)
            mul!(c, op, fill(T(NaN), size(b)), 0, beta)
            @test c≈beta*c0 rtol=tol
            c=copy(c0)
            @test _hd_bytes(()->mul!(c, op, b)) < 4096
            @test _hd_bytes(()->mul!(c, op, b, alpha, beta)) < 4096
        end
        # Square operators: in-place and aliased products.
        op=FaerLinearOperator(FaerMatrix(sq))
        b=randn(rng, T, 4, 3)
        c=copy(b)
        @test lmul!(op, c)===c
        @test c≈sq*b rtol=tol
        r=copy(b)
        rv=faerview(view(r, 4:-1:1, :))
        lmul!(op, rv)
        @test Matrix(rv)≈sq*b[4:-1:1, :] rtol=tol
        c=copy(b)
        mul!(c, op, c, 2, 1)
        @test c≈2*sq*b+b rtol=tol
        l=randn(rng, T, 3, 4)
        c=copy(l)
        rmul!(c, op)
        @test c≈l*sq rtol=tol
    end
    # Destination writes do not depend on the problem size.
    rng=MersenneTwister(96)
    small, large=map((16, 128)) do n
        op=FaerLinearOperator(FaerMatrix(randn(rng, n, n)))
        b, c=randn(rng, n, n), randn(rng, n, n)
        (_hd_bytes(()->mul!(c, op, b)), _hd_bytes(()->mul!(c, b, op, 2.0, 3.0)))
    end
    @test large[1]<=small[1]+256 && large[2]<=small[2]+256
    bytes, ro=_hd_readonly([1.0 2; 3 4])
    before=copy(bytes)
    op=FaerLinearOperator(FaerMatrix([1.0 2; 3 4]))
    @test_throws ArgumentError mul!(ro, op, [1.0 0; 0 1])
    @test_throws ArgumentError mul!(ro, op, [1.0 0; 0 1], 2, 3)
    @test bytes==before
end
