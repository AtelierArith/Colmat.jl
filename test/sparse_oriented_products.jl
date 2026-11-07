using Test, LinearAlgebra, Random, SparseArrays, Colmat

const _ops=(identity, transpose, adjoint)
_tol(T)=real(T)===Float32 ? 5e-4 : 1e-11

@testset "Transposed and adjoint native sparse×sparse products" begin
    rng=MersenneTwister(1141)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        afmt in (:csc, :csr),
        bfmt in (:csc, :csr)

        tol=_tol(T)
        a, b=sprand(rng, T, 5, 5, 0.4), sprand(rng, T, 5, 5, 0.4)
        aa, bb=FaerSparseMatrix(a; format = afmt), FaerSparseMatrix(b; format = bfmt)
        for f in _ops, g in _ops
            c=f(aa)*g(bb)
            @test c isa FaerSparseMatrix{T}
            # The result follows the left operand: CSC for A, Aᵀ/Aᴴ of a CSR
            # owner, CSR otherwise.
            left_csc=(afmt===:csc)==(f===identity)
            @test c.format===(left_csc ? :csc : :csr)
            @test SparseMatrixCSC(c)≈f(a)*g(b) rtol=tol atol=tol
        end
        # Rectangular shapes and empty dimensions.
        for (m, k, n) in ((4, 3, 2), (0, 3, 2), (4, 0, 2), (4, 3, 0))
            x, y=sprand(rng, T, k, m, 0.5), sprand(rng, T, k, n, 0.5)
            xx, yy=FaerSparseMatrix(x; format = afmt), FaerSparseMatrix(y; format = bfmt)
            @test SparseMatrixCSC(xx'*yy)≈x'*y rtol=tol atol=tol
            @test SparseMatrixCSC(transpose(xx)*yy)≈transpose(x)*y rtol=tol atol=tol
            @test SparseMatrixCSC(yy'*xx)≈y'*x rtol=tol atol=tol
        end
        @test_throws DimensionMismatch FaerSparseMatrix(sprand(rng, T, 3, 4, 0.5))'*bb
    end
    # Mixed scalar types promote first.
    a, b=sprand(rng, Float32, 4, 4, 0.5), sprand(rng, ComplexF64, 4, 4, 0.5)
    c=FaerSparseMatrix(a)'*FaerSparseMatrix(b; format = :csr)
    @test c isa FaerSparseMatrix{ComplexF64}
    @test SparseMatrixCSC(c)≈a'*b
end

@testset "Native sparse×dense products under transpose and adjoint" begin
    rng=MersenneTwister(1142)
    for T in (Float32, Float64, ComplexF32, ComplexF64), fmt in (:csc, :csr)
        tol=_tol(T)
        s=sprand(rng, T, 6, 4, 0.5)
        ss=FaerSparseMatrix(s; format = fmt)
        for f in _ops
            op=f(s)
            x=randn(rng, T, size(op, 2), 3)
            y=randn(rng, T, 3, size(op, 1))
            @test Matrix(f(ss)*FaerMatrix(x))≈op*x rtol=tol atol=tol
            @test Matrix(FaerMatrix(y)*f(ss))≈y*op rtol=tol atol=tol
            for (α, β) in ((true, false), (T(2), T(0)), (T(2), T(3)), (T(0), T(1)))
                c0=randn(rng, T, size(op, 1), 3)
                c=FaerMatrix(c0)
                r=mul!(c, f(ss), FaerMatrix(x), α, β)
                @test r===c
                @test Matrix(c)≈α*(op*x)+β*c0 rtol=tol atol=tol
                d0=randn(rng, T, 3, size(op, 2))
                d=FaerMatrix(d0)
                mul!(d, FaerMatrix(y), f(ss), α, β)
                @test Matrix(d)≈α*(y*op)+β*d0 rtol=tol atol=tol
            end
            # Dense views, transposed dense operands and three-argument mul!.
            xx=FaerMatrix(randn(rng, T, size(op, 2)+2, 3))
            v=view(xx, 2:(size(op, 2)+1), :)
            c=FaerMatrix{T}(undef, size(op, 1), 3)
            mul!(c, f(ss), v)
            @test Matrix(c)≈op*Matrix(v) rtol=tol atol=tol
            z=FaerMatrix(randn(rng, T, 3, size(op, 2)))
            mul!(c, f(ss), z')
            @test Matrix(c)≈op*Matrix(z)' rtol=tol atol=tol
        end
        @test_throws DimensionMismatch mul!(
            FaerMatrix{T}(undef, 6, 2),
            ss,
            FaerMatrix{T}(undef, 3, 2),
        )
        # Vectors (same and promoted scalar types) and borrowed Julia arrays.
        for f in (transpose, adjoint)
            op=f(s)
            x=randn(rng, T, 6)
            @test Vector(f(ss)*FaerVector(x))≈op*x rtol=tol atol=tol
            @test f(ss)*x≈op*x rtol=tol atol=tol
            x64=randn(rng, ComplexF64, 6)
            @test Vector(f(ss)*FaerVector(x64))≈op*x64 rtol=tol atol=tol
            y=randn(rng, T, 6, 2)
            @test Matrix(f(ss)*faerview(y))≈op*y rtol=tol atol=tol
            @test Matrix(f(ss)*y)≈op*y rtol=tol atol=tol
            c=FaerMatrix{T}(undef, 4, 2)
            mul!(c, f(ss), faerview(y), T(2), T(0))
            @test Matrix(c)≈2op*y rtol=tol atol=tol
            z=randn(rng, T, 2, 4)
            d=FaerMatrix{T}(undef, 2, 6)
            mul!(d, faerview(z), f(ss), T(1), T(0))
            @test Matrix(d)≈z*op rtol=tol atol=tol
            @test Matrix(faerview(z)*f(ss))≈z*op rtol=tol atol=tol
        end
    end
    # A dense operand viewing the destination is staged before writing.
    s=sprand(rng, 4, 4, 0.6)
    ss=FaerSparseMatrix(s; format = :csr)
    c0=randn(rng, 4, 4)
    c=FaerMatrix(c0)
    mul!(c, ss', view(c, :, :), 2.0, 1.0)
    @test Matrix(c)≈2s'*c0+c0
    # Mixed scalar types promote and assign once.
    c=FaerMatrix{ComplexF64}(undef, 4, 2)
    x=randn(rng, Float32, 4, 2)
    mul!(c, FaerSparseMatrix(s), FaerMatrix(x))
    @test Matrix(c)≈s*x rtol=1e-6
end

@testset "In-place sparse×dense mul! allocations" begin
    rng=MersenneTwister(1143)
    bytes(n, fmt, f)=begin
        s=FaerSparseMatrix(sprand(rng, n, n, 0.02); format = fmt)
        x=FaerMatrix(randn(rng, n, 8))
        c=FaerMatrix(randn(rng, n, 8))
        run()=mul!(c, f(s), x, 2.0, 1.0)
        run();
        run()
        @allocated run()
    end
    for fmt in (:csc, :csr), f in _ops
        small, large=bytes(100, fmt, f), bytes(1000, fmt, f)
        @test large<4096
        @test large-small<512
    end
end
