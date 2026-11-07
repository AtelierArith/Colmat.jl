using Test, LinearAlgebra, SparseArrays, Colmat

# Construction, conversion, copies and scalar arithmetic store Inf/NaN like
# Julia arrays (#174); finiteness is checked only by the numeric kernels.
_nonfinite_data(::Type{T}) where {T<:Real} = T[1 Inf; NaN -0.0; -Inf 2]
_nonfinite_data(::Type{T}) where {T<:Complex} =
    T[complex(1, -Inf) Inf; complex(NaN, 1) complex(-0.0, -0.0); -Inf complex(2, NaN)]
_same(a, b) = size(a)==size(b) && all(map(isequal, collect(a), collect(b)))
# Mixed complex products may differ from Julia only in the sign of zero parts.
_part(x, y) = isequal(x, y) || x==y
_samezero(a, b) =
    size(a)==size(b) &&
    all(map((x, y)->_part(real(x), real(y))&&_part(imag(x), imag(y)), collect(a), collect(b)))
const _NativeError=Union{ArgumentError,Colmat.RustCall.RustError}

@testset "Non-finite values in owned storage" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        data=_nonfinite_data(T)
        A=FaerMatrix(data)
        @test A isa FaerMatrix{T}
        @test _same(Matrix(A), data)
        @test signbit(real(A[2, 2]))
        @test _same(Matrix(FaerMatrix(view(data, 3:-1:1, :))), data[3:-1:1, :])
        @test _same(Matrix(FaerMatrix(transpose(data))), transpose(data))
        @test _same(Matrix(copy(A)), data)
        v=vec(data)
        @test _same(Vector(FaerVector(v)), v)
        @test _same(Matrix(FaerRow(v)), reshape(v, 1, :))
        @test _same(diag(Matrix(FaerDiagonal(v))), v)
        @test signbit(real(FaerVector(v)[5]))
        S=FaerSparseMatrix(sparse(data))
        @test _same(Matrix(S), Matrix(sparse(data)))

        # Conversion keeps non-finite values, including through widening.
        for U in (Float32, Float64, ComplexF32, ComplexF64)
            T<:Complex && U<:Real && continue
            B=FaerMatrix{U}(A)
            @test B isa FaerMatrix{U}
            @test _same(Matrix(B), U.(data))
            @test _same(Vector(FaerVector{U}(FaerVector(v))), U.(v))
            # copyto! across scalar types and from views and vectors.
            dest=FaerMatrix{U}(undef, size(data)...)
            @test copyto!(dest, A)===dest
            @test _same(Matrix(dest), U.(data))
            copyto!(dest, view(A, 3:-1:1, :))
            @test _same(Matrix(dest), U.(data[3:-1:1, :]))
            copyto!(dest, data)
            @test _same(Matrix(dest), U.(data))
            w=FaerVector{U}(undef, length(v))
            copyto!(w, FaerVector(v))
            @test _same(Vector(w), U.(v))
            copyto!(w, view(A, :))
            @test _same(Vector(w), U.(v))
        end

        # Scalar products and quotients, including ones that widen. Real
        # factors scale complex entries componentwise, as in Julia.
        for α in (2, 2.0f0, 2.0, 0.5, -0.0)
            @test _same(Matrix(α*A), α*data)
            @test _same(Matrix(A*α), data*α)
            @test _same(Matrix(A/α), data/α)
            @test _same(Vector(α*FaerVector(v)), α*v)
            @test _same(Matrix(FaerRow(v)/α), reshape(v, 1, :)/α)
        end
        for α in (2.0f0+0.5f0im, 2.0+0.5im)
            @test _samezero(Matrix(α*A), α*data)
            @test _samezero(Vector(α*FaerVector(v)), α*v)
        end
        @test _samezero(Matrix(A/(2.0+0.0im)), data/(2.0+0.0im))
        @test _same(Matrix(A/0), data/0)
        @test _same(Matrix(-A), -data)
        @test _same(Vector(-FaerVector(v)), -v)
        @test _same(Matrix(A+A), data+data)
        @test _same(Matrix(A-A), data-data)
    end
end

@testset "Kernels on non-finite input stay checked" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), bad in (NaN, Inf, -Inf)
        J=T[4 1; 1 bad]
        A=FaerMatrix(J)
        I2=FaerMatrix(Matrix{T}(I, 2, 2))
        @test _same(Matrix(A), J)
        for f in (
            lu,
            x->lu(x, RowMaximum()),
            x->lu!(copy(x)),
            qr,
            x->qr(x, ColumnNorm()),
            x->cholesky(Hermitian(x)),
            x->cholesky!(Hermitian(copy(x))),
            x->ldlt(Hermitian(x)),
            x->bunchkaufman(Hermitian(x)),
            svd,
            svdvals,
            eigen,
            eigvals,
            x->eigen(Hermitian(x)),
            x->eigvals(Hermitian(x)),
            x->eigen(x, I2),
            logabsdet,
            inv,
            pinv,
            x->x\I2,
            x->x\T[1, 2],
            x->I2\x,
            schur,
            hessenberg,
            rank,
            opnorm,
            cond,
            factorize,
            x->lu(FaerSparseMatrix(sparse(J))),
            x->qr(FaerSparseMatrix(sparse(J))),
            x->cholesky(Hermitian(FaerSparseMatrix(sparse(J)))),
        )
            @test_throws _NativeError f(A)
        end
        # Elementwise kernels propagate non-finite values like Julia.
        # Complex matmul may pick NaN over Inf for 0*Inf terms (as BLAS may).
        @test isequal(isfinite.(Matrix(A*I2)), isfinite.(J*Matrix(I2)))
        T<:Real && @test _same(Matrix(A*I2), J*Matrix(I2))
        @test isequal(tr(A), tr(J))
        @test isnan(det(A))==isnan(det(J))
        # Factorizations of finite matrices still reject non-finite right-hand sides.
        F=lu(I2)
        @test_throws _NativeError F\FaerMatrix(J)
        @test_throws _NativeError F\J
    end
end
