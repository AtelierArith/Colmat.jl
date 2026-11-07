using Colmat, Test, LinearAlgebra, SparseArrays

const PARTIAL_STAGING_TYPES=(Float32, Float64, ComplexF32, ComplexF64)
_partial_tol(::Type{T}) where {T} = real(T)===Float32 ? 2e-4 : 1e-8
# Bytes of the arrays a partial spectrum returns (vectors, values, residuals).
function _partial_result_bytes(F)
    arrays=hasproperty(F, :U) ? (F.U, F.S, F.Vt, F.residuals) :
           (F.vectors, F.values, F.residuals)
    return sum(a->sizeof(a), arrays)
end
# Real nonsymmetric matrix with complex-conjugate eigenvalue pairs.
function _partial_rotation_blocks(::Type{T}, n) where {T}
    A=zeros(T, n, n)
    for j = 1:2:(n-1)
        r=T(j)
        A[j, j]=A[j+1, j+1]=r
        A[j, j+1]=r/2
        A[j+1, j]=-r/2
    end
    isodd(n) && (A[n, n]=T(n))
    return A
end

@testset "Partial spectra are returned in faer's magnitude order" begin
    for T in PARTIAL_STAGING_TYPES, n in (12, 160)
        R=real(T)
        tol=_partial_tol(T)
        H=T.(Matrix(SymTridiagonal(collect(R, 1:n), fill(R(1//4), n-1))))
        B=hcat(Matrix(Diagonal(T.(1:n))), zeros(T, n, 3))
        if T<:Complex
            # Nonreal Hermitian and rectangular data: complex singular vectors
            # exercise the in-place conjugation into Vt.
            for i = 1:(n-1)
                H[i, i+1]=T(1//4)+T(1//5)*im
                H[i+1, i]=conj(H[i, i+1])
                B[i, i+1]=T(1//3)*im
            end
            B[n, n+1]=T(1//2)-T(1//3)*im
        end
        G=_partial_rotation_blocks(T, n)
        for (f, A) in (
            (partial_self_adjoint_eigen, H),
            (partial_eigen, H),
            (partial_eigen, G),
            (partial_svd, B),
        )
            op=FaerLinearOperator(A)
            F=f(op, 5; tol, max_restarts = 50)
            @test F.algorithm===(n<=64 && f!==partial_eigen ? :dense_fallback : :partial)
            @test F.nconverged==5
            magnitudes=f===partial_svd ? F.S : abs.(F.values)
            @test issorted(magnitudes; rev = true)
            if f===partial_svd
                @test F.Vt isa Matrix{T}
                @test size(F.Vt)==(5, size(A, 2))
                @test norm(A*F.Vt'-F.U*Diagonal(F.S))<100tol*n
                @test norm(A'*F.U-F.Vt'*Diagonal(F.S))<100tol*n
                for j = 1:5
                    v=conj(F.Vt[j, :])
                    u=F.U[:, j]
                    @test F.residuals[j]≈max(norm(A*v-F.S[j]*u), norm(A'*u-F.S[j]*v)) atol=10tol*n
                    @test norm(A*v-F.S[j]*u)<100tol*n
                end
                @test F.Vt*F.Vt'≈I atol=100tol
            else
                @test F.vectors isa Matrix
                f===partial_self_adjoint_eigen && @test F.values isa Vector{R}
                for j = 1:5
                    v=F.vectors[:, j]
                    @test F.residuals[j]≈norm(A*v-F.values[j]*v) atol=10tol*n
                end
                @test maximum(F.residuals)<100tol*n
            end
            close(op)
        end
    end
end
@testset "Partially converged partial spectra truncate in place" begin
    for T in PARTIAL_STAGING_TYPES
        n=200
        R=real(T)
        A=T.(Matrix(Diagonal(collect(R, 1:n))))
        for i = 1:(n-1)
            A[i, i+1]=T<:Complex ? T(1//10)*im : T(1//10)
            A[i+1, i]=conj(A[i, i+1])
        end
        op=FaerLinearOperator(A)
        for f in (partial_eigen, partial_self_adjoint_eigen, partial_svd)
            F=f(op, 30; tol = _partial_tol(T), max_restarts = 3)
            c=F.nconverged
            @test 0<c<30
            @test length(F.residuals)==c
            if f===partial_svd
                @test size(F.U)==(n, c)
                @test size(F.Vt)==(c, n)
                @test length(F.S)==c
                for j = 1:c
                    v=conj(F.Vt[j, :])
                    @test norm(A*v-F.S[j]*F.U[:, j])≤F.residuals[j]*(1+1e-3)+10eps(R)*n
                    @test norm(A'*F.U[:, j]-F.S[j]*v)≤F.residuals[j]*(1+1e-3)+10eps(R)*n
                end
                @test F.residuals[1]<100_partial_tol(T)*n
            else
                @test size(F.vectors)==(n, c)
                @test length(F.values)==c
            end
        end
        close(op)
    end
end
@testset "Partial-spectrum start vectors are read in place" begin
    for T in PARTIAL_STAGING_TYPES
        n=96
        R=real(T)
        A=T.(Matrix(Diagonal(collect(R, 1:n))))
        op=FaerLinearOperator(A)
        v=T.(collect(range(1, 2; length = n)))
        owner=FaerMatrix(hcat(reverse(v), v))
        for f in (partial_eigen, partial_self_adjoint_eigen, partial_svd)
            reference=f(op, 2; v0 = v, tol = _partial_tol(T), max_restarts = 20)
            for start in (
                FaerVector(v),
                faerview(v),
                view(owner, :, 2),
                3v,
                view(hcat(v, v), :, 1),
                Float64.(real.(v)) .+ (T<:Complex ? 0im : 0),
            )
                F=f(op, 2; v0 = start, tol = _partial_tol(T), max_restarts = 20)
                @test F.nconverged==reference.nconverged
                if f===partial_svd
                    @test F.S≈reference.S
                else
                    @test F.values≈reference.values
                end
            end
            @test_throws ArgumentError f(op, 1; v0 = fill(T(NaN), n))
            @test_throws ArgumentError f(op, 1; v0 = FaerVector(fill(T(Inf), n)))
            @test_throws ArgumentError f(op, 1; v0 = FaerVector(zeros(T, n)))
            @test_throws DimensionMismatch f(op, 1; v0 = FaerVector(ones(T, n-1)))
        end
        close(op)
    end
end
@testset "Partial spectra stage only their returned Julia arrays" begin
    for T in PARTIAL_STAGING_TYPES, n in (400, 1600)
        R=real(T)
        H=T.(Matrix(SymTridiagonal(collect(R, 1:n), fill(R(1//4), n-1))))
        B=H[:, 1:(n÷2)]
        native=FaerLinearOperator(FaerMatrix(H))
        rect=FaerLinearOperator(FaerMatrix(B))
        callback=FaerLinearOperator(
            (Y, X)->mul!(Y, H, X),
            (Y, X)->mul!(Y, H, X),
            n,
            n;
            T,
        )
        start=FaerVector(T.(ones(n)))
        half=FaerVector(T.(ones(n÷2)))
        tol=_partial_tol(T)
        for (f, op) in (
            (partial_eigen, native),
            (partial_self_adjoint_eigen, native),
            (partial_svd, rect),
            (partial_self_adjoint_eigen, callback),
            (partial_eigen, callback),
        )
            for v0 in (nothing, size(op, 2)==n ? start : half)
                call=()->f(op, 6; tol, v0)
                F=call()
                @test F.nconverged==6
                bytes=_partial_result_bytes(F)
                # Beyond the returned arrays only Colmat's fixed FFI marshalling
                # (and per-application callback trampolines) allocate. Inside
                # the parallel full suite a worker that already ran other files
                # measured up to 23 KB more than a fresh process.
                slack=op===callback ? 160_000 : 64_000
                @test @allocated(call())<bytes+slack
            end
        end
        close(native)
        close(rect)
        close(start)
        close(half)
    end
end
