using Colmat, LinearAlgebra, Test, Random, Libdl

@testset "Native LAPACK decomposition drivers" begin
    Random.seed!(392)
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        @testset "$T" begin
            for (m, n) in ((6, 6), (8, 5), (5, 8))
                A = randn(T, m, n)
                F = lu(A)
                @test F.L * F.U ≈ A[F.p, :]
                S = svd(A)
                @test S.U * Diagonal(S.S) * S.Vt ≈ A
                S = svd(A; full=true)
                r = min(m, n)
                @test S.U[:, 1:r] * Diagonal(S.S) * S.Vt[1:r, :] ≈ A
                S = svd(A; alg=LinearAlgebra.QRIteration())
                @test S.U * Diagonal(S.S) * S.Vt ≈ A
            end
            A = randn(T, 6, 6) + 8I
            B = randn(T, 6, 2)
            F = lu(A)
            for t in ('N', 'T', 'C')
                X = copy(B)
                LAPACK.getrs!(t, F.factors, F.ipiv, X)
                M = if t == 'N'
                    A
                elseif t == 'T'
                    transpose(A)
                else
                    A'
                end
                @test M * X ≈ B
            end
            @test inv(F) * A ≈ Matrix{T}(I, 6, 6)
            for permute in (false, true), scale in (false, true)
                E = eigen(A; permute, scale)
                @test A * E.vectors ≈ E.vectors * Diagonal(E.values)
            end
            R = T[0 -1; 1 0]
            E = eigen(R)
            @test R * E.vectors ≈ E.vectors * Diagonal(E.values)
            for u in (:U, :L)
                H = Hermitian(A + A', u)
                E = eigen(H)
                @test H * E.vectors ≈ E.vectors * Diagonal(E.values)
                for alg in (LinearAlgebra.QRIteration(), LinearAlgebra.DivideAndConquer())
                    G = eigen(H; alg)
                    @test H * G.vectors ≈ G.vectors * Diagonal(G.values)
                end
                ev = eigen(H, 2:4)
                @test H * ev.vectors ≈ ev.vectors * Diagonal(ev.values)
                @test ev.values ≈ E.values[2:4]
                mid = (E.values[2] + E.values[3]) / 2
                @test eigvals(H, mid, E.values[end] + 1) ≈ E.values[3:end]
                C = cholesky(H)
                @test C.L * C.U ≈ H
                @test H * (C \ B) ≈ B
                @test H * inv(C) ≈ Matrix{T}(I, 6, 6)
            end
            singular = T[1 2; 2 4]
            F = lu(singular; check=false)
            @test !issuccess(F)
            @test_throws SingularException lu(singular)
            @test_throws PosDefException cholesky(Hermitian(T[1 2; 2 1]))
            for (m, n) in ((0, 0), (0, 3), (3, 0))
                S = svd(zeros(T, m, n); full=true)
                @test size(S.U) == (m, m)
                @test size(S.Vt) == (n, n)
            end
            @test isempty(eigen(zeros(T, 0, 0)).values)
            for interface in (:lp64, :ilp64),
                suffix in (
                    "getrf_",
                    "getrs_",
                    "getri_",
                    "gesdd_",
                    "gesvd_",
                    "geev_",
                    "geevx_",
                    "potrf_",
                    "potrs_",
                    "potri_",
                    T <: Real ? "syevr_" : "heevr_",
                )

                prefix = if T == Float32
                    "s"
                elseif T == Float64
                    "d"
                elseif T == ComplexF32
                    "c"
                else
                    "z"
                end
                @test realpath(
                    BLAS.lbt_find_backing_library(prefix * suffix, interface).libname
                ) == realpath(Colmat.backend_library())
            end
        end
    end
end

@testset "General eigenvectors, balance and expert fallback" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        A = T[2 3 0 0; -1 4 0 0; 0 0 5 7; 0 0 -2 1]
        for job in ('N', 'P', 'S', 'B'), sense in ('N', 'B')
            result = LAPACK.geevx!(job, 'V', 'V', sense, copy(A))
            if T <: Real
                _, wr, wi, L, R = result[1:5]
                values = complex.(wr, wi)
                function unpack(X, wi)
                    Z = complex.(X)
                    j = 1
                    while j <= length(wi)
                        if wi[j] > 0
                            Z[:, j] = X[:, j] + im * X[:, j + 1]
                            Z[:, j + 1] = conj.(Z[:, j])
                            j += 2
                        else
                            j += 1
                        end
                    end
                    Z
                end
                L = unpack(L, wi)
                R = unpack(R, wi)
            else
                _, values, L, R = result[1:4]
            end
            @test A * R ≈ R * Diagonal(values)
            @test A' * L ≈ L * Diagonal(conj.(values))
        end
        @test LAPACK.geev!('V', 'V', copy(A)) isa Tuple
    end
end

for I in (Int32, Int64), T in (Float32, Float64, ComplexF32, ComplexF64)
    @eval function direct_lu_abi(f, ::Type{$I}, A::Matrix{$T}, piv::Vector{$I})
        m, n = size(A)
        info = Ref{$I}(0)
        ccall(
            f,
            Cvoid,
            (Ref{$I}, Ref{$I}, Ptr{$T}, Ref{$I}, Ptr{$I}, Ref{$I}),
            m,
            n,
            A,
            max(1, m),
            piv,
            info,
        )
        info[]
    end
end
@testset "Native LU in both integer ABIs" begin
    h = Libdl.dlopen(Colmat.backend_library())
    for I in (Int32, Int64), T in (Float32, Float64, ComplexF32, ComplexF64)
        prefix = if T == Float32
            "s"
        elseif T == Float64
            "d"
        elseif T == ComplexF32
            "c"
        else
            "z"
        end
        f = Libdl.dlsym(h, prefix * "getrf" * (I == Int32 ? "_" : "_64_"))
        A = T[0 2 1; 3 1 2; 1 5 4]
        LU = copy(A)
        piv = Vector{I}(undef, 3)
        @test direct_lu_abi(f, I, LU, piv) == 0
        perm = collect(1:3)
        for (i, p) in enumerate(piv)
            perm[i], perm[p] = perm[p], perm[i]
        end
        @test Matrix(UnitLowerTriangular(LU)) * Matrix(UpperTriangular(LU)) ≈ A[perm, :]
    end
end

@testset "GEEVX balance metadata interoperates with existing GEBAK" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        core = T[1 64 0 0 0; 1 / 64 2 0 0 0; 0 0 3 0 0; 0 0 0 4 32; 0 0 0 -1 / 32 5]
        if T <: Complex
            core[1, 2] *= T(1 + im)
        end
        perm = [3, 5, 2, 1, 4]
        A = core[perm, perm]
        for job in ('N', 'P', 'S', 'B')
            data = LAPACK.geevx!(job, 'V', 'V', 'N', copy(A))
            offset = T <: Real ? 1 : 0
            ilo, ihi, scale, abnrm = data[(5 + offset):(8 + offset)]
            transform = Matrix{T}(I, 5, 5)
            LAPACK.gebak!(job, 'R', ilo, ihi, scale, transform)
            balanced = transform \ (A * transform)
            @test abnrm ≈ maximum(sum(abs(balanced[i, j]) for i in 1:5) for j in 1:5)
        end
    end
end
