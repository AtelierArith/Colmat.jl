using Test, Colmat, LinearAlgebra, SparseArrays
@testset "Generalized EVD scratch and team" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), n in (0, 1, 4), width in (1, 2)
        A=Matrix{T}(I, n, n)
        n>1 && (A[1, 2] = -2; A[2, 1] = 3)
        n>2 && (A[1, 3]=T(0.2))
        T<:Complex && (A .*= T(1+0.2im))
        B=Matrix{T}(I, n, n)
        n>1 && (B[1, 1] = 2; B[1, 2] = T(0.1))
        a, b=FaerMatrix(A), FaerMatrix(B)
        FaerTeam(width) do team
            s=FaerScratch(eigen, a, b; team)
            @test scratch_capacity(s)==scratch_requirements(eigen!, a, b; team)
            F=eigen(a, b; scratch = s, team)
            @test A*F.vectors≈B*F.vectors*Diagonal(F.values) atol=1e-3
            @test all(v->isapprox(norm(v), 1; atol = 1e-4), eachcol(F.vectors))
            @test Matrix(a)==A
            @test Matrix(b)==B
            @test eltype(F.values)==Complex{typeof(real(zero(T)))}
            reserve!(s, eigvals!, a, b; team)
            @test sort(eigvals(a, b; scratch = s, team); by = x->(real(x), imag(x)))≈sort(
                F.values;
                by = x->(real(x), imag(x)),
            ) atol=1e-3
            @test eigen(a, b; team).values≈F.values atol=1e-3
            x, y=FaerMatrix(A), FaerMatrix(B)
            G=eigen!(x, y; scratch = s, team)
            @test A*G.vectors≈B*G.vectors*Diagonal(G.values) atol=1e-3
            @test istriu(Matrix(y))
            @test all(iszero, tril(Matrix(x), -2))
            x, y=FaerMatrix(A), FaerMatrix(B)
            vals=eigvals!(x, y; scratch = s, team)
            @test vals≈F.values atol=1e-3
            @test istriu(Matrix(y))
            close(s)
        end
    end
end
@testset "Generalized EVD larger analytic spectrum" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        n=64
        C=Matrix{T}(SymTridiagonal(fill(2.0, n), fill(-1.0, n-1)))
        X=reshape(T.(1:(n*n)), n, n)/(n*n)
        T<:Complex && (X .+= T(0.1im))
        B=X*X'+Matrix{T}(I, n, n)
        A=B*C
        a, b=FaerMatrix(A), FaerMatrix(B)
        FaerTeam(2) do team
            s=FaerScratch(eigen, a, b; team)
            F=eigen(a, b; scratch = s, team)
            @test real.(F.values)≈[2-2cos(k*pi/(n+1)) for k = 1:n] atol=3e-3
            @test maximum(abs, imag.(F.values))<3e-3
            @test A*F.vectors≈B*F.vectors*Diagonal(F.values) rtol=1e-3 atol=1e-3
            close(s)
        end
    end
end
@testset "Generalized EVD boundary and transactions" begin
    a, b=FaerMatrix([0.0 -1; 1 0]), FaerMatrix([2.0 0; 0 1])
    A, B=Matrix(a), Matrix(b)
    s=FaerScratch(eigen, a, b)
    @test_throws ArgumentError eigen(a, b; scratch = FaerScratch())
    @test_throws ArgumentError eigen(a, b; team = 1)
    @test_throws ErrorException eigen!(a, b; scratch = s, sortby = x->error("sort failed"))
    @test Matrix(a)==A
    @test Matrix(b)==B
    @test_throws ArgumentError eigen!(a, a; scratch = s)
    b[1, 1]=NaN
    @test_throws Colmat.RustCall.RustError eigen!(a, b; scratch = s)
    @test Matrix(a)==A
    close(s)
    a, b=FaerMatrix(Float32[0 -1; 1 0]), FaerMatrix(B)
    s=FaerScratch(eigen, a, b)
    @test eltype(eigen(a, b; scratch = s).values)==ComplexF64
    close(s)
    a.dims=(3, 3)
    @test_throws DimensionMismatch scratch_requirements(eigen, a, b)
    b.dims=(3, 3)
    s=FaerScratch(eigen, a, b)
    @test_throws Colmat.RustCall.RustError eigen(a, b; scratch = s)
    close(s)
    @test isempty(
        Test.detect_ambiguities(
            Colmat,
            Base,
            LinearAlgebra,
            SparseArrays;
            recursive = true,
        ),
    )
end
@testset "Repeated finite generalized eigenvalues" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        a, b=FaerMatrix(Matrix{T}(I, 3, 3)), FaerMatrix(Matrix{T}(I, 3, 3))
        F=eigen(a, b)
        @test all(isfinite, F.vectors)
        @test F.values≈ones(3)
        @test F.vectors'*F.vectors≈I atol=1e-4
    end
end
# Regression guards for tensor4all/faer#217 (fixed by faer#221) and
# tensor4all/faer#219 (fixed by faer#224/#226). Colmat no longer post-processes
# native GEVD output, so these exercise the native driver directly.
# Largest backward-error style residual over homogeneous pairs.
function _gevd_residual(A, B, V, α, β; side = :right)
    nA, nB=norm(A), norm(B)
    return maximum(axes(V, 2); init = zero(real(eltype(A)))) do j
        v=view(V, :, j)
        r=side===:right ? β[j]*(A*v)-α[j]*(B*v) : conj(β[j])*(A'*v)-conj(α[j])*(B'*v)
        denom=(abs(β[j])*nA+abs(α[j])*nB)*norm(v)
        iszero(denom) ? norm(r) : norm(r)/denom
    end
end
_gevd_right_residual(A, B, V, α, β)=_gevd_residual(A, B, V, α, β; side = :right)
_gevd_left_residual(A, B, U, α, β)=_gevd_residual(A, B, U, α, β; side = :left)
@testset "Native identity pencil with repeated eigenvalues (faer#217)" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), n in (2, 3), width in (1, 2)
        A=Matrix{T}(I, n, n)
        a, b=FaerMatrix(A), FaerMatrix(A)
        FaerTeam(width) do team
            F=eigen(a, b; team)
            @test all(isfinite, F.vectors)
            @test F.values≈ones(n)
            @test F.vectors'*F.vectors≈I atol=1e-4
            for side in (:right, :left)
                H=eigen(a, b; side, homogeneous = true, team)
                @test all(isfinite, H.vectors)
                @test H.vectors'*H.vectors≈I atol=1e-4
                @test H.α≈H.β atol=1e-4
            end
            H=eigen(a, b; side = :both, homogeneous = true, team)
            @test all(isfinite, H.left) && all(isfinite, H.right)
            @test H.left'*H.left≈I atol=1e-4
            @test H.right'*H.right≈I atol=1e-4
        end
    end
end
@testset "Native left-only spectrum matches right/both (faer#219)" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        n=64
        C=Matrix{T}(SymTridiagonal(fill(2.0, n), fill(-1.0, n-1)))
        X=reshape(T.(1:(n*n)), n, n)/(n*n)
        T<:Complex && (X .+= T(0.1im))
        B=X*X'+Matrix{T}(I, n, n)
        A=B*C
        a, b=FaerMatrix(A), FaerMatrix(B)
        tol=1000eps(real(T))
        exact=[2-2cos(k*pi/(n+1)) for k = 1:n]
        for width in (1, 2)
            FaerTeam(width) do team
                values=eigvals(a, b; team)
                @test count(isfinite, values)==n
                @test real.(values)≈exact atol=3e-3
                R=eigen(a, b; side = :right, homogeneous = true, team)
                L=eigen(a, b; side = :left, homogeneous = true, team)
                D=eigen(a, b; side = :both, homogeneous = true, team)
                for H in (R, L, D)
                    @test all(isfinite, H.α) && all(isfinite, H.β)
                    @test sort(real.(H.α ./ H.β))≈exact atol=3e-3
                end
                @test all(isfinite, L.vectors)
                @test _gevd_right_residual(A, B, R.vectors, R.α, R.β)<tol
                @test _gevd_left_residual(A, B, L.vectors, L.α, L.β)<tol
                @test _gevd_left_residual(A, B, D.left, D.α, D.β)<tol
                @test _gevd_right_residual(A, B, D.right, D.α, D.β)<tol
                @test Matrix(a)==A
                @test Matrix(b)==B
            end
        end
    end
end
@testset "Defective and singular generalized pencils" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        tol=100eps(real(T))
        cases=(
            # Jordan block with a finite double eigenvalue.
            (T[1 1; 0 1], Matrix{T}(I, 2, 2), 2, 0),
            # Jordan block of size 3.
            (T[2 1 0; 0 2 1; 0 0 2], Matrix{T}(I, 3, 3), 3, 0),
            # Defective infinite eigenvalue (nilpotent B).
            (Matrix{T}(I, 2, 2), T[0 1; 0 0], 0, 2),
            # Defective finite eigenvalue plus a simple infinite one.
            (T[1 1 0; 0 1 0; 0 0 3], Matrix{T}(Diagonal(T[1, 1, 0])), 2, 1),
        )
        for (A, B, nfinite, ninfinite) in cases
            a, b=FaerMatrix(A), FaerMatrix(B)
            FaerTeam(2) do team
                for side in (:right, :left, :both)
                    H=eigen(a, b; side, homogeneous = true, team)
                    @test all(isfinite, H.α) && all(isfinite, H.β)
                    @test count(iszero, H.β)==ninfinite
                    @test count(!iszero, H.β)==nfinite
                    @test all(x->!iszero(x), H.α[iszero.(H.β)])
                    if side===:both
                        @test all(isfinite, H.left) && all(isfinite, H.right)
                        @test _gevd_left_residual(A, B, H.left, H.α, H.β)<tol
                        @test _gevd_right_residual(A, B, H.right, H.α, H.β)<tol
                    else
                        @test all(isfinite, H.vectors)
                        @test all(v->isapprox(norm(v), 1; atol = 1e-4), eachcol(H.vectors))
                        residual=side===:left ? _gevd_left_residual : _gevd_right_residual
                        @test residual(A, B, H.vectors, H.α, H.β)<tol
                    end
                end
            end
        end
        # Singular pencils with a shared null vector give indeterminate 0/0
        # pairs; the native driver still returns finite vectors.
        for (A, B) in (
            (zeros(T, 2, 2), zeros(T, 2, 2)),
            (Matrix{T}(Diagonal(T[1, 2, 0])), Matrix{T}(Diagonal(T[1, 1, 0]))),
        )
            a, b=FaerMatrix(A), FaerMatrix(B)
            for side in (:right, :left)
                H=eigen(a, b; side, homogeneous = true)
                @test all(isfinite, H.vectors)
                @test count(j->iszero(H.α[j])&&iszero(H.β[j]), eachindex(H.α))==
                      size(A, 1)-count(!iszero, diag(B))
            end
        end
    end
end
@testset "Extremely scaled generalized pencils" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        R=real(T)
        big, small=floatmax(R)^R(1//4), floatmin(R)^R(1//4)
        base=T[2 1; 0 3]
        for (sa, sb) in ((big, one(R)), (small, one(R)), (big, small), (small, big))
            A, B=base*sa, Matrix{T}(I, 2, 2)*sb
            a, b=FaerMatrix(A), FaerMatrix(B)
            expected=sort([2, 3]*(sa/sb))
            FaerTeam(2) do team
                values=eigvals(a, b; team)
                @test all(isfinite, values)
                @test sort(real.(values))≈expected rtol=1e-4
                for side in (:right, :left)
                    H=eigen(a, b; side, homogeneous = true, team)
                    @test all(isfinite, H.α) && all(isfinite, H.β)
                    @test all(isfinite, H.vectors)
                    @test sort(real.(H.α ./ H.β))≈expected rtol=1e-4
                    residual=side===:left ? _gevd_left_residual : _gevd_right_residual
                    @test residual(A, B, H.vectors, H.α, H.β)<100eps(R)
                end
            end
        end
    end
end
