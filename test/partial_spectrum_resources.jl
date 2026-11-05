using Colmat, LinearAlgebra, SparseArrays, Test
@testset "Partial-spectrum kernels, fallback and residual resources" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), width in (0, 1, 2)
        team=width==0 ? nothing : FaerTeam(width)
        tol=typeof(real(zero(T)))===Float32 ? 2e-4 : 1e-8
        for fun in (partial_eigen, partial_self_adjoint_eigen, partial_svd),
            storage in (:dense, :csc, :csr, :julia),
            (m, n, force) in (
                (0, 0, false),
                (1, 1, false),
                (5, 5, false),
                (96, 96, false),
                (96, 96, true),
                (104, 96, false),
                (96, 104, false),
                (5, 3, false),
                (3, 5, false),
                (0, 3, false),
                (3, 0, false),
            )

            fun!==partial_svd && m!=n && continue
            A=zeros(T, m, n)
            for i = 1:min(m, n)
                A[i, i]=T(i+1)
                if fun===partial_eigen && T<:Complex
                    A[i, i]+=T(0.25im)
                end
            end
            calls=Ref(0)
            forward=(y, x)->begin
                calls[]+=1
                mul!(y, A, x)
            end
            backward=(y, x)->begin
                calls[]+=1
                mul!(y, A', x)
            end
            op=if storage===:julia
                FaerLinearOperator(forward, backward, m, n; T)
            else
                src=storage===:dense ? FaerMatrix(A) :
                    FaerSparseMatrix(sparse(A); format = storage)
                result=FaerLinearOperator(src)
                close(src)
                result
            end
            k=min(2, m, n)
            max_dim=force ? 96 : 0
            req=scratch_requirements(fun, op, k; team, max_dim, max_restarts = 20, tol)
            @test calls[]==0
            work=FaerScratch(fun, op, k; team, max_dim, max_restarts = 20)
            cap=scratch_capacity(work)
            result=fun(op, k; scratch = work, team, max_dim, max_restarts = 20, tol)
            @test result.nconverged==k
            @test scratch_capacity(work)==cap
            expected=k==0 ? :partial :
                     fun!==partial_eigen && (min(m, n)<=64 || force) ? :dense_fallback :
                     :partial
            @test result.algorithm===expected
            if fun===partial_svd
                gold=svdvals(A)[1:k]
                @test result.S≈gold atol=100tol rtol=100tol
                for j = 1:k
                    u=result.U[:, j]
                    v=result.Vt'[:, j]
                    s=result.S[j]
                    @test result.residuals[j]≈max(norm(A*v-s*u), norm(A'*u-s*v)) atol=100tol rtol=100tol
                end
            else
                gold=sort(diag(A); by = abs, rev = true)[1:k]
                @test result.values≈gold atol=100tol rtol=100tol
                for j = 1:k
                    @test result.residuals[j]≈norm(
                        A*result.vectors[:, j]-result.values[j]*result.vectors[:, j],
                    ) atol=100tol rtol=100tol
                end
            end
            @test maximum(result.residuals; init = 0.0)<100tol*max(m, n, 1)
            if req.bytes>0 && storage===:julia
                short=FaerScratch(req.bytes-1; alignment = req.alignment)
                before=calls[]
                @test_throws ArgumentError fun(
                    op,
                    k;
                    scratch = short,
                    team,
                    max_dim,
                    max_restarts = 20,
                    tol,
                )
                @test calls[]==before
                @test reserve!(short, fun, op, k; team, max_dim, max_restarts = 20)===short
                close(short)
            end
            close(work)
            close(op)
        end
        isnothing(team)||close(team)
    end
end
@testset "Partial-spectrum owner validation and idle resources" begin
    for fun in (partial_eigen, partial_self_adjoint_eigen, partial_svd)
        op=FaerLinearOperator(Diagonal(collect(1.0:96)))
        team=FaerTeam(2)
        work=FaerScratch(fun, op, 2; team)
        @test fun(op, 2; scratch = work, team, max_restarts = 0).nconverged==0
        @test scratch_requirements(fun, op, 2; team, max_restarts = 0).bytes==0
        @test scratch_requirements(fun, op, 0; team).bytes==0
        @test_throws ArgumentError fun(op, 2; scratch = :invalid)
        @test_throws ArgumentError fun(op, 2; team = :invalid)
        @test_throws ArgumentError scratch_requirements(
            fun,
            op,
            2;
            min_dim = 90,
            max_dim = 70,
        )
        @test_throws DimensionMismatch scratch_requirements(fun, op, 2; v0 = ones(3))
        dims=op.dims
        op.dims=(95, 95)
        @test_throws Colmat.RustCall.RustError scratch_requirements(fun, op, 2; team)
        @test_throws Colmat.RustCall.RustError fun(op, 2; team, scratch = work)
        op.dims=(typemax(Int), typemax(Int))
        @test_throws Colmat.RustCall.RustError scratch_requirements(fun, op, 1)
        @test_throws Colmat.RustCall.RustError fun(op, 1)
        op.dims=dims
        fake=FaerLinearOperator{Float32,typeof(op.handle)}(
            op.handle,
            op.dims,
            op.storage,
            op.transposed,
            op.conjugated,
            op.callbacks,
        )
        @test_throws Colmat.RustCall.RustError scratch_requirements(fun, fake, 2)
        fake.handle=nothing
        if fun!==partial_svd
            rect=FaerLinearOperator(ones(3, 4))
            @test_throws DimensionMismatch fun(rect, 1)
            @test_throws DimensionMismatch scratch_requirements(fun, rect, 1)
            close(rect)
        end
        start=FaerVector(ones(96))
        start.len=95
        @test_throws Colmat.RustCall.RustError fun(op, 2; v0 = start, team, scratch = work)
        start.len=96
        close(start)
        close(team)
        @test_throws ArgumentError fun(op, 2; scratch = work, team)
        close(work)
        @test_throws ArgumentError fun(op, 2; scratch = work)
        close(op)
        @test_throws ArgumentError scratch_requirements(fun, op, 2)
    end
end
@testset "Partial spectra non-diagonal blocked fallback and composed resources" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), n in (5, 129), width in (0, 2)
        R=typeof(real(zero(T)))
        tol=R===Float32 ? 2e-4 : 1e-8
        A=T[i==j ? 5+i/n : 0.01sin(i+2j) for i = 1:n, j = 1:n]
        T<:Complex && (A .+= T(0.01im) .* T[sin(i-j) for i = 1:n, j = 1:n])
        A=A+A'
        left=FaerLinearOperator(A)
        right=identity_preconditioner(n; T)
        op=adjoint(left*right)
        close(left)
        close(right)
        team=width==0 ? nothing : FaerTeam(width)
        for fun in (partial_eigen, partial_self_adjoint_eigen, partial_svd)
            work=FaerScratch(fun, op, 2; team, max_dim = n, max_restarts = 30)
            result=fun(op, 2; scratch = work, team, max_dim = n, max_restarts = 30, tol)
            @test result.nconverged==2
            if fun===partial_svd
                @test result.S≈svdvals(A)[1:2] atol=100tol rtol=100tol
            else
                gold=sort(eigvals(Hermitian(A)); by = abs, rev = true)[1:2]
                @test result.values≈gold atol=100tol rtol=100tol
            end
            @test maximum(result.residuals)<100tol*n
            close(work)
        end
        close(op)
        isnothing(team)||close(team)
    end
end
