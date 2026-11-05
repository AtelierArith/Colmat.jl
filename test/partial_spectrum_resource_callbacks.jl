using Colmat, LinearAlgebra, Test
@testset "Partial spectra yield, exceptions and shared resources" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        fun in (partial_eigen, partial_self_adjoint_eigen, partial_svd),
        n in (5, 96)

        A=Matrix(Diagonal(T.(1:n)))
        fail=Ref(false)
        failnothing=Ref(false)
        ids=Int[]
        buffers=Any[]
        forward=(y, x)->begin
            @test current_task().sticky
            push!(ids, Threads.threadid())
            push!(buffers, (y, x))
            yield()
            fail[]&&error("partial callback test")
            failnothing[]&&throw(nothing)
            mul!(y, A, x)
        end
        backward=(y, x)->begin
            @test current_task().sticky
            push!(ids, Threads.threadid())
            yield()
            fail[]&&error("partial callback test")
            failnothing[]&&throw(nothing)
            mul!(y, A', x)
        end
        op=FaerLinearOperator(forward, backward, n, n; T)
        team=FaerTeam(2)
        work=FaerScratch(fun, op, 2; team, max_restarts = 20)
        @test isempty(ids)
        tol=T in (Float32, ComplexF32) ? 2e-4 : 1e-8
        task=current_task()
        sticky=task.sticky
        task.sticky=false
        try
            result=fun(op, 2; scratch = work, team, tol, max_restarts = 20)
            @test result.nconverged==2
            @test !task.sticky
            @test all(==(Threads.threadid()), ids)
            before=deepcopy(buffers[1])
            @test all(isfinite, buffers[1][1])
            fail[]=true
            @test_throws ErrorException fun(
                op,
                2;
                scratch = work,
                team,
                tol,
                max_restarts = 20,
            )
            @test !task.sticky
            fail[]=false
            failnothing[]=true
            @test_throws Nothing fun(op, 2; scratch = work, team, tol, max_restarts = 20)
            @test !task.sticky
            failnothing[]=false
            @test fun(op, 2; scratch = work, team, tol, max_restarts = 20).nconverged==2
            @test buffers[1]==before
            short=FaerScratch(0; alignment = 1)
            count=length(ids)
            @test_throws ArgumentError fun(
                op,
                2;
                scratch = short,
                team,
                tol,
                max_restarts = 20,
            )
            @test length(ids)==count
            close(short)
        finally
            task.sticky=sticky
        end
        if T===Float64 && n==5
            jobs=[@async begin
                yield()
                fun(op, 2; scratch = work, team, tol, max_restarts = 20)
            end for _ = 1:3]
            @test all(r->r.nconverged==2, fetch.(jobs))
        end
        close(work)
        close(team)
        close(op)
    end
end
@testset "Saved inverse and composed partial actions" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        A=Matrix(Diagonal(T.(1:5)))
        source=FaerMatrix(A)
        F=cholesky(Hermitian(source))
        inverse=FaerLinearOperator(F)
        id=identity_preconditioner(5; T)
        composed=adjoint(inverse*id)
        close(source)
        close(F)
        close(inverse)
        close(id)
        team=FaerTeam(2)
        for fun in (partial_eigen, partial_self_adjoint_eigen, partial_svd)
            work=FaerScratch(fun, composed, 2; team)
            result=fun(composed, 2; scratch = work, team)
            @test result.nconverged==2
            values=fun===partial_svd ? result.S : result.values
            @test values≈[1, 0.5] atol=1e-3 rtol=1e-3
            @test maximum(result.residuals)<1e-3
            close(work)
        end
        close(team)
        close(composed)
    end
end
