using Colmat, LinearAlgebra, Test
irc_array(x) = x isa AbstractVector ? Vector(x) : Matrix(x)
@testset "Iteration callbacks and Julia operators with reserved resources" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        (fun, bang) in ((cg, cg!), (bicgstab, bicgstab!), (lsmr, lsmr!)),
        julia_operator in (false, true)

        A=T[5 0.1 0.2; 0.1 6 0.1; 0.2 0.1 7]
        T<:Complex&&(A[1, 2]+=T(0.1im); A[2, 1]-=T(0.1im))
        calls=Int[]
        forward=(y, x)->begin
            push!(calls, Threads.threadid())
            yield()
            mul!(y, A, x)
        end
        backward=(y, x)->begin
            push!(calls, Threads.threadid())
            yield()
            mul!(y, A', x)
        end
        op=julia_operator ? FaerLinearOperator(forward, backward, 3, 3; T) :
           FaerLinearOperator(A)
        b=T[1, 2, 3]
        team=FaerTeam(2)
        work=FaerScratch(fun, op, b; team)
        @test isempty(calls) # Query does not apply a Julia operator.
        rtol=typeof(real(zero(T)))===Float32 ? 1e-4 : 1e-10
        tol=typeof(real(zero(T)))===Float32 ? 1e-3 : 1e-7
        snapshots=Vector{Vector{T}}()
        ids=Int[]
        task=current_task()
        sticky=task.sticky
        task.sticky=false
        try
            result=fun(
                op,
                b;
                scratch = work,
                team,
                rtol,
                history = true,
                callback = s->begin
                    push!(ids, Threads.threadid())
                    @test current_task().sticky
                    push!(snapshots, s.x)
                    yield()
                    true
                end,
            )
            @test !task.sticky
            @test result.converged
            @test irc_array(result.x)≈pinv(A)*b atol=tol rtol=tol
            @test length(result.history)==length(snapshots)
            @test all(==(Threads.threadid()), ids)
            @test all(==(Threads.threadid()), calls)
            saved=copy(irc_array(result.x))
            isempty(snapshots)||(snapshots[1].=0)
            @test irc_array(result.x)==saved
            stopped=fun(
                op,
                b;
                scratch = work,
                team,
                rtol,
                history = true,
                callback = s->false,
            )
            @test stopped.reason==:stopped
            @test !stopped.converged
            x=zeros(T, 3)
            stopped_bang=bang(x, op, b; scratch = work, team, rtol, callback = s->false)
            @test stopped_bang.x===x
            @test x≈irc_array(stopped.x) atol=tol rtol=tol
            before=copy(x)
            @test_throws ErrorException bang(
                x,
                op,
                b;
                scratch = work,
                team,
                rtol,
                callback = s->error("iteration test error"),
            )
            @test x==before
            @test !task.sticky
            @test fun(op, b; scratch = work, team, rtol).converged
            @test_throws Nothing bang(
                x,
                op,
                b;
                scratch = work,
                team,
                rtol,
                callback = s->throw(nothing),
            )
            @test x==before
            @test !task.sticky
            @test fun(op, b; scratch = work, team, rtol).converged
        finally
            task.sticky=sticky
        end
        close(op)
        close(work)
        close(team)
    end
end
@testset "Shared iterative resources and callback-error recovery" begin
    A=[5.0 0.1; 0.1 6]
    fail=Ref(false)
    op=FaerLinearOperator(
        (y, x)->begin
            fail[]&&error("operator test error")
            mul!(y, A, x)
        end,
        (y, x)->mul!(y, A', x),
        2,
        2,
    )
    for (fun, bang) in ((cg, cg!), (bicgstab, bicgstab!), (lsmr, lsmr!))
        team=FaerTeam(2)
        work=FaerScratch(fun, op, ones(2); team)
        x=fill(19.0, 2)
        before=copy(x)
        fail[]=true
        @test_throws ErrorException bang(x, op, ones(2); scratch = work, team)
        @test x==before
        fail[]=false
        @test fun(op, ones(2); scratch = work, team).converged
        jobs=[@async begin
            yield()
            fun(op, ones(2); scratch = work, team)
        end for _ = 1:3]
        @test all(r->r.converged, fetch.(jobs))
        short=FaerScratch(0; alignment = 1)
        counter=Ref(0)
        @test_throws ArgumentError fun(
            op,
            ones(2);
            scratch = short,
            team,
            callback = s->(counter[]+=1),
        )
        @test counter[]==0
        close(short)
        close(work)
        close(team)
    end
    close(op)
end
