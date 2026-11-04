using Colmat, LinearAlgebra, SparseArrays, Test

@testset "Native iteration callbacks and true-residual histories" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        solver in (cg, bicgstab, lsmr),
        storage in (:dense, :csc, :csr),
        Rhs in (T, ComplexF64),
        columns in (1, 2),
        native_rhs in (false, true)

        n=10
        A=T.(diagm(0=>collect(2.0:11.0), 1=>fill(0.15, n-1), -1=>fill(0.15, n-1)))
        if T<:Complex
            for i = 1:(n-1)
                A[i, i+1]+=T(0.1im)
                A[i+1, i]-=T(0.1im)
            end
        end
        if solver===lsmr
            A=vcat(A, T.([0.02i+0.03j for i = 1:2, j = 1:n]))
        end
        B=Rhs.([1+0.02i+(j==2 ? (-1)^i : 0) for i = 1:size(A, 1), j = 1:columns])
        Rhs<:Complex && (B .*= Rhs(1+0.2im))
        b=columns==1 ? vec(B) : B
        rhs=native_rhs ? (columns==1 ? FaerVector(b) : FaerMatrix(b)) : b
        source=storage===:dense ? FaerMatrix(A) :
               FaerSparseMatrix(sparse(A); format = storage)
        op=FaerLinearOperator(source)
        close(source)
        precond=solver===lsmr ? Diagonal(T.(range(0.7, 1.2; length = n))) : nothing
        states=Any[]
        callback=state->begin
            push!(states, deepcopy(state))
            fill!(state.x, 99) # A callback receives its own copy.
            state.iteration<2
        end
        result=solver(
            op,
            rhs;
            M = precond,
            history = true,
            callback,
            atol = 0,
            rtol = 0,
            maxiter = 20,
        )
        X=columns==1 ? Vector(result.x) : Matrix(result.x)
        @test result.reason===:stopped
        @test !result.converged
        @test !issuccess(result)
        @test result.iterations==2
        @test length(states)==length(result.history)==2
        @test X≈states[end].x
        @test states[1].x!=states[2].x
        P=eltype(X)
        tol=100eps(real(P))*max(norm(B), 1)
        for (state, entry) in zip(states, result.history)
            residual=P.(b)-P.(A)*state.x
            @test entry.iteration==state.iteration
            @test entry.abs_residual≈norm(residual) atol=tol rtol=1e-4
            @test entry.rel_residual≈norm(residual)/norm(B) atol=tol rtol=1e-4
            @test !hasproperty(entry, :x)
            if solver===lsmr
                normal=P.(precond)'*P.(A)'*residual
                normalref=P.(precond)'*P.(A)'*P.(b)
                @test entry.normal_abs_residual≈norm(normal) atol=tol rtol=1e-4
                @test entry.normal_rel_residual≈norm(normal)/norm(normalref) atol=tol rtol=1e-4
            else
                @test isnothing(entry.normal_abs_residual)
                @test isnothing(entry.normal_rel_residual)
            end
        end
        @test result.abs_residual≈result.history[end].abs_residual atol=tol rtol=1e-4
        native_rhs && close(rhs)
        result.x isa Union{FaerMatrix,FaerVector} && close(result.x)
        close(op)
    end
end

@testset "Continuing callbacks, convergence and empty histories" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), solver in (cg, bicgstab, lsmr)
        A=Diagonal(T.([1, 2, 3, 4]))
        b=ones(T, 4)
        visits=Int[]
        result=solver(
            A,
            b;
            history = true,
            callback = s->(push!(visits, s.iteration); nothing),
            rtol = 1e-4,
            maxiter = 40,
        )
        @test result.converged
        @test result.x≈A\b atol=1e-3 rtol=1e-3
        @test visits==collect(1:result.iterations)
        @test length(result.history)==result.iterations
        @test last(result.history).abs_residual≈result.abs_residual atol=1e-4
        @test isempty(solver(A, b; rtol = 1e-4, maxiter = 40).history)
        identity=FaerLinearOperator(I, 4; T)
        final=solver(identity, b; history = true, callback = s->true)
        @test final.iterations==length(final.history)==1
        @test final.x≈b
        @test solver(
            identity,
            b;
            x0 = b,
            history = true,
            callback = _->error("no update expected"),
        ).iterations==0
        initial=solver(
            identity,
            b;
            maxiter = 0,
            check = false,
            history = true,
            callback = _->error("no update expected"),
        )
        @test initial.reason===:no_convergence
        @test isempty(initial.history)
        close(identity)
    end
end

@testset "Callback stops and errors preserve in-place contracts" begin
    for (solver, solver!) in ((cg, cg!), (bicgstab, bicgstab!), (lsmr, lsmr!)),
        T in (Float32, Float64, ComplexF32, ComplexF64)

        A=Diagonal(T.([1, 2, 3, 4]))
        b=ones(T, 4)
        x=zeros(T, 4)
        result=solver!(x, A, b; history = true, callback = _->false)
        @test result.x===x
        @test result.reason===:stopped
        @test length(result.history)==result.iterations==1
        @test any(!iszero, x)
        fill!(x, 0)
        sentinel=ErrorException("iteration sentinel")
        caught=try
            solver!(x, A, b; callback = _->throw(sentinel))
            nothing
        catch error
            error
        end
        @test caught===sentinel
        @test x==zeros(T, 4)
        @test_throws ArgumentError solver!(x, A, b; callback = _->"stop")
        @test x==zeros(T, 4)
        @test solver(A, b; callback = _->false).reason===:stopped
    end
end

@testset "Nested callbacks and Julia Tasks" begin
    seen=Int[]
    A=Diagonal([1.0, 2.0, 3.0, 4.0])
    result=cg(
        A,
        ones(4);
        history = true,
        callback = state->begin
            inner=bicgstab(
                Matrix{Float64}(I, 2, 2),
                ones(2);
                history = true,
                callback = s->(push!(seen, s.iteration); true),
            )
            @test inner.x≈ones(2)
            state.iteration<2
        end,
    )
    @test seen==[1, 1]
    @test result.reason===:stopped
    task=@async lsmr(A, ones(4); history = true, callback = s->s.iteration<2)
    @test fetch(task).reason===:stopped
    @test length(fetch(task).history)==2
end

@testset "In-place native matrices, views, aliasing and checked failures" begin
    for (solver, solver!, Failure) in (
            (cg, cg!, FaerCgError),
            (bicgstab, bicgstab!, FaerBicgstabError),
            (lsmr, lsmr!, FaerLsmrError),
        ),
        T in (Float32, Float64, ComplexF32, ComplexF64)

        A=Diagonal(T.([1, 2, 3, 4]))
        original=T.([1 2; 2 -1; 3 1; 4 -2])
        for kind in (:native, :view, :julia)
            owner=FaerMatrix{T}(hcat(zeros(T, 4), original, zeros(T, 4)))
            dest=kind===:native ? FaerMatrix{T}(original) :
                 kind===:view ? view(owner, :, 2:3) : copy(original)
            observed=Ref{Any}()
            result=solver!(
                dest,
                A,
                dest;
                history = true,
                callback = state->(observed[] = copy(state.x); false),
            )
            @test result.x===dest
            @test result.reason===:stopped
            @test Matrix(dest)≈observed[]
            @test result.history[1].abs_residual≈norm(original-A*Matrix(dest)) atol=1e-3
            dest isa Union{FaerMatrix,FaerMatrixView} && close(dest)
            close(owner)
        end
        x=zeros(T, 4)
        caught=try
            solver!(
                x,
                A,
                ones(T, 4);
                history = true,
                callback = _->nothing,
                maxiter = 1,
                atol = 0,
                rtol = 0,
            )
            nothing
        catch error
            error
        end
        @test caught isa Failure
        @test x==zeros(T, 4)
        @test length(caught.result.history)==caught.result.iterations==1
        @test caught.result.reason===:no_convergence
    end
end


@testset "Histories preserve nonfinite failure diagnostics" begin
    # LSMR on this finite, tiny-scaled system returns a nonfinite failed iterate.
    # Observing it must preserve the unchecked result instead of constructing a
    # new finite-only input matrix and masking it with an input-validation error.
    A=Diagonal(fill(1e-300, 4))
    plain=lsmr(A, ones(4); check = false, maxiter = 2)
    observed=lsmr(A, ones(4); check = false, maxiter = 2, history = true)
    @test observed.reason===plain.reason
    @test isequal(observed.x, plain.x)
    @test isequal(observed.abs_residual, plain.abs_residual)
    @test length(observed.history)==observed.iterations
    @test any(entry->!isfinite(entry.abs_residual), observed.history)
end
