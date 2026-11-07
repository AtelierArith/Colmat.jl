using Test, Colmat, LinearAlgebra
jcb_op(a; counter = Ref(0)) = FaerLinearOperator(
    (y, x)->(counter[]+=1; mul!(y, a, x)),
    (y, x)->(counter[]+=1; mul!(y, a', x)),
    size(a)...;
    T = eltype(a),
)
jcb_tol(T, n) = 500eps(real(T))*max(n, 1)
@testset "Julia callbacks through native LinOp/BiLinOp" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        dims in ((0, 0), (0, 3), (3, 0), (1, 1), (5, 3), (3, 5))

        m, n=dims
        a=T[sin(i+2j)+(i==j ? 3 : 0) for i = 1:m, j = 1:n]
        T<:Complex && (a .+= T(0.2im))
        op=jcb_op(a)
        @test size(op)==dims
        @test_throws ArgumentError op.coefficients
        @test Matrix(op)≈a atol=jcb_tol(T, max(m, n))
        for (g, ref) in (
            (op, a),
            (transpose(op), transpose(a)),
            (adjoint(op), adjoint(a)),
            (conj(op), conj(a)),
        )
            b=ones(T, size(ref, 2), 2)
            expected=ref*b
            @test Matrix(g*b)≈expected atol=jcb_tol(T, max(m, n))
            @test Matrix(ones(T, 2, size(ref, 1))*g)≈ones(T, 2, size(ref, 1))*ref atol=jcb_tol(
                T,
                max(m, n),
            )
            team=FaerTeam(2)
            dest=fill(T(42), size(ref, 1), 2)
            work=FaerScratch(mul!, dest, g, b; team)
            cap=scratch_capacity(work)
            for _ = 1:2
                @test mul!(dest, g, b; scratch = work, team)===dest
                @test dest≈expected atol=jcb_tol(T, max(m, n))
                @test scratch_capacity(work)==cap
            end
            saved=copy(dest)
            @test mul!(dest, g, b, T(2), T(3); scratch = work, team)===dest
            @test dest≈2expected+3saved atol=10jcb_tol(T, max(m, n))
            req=scratch_requirements(mul!, dest, g, b; team)
            if req.bytes>0
                short=FaerScratch(req.bytes-1; alignment = req.alignment)
                saved=copy(dest)
                @test_throws ArgumentError mul!(dest, g, b; scratch = short, team)
                @test dest==saved
                reserve!(short, mul!, dest, g, b; team)
                @test mul!(dest, g, b; scratch = short, team)===dest
                close(short)
            end
            close(work)
            close(team)
            g===op || close(g)
        end
        copyop=deepcopy(op)
        close(op)
        @test typeof(copyop)==typeof(op)
        @test Matrix(copyop)≈a atol=jcb_tol(T, max(m, n))
        close(copyop)
    end
end
@testset "callback ownership, promotion, nesting and escaped arrays" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        a=T[4 1; 2 3]
        T<:Complex && (a[1, 2]+=T(im))
        op=jcb_op(a)
        for S in (Float32, Float64, ComplexF32, ComplexF64)
            b=fill(S<:Complex ? S(1+0.2im) : one(S), 2, 2)
            P=promote_type(T, S)
            @test Matrix(op*b)≈P.(a)*P.(b) atol=jcb_tol(T, 2)
            composed=op*FaerLinearOperator(I, 2; T = S)
            @test Matrix(composed*b)≈P.(a)*P.(b) atol=jcb_tol(T, 2)
            close(composed)
        end
        nested=FaerLinearOperator(
            (y, x)->copyto!(y, Matrix(op*x)),
            (y, x)->copyto!(y, Matrix(op'*x)),
            2,
            2;
            T,
        )
        @test Matrix(nested)≈a atol=jcb_tol(T, 2)
        escaped=Any[]
        f=(y, x)->begin
            push!(escaped, (y, x))
            mul!(y, a, x)
            fill!(x, T(99))
            GC.gc(false)
        end
        outer=FaerLinearOperator(f, (y, x)->mul!(y, a', x), 2, 2; T)
        b=ones(T, 2, 2)
        original=copy(b)
        @test Matrix(outer*b)≈a*b atol=jcb_tol(T, 2)
        @test b==original
        old=copy(escaped[1][1])
        close(outer)
        GC.gc(false)
        @test escaped[1][1]==old
        @test escaped[1][2]==fill(T(99), 2, 2)
        left=copy(op)
        composition=left*op
        close(left)
        close(nested)
        close(op)
        GC.gc(false)
        @test Matrix(composition)≈a*a atol=5jcb_tol(T, 2)
        close(composition)
    end
end
@testset "callback failures preserve outputs and restore scopes" begin
    err=ErrorException("user action failure")
    good=jcb_op([4.0 1; 1 3])
    for f in ((y, x)->throw(err), (y, x)->nothing, (y, x)->fill!(y, NaN))
        bad=FaerLinearOperator(f, f, 2, 2)
        out=fill(42.0, 2, 2)
        saved=copy(out)
        caught=try
            mul!(out, bad, ones(2, 2))
            nothing
        catch e
            e
        end
        @test caught isa Exception
        @test !isnothing(caught)
        @test out==saved
        @test Matrix(good*ones(2, 2))≈[5.0 5; 4 4]
        close(bad)
    end
    bad=FaerLinearOperator((y, x)->throw(err), (y, x)->throw(err), 2, 2)
    got=try
        bad*ones(2)
        nothing
    catch e
        e
    end
    @test got===err
    @test_throws ErrorException cg(bad, ones(2))
    close(bad)
    @test_throws ArgumentError FaerLinearOperator(identity, identity, -1, 2)
    @test_throws ArgumentError FaerLinearOperator(identity, identity, 2, 2; T = BigFloat)
    close(good)
end
@testset "native iterative solvers and callback preconditioners" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        solver in (cg, bicgstab, lsmr),
        observing in (false, true)

        a=Matrix(Diagonal(T.(2:9)))
        solver===lsmr && (a=vcat(a, zeros(T, 2, 8)))
        count=Ref(0)
        op=jcb_op(a; counter = count)
        inverse=Matrix(Diagonal(T.(1 ./ (2:9))))
        precond=jcb_op(inverse)
        b=a*ones(T, 8)
        tol=512eps(real(T))
        result=solver(op, b; M = precond, rtol = tol, check = false, history = observing)
        @test result.converged
        @test a*Vector(result.x)≈b atol=10tol rtol=10tol
        @test count[]>0
        observing && @test !isempty(result.history)
        close(op)
        close(precond)
    end
    for solver in (cg, bicgstab, lsmr)
        a=Matrix(Diagonal(collect(2.0:9.0)))
        op=jcb_op(a)
        precond=jcb_op(Matrix{Float64}(I, 8, 8))
        seen=Int[]
        stop=s->(push!(seen, s.iteration); false)
        result=solver(
            op,
            ones(8);
            M = precond,
            callback = stop,
            history = true,
            check = false,
            rtol = 0,
            atol = 0,
            maxiter = 8,
        )
        @test result.reason==:stopped
        @test seen==[1]
        @test length(result.history)==1
        # A native coefficient operator with only a callback preconditioner also
        # establishes the callback scope from the optional owner argument.
        native=FaerLinearOperator(a)
        result=solver(native, ones(8); M = precond, check = false)
        @test result.converged
        close(native)
        close(op)
        close(precond)
    end
end
@testset "callback partial spectra" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        a=Matrix(Diagonal(T.(1:96)))
        op=jcb_op(a)
        tol=real(T)===Float32 ? 2e-4 : 1e-8
        for solver in (partial_eigen, partial_self_adjoint_eigen, partial_svd)
            result=solver(op, 2; tol, max_restarts = 20)
            @test result.algorithm===:partial
            @test result.nconverged==2
            @test maximum(result.residuals)<100tol*96
            values=solver===partial_svd ? result.S : result.values
            @test values≈T[96, 95] rtol=100tol
        end
        close(op)
    end
end
function jcb_lifetime_fixture()
    state=Ref(2.0)
    f=(y, x)->copyto!(y, state[] .* x)
    return FaerLinearOperator(f, f, 2, 2), WeakRef(state)
end
# Cross a function boundary before checking release so the test's active SSA
# values cannot retain a previously read callback dictionary or captured state.
function jcb_released_fixture()
    op, weak=jcb_lifetime_fixture()
    held=copy(op)
    close(op)
    GC.gc()
    alive=weak.value!==nothing
    product=Matrix(held*ones(2, 1))
    close(held)
    return weak, alive, product
end
@testset "callback context lifetime, tasks, alias and error recovery" begin
    weak, alive, product=jcb_released_fixture()
    @test alive
    @test product==fill(2.0, 2, 1)
    GC.gc()
    @test weak.value===nothing
    a=ComplexF64[4 im; 1 3]
    op=jcb_op(a)
    g=copy(op)
    a[1, 1]+=1
    @test Matrix(op)≈a
    @test Matrix(g)≈a
    b=ones(ComplexF64, 2, 2)
    native=FaerMatrix(b)
    @test mul!(native, op, native)===native
    @test Matrix(native)≈a*b
    @test lmul!(op, native)===native
    @test Matrix(native)≈a*a*b
    close(native)
    outputs=[zeros(ComplexF64, 2, 2) for _ = 1:4]
    FaerTeam(2) do team
        work=FaerScratch(mul!, outputs[1], op, b; team)
        @sync for dst in outputs
            @async mul!(dst, op, b; scratch = work, team)
        end
        @test all(x->x≈a*b, outputs)
        close(work)
    end
    calls=Ref(0)
    empty=FaerLinearOperator(
        (y, x)->(calls[]+=1; error("zero-column callback ran")),
        identity,
        2,
        2,
    )
    @test size(empty*zeros(2, 0))==(2, 0)
    @test calls[]==0
    close(empty)
    err=ErrorException("scoped action and observer failure")
    bad=FaerLinearOperator((y, x)->throw(err), (y, x)->throw(err), 2, 2; T = ComplexF64)
    dest=FaerMatrix(fill(ComplexF64(42), 2, 2))
    saved=Matrix(dest)
    @test_throws ErrorException mul!(dest, bad, b)
    @test Matrix(dest)==saved
    @test_throws ErrorException lmul!(bad, dest)
    @test Matrix(dest)==saved
    for solve! in (cg!, bicgstab!, lsmr!)
        dst=fill(42.0, 2)
        @test_throws ErrorException solve!(dst, bad, ones(ComplexF64, 2))
        @test dst==fill(42.0, 2)
    end
    for solve in (cg, bicgstab, lsmr)
        positive=jcb_op([4.0 1; 1 3])
        caught=try
            solve(positive, ones(2); callback = s->throw(err), history = true)
            nothing
        catch e
            e
        end
        @test caught===err
        @test Matrix(positive)≈[4.0 1; 1 3]
        close(positive)
    end
    right=jcb_op(Matrix(Diagonal([0.25, 1/3])))
    result=bicgstab(jcb_op(Matrix(Diagonal([4.0, 3]))), ones(2); N = right)
    @test result.converged
    @test Vector(result.x)≈[0.25, 1/3]
    close(right)
    close(dest)
    close(bad)
    close(g)
    close(op)
end

@testset "migratable task callback scope" begin
    for fail in (false, true)
        task=Threads.@spawn begin
            before=current_task().sticky
            ids=Int[]
            action=(y, x)->begin
                push!(ids, Threads.threadid())
                for _ = 1:10
                    yield()
                    push!(ids, Threads.threadid())
                end
                fail && error("thread scope test")
                copyto!(y, x)
            end
            op=FaerLinearOperator(action, action, 2, 2)
            caught=try
                op*ones(2)
                nothing
            catch e
                e
            end
            after=current_task().sticky
            close(op)
            return before, after, ids, caught
        end
        before, after, ids, caught=fetch(task)
        @test !before
        @test before==after
        @test length(unique(ids))==1
        @test fail ? caught isa ErrorException : isnothing(caught)
    end
end

@testset "arbitrary Julia thrown values survive the callback boundary" begin
    bad=FaerLinearOperator((y, x)->throw(nothing), (y, x)->throw(nothing), 2, 2)
    caught=try
        bad*ones(2)
        :no_error
    catch e
        Some(e)
    end
    @test caught===Some(nothing)
    close(bad)
    good=jcb_op([4.0 1; 1 3])
    caught=try
        cg(good, ones(2); callback = s->throw(nothing))
        :no_error
    catch e
        Some(e)
    end
    @test caught===Some(nothing)
    @test Matrix(good)≈[4.0 1; 1 3]
    close(good)
end

# Callback marshalling reuses owner buffers, so a stale result from a previous
# application must never satisfy the overwrite check of a later one.
@testset "reused callback buffers reject stale outputs and recursion" begin
    calls=Ref(0)
    lazy=FaerLinearOperator(
        (y, x)->(calls[]+=1; calls[]==1 && copyto!(y, x)),
        (y, x)->copyto!(y, x),
        2,
        2,
    )
    @test Matrix(lazy*ones(2, 1))==ones(2, 1)
    out=fill(42.0, 2, 1)
    @test_throws ArgumentError mul!(out, lazy, ones(2, 1))
    @test out==fill(42.0, 2, 1)
    close(lazy)
    for T in (Float64, ComplexF64)
        a=T[2 1; 0 3]
        depth=Ref(0)
        holder=Ref{Any}()
        f=(y, x)->begin
            depth[]+=1
            if depth[]==1
                inner=Matrix(holder[]*x)
                mul!(y, a, inner)
            else
                mul!(y, a, x)
            end
            depth[]-=1
            y
        end
        op=FaerLinearOperator(f, (y, x)->mul!(y, a', x), 2, 2; T)
        holder[]=op
        b=T[1 2; 3 4]
        @test Matrix(op*b)≈a*a*b
        @test Matrix(op*b)≈a*a*b
        promoted=op*FaerLinearOperator(I, 2; T = ComplexF32)
        @test Matrix(promoted*ComplexF32.(b))≈a*a*b rtol=1e-5
        close(promoted)
        close(op)
    end
end

function jcb_bytes_per_application(n)
    a=Matrix(Diagonal(collect(2.0:(n+1))))
    count=Ref(0)
    op=FaerLinearOperator(
        (y, x)->(count[]+=1; mul!(y, a, x)),
        (y, x)->(count[]+=1; mul!(y, a', x)),
        n,
        n,
    )
    b=a*ones(n)
    run(maxiter)=cg(op, b; maxiter, check = false, rtol = 0, atol = 0)
    run(5)
    count[]=0
    short=@allocated run(5)
    c1=count[]
    count[]=0
    long=@allocated run(25)
    c2=count[]
    close(op)
    return (long-short)/(c2-c1)
end
@testset "callback marshalling allocation does not scale with dimension" begin
    small=jcb_bytes_per_application(16)
    large=jcb_bytes_per_application(512)
    @test large<1024
    @test abs(large-small)<256
end
