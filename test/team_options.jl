using Test, Colmat, LinearAlgebra

@testset "FaerTeam pool options and diagnostics" begin
    t=FaerTeam(3; spin_polls = 0, qos = :utility, pin = false, acquire_timeout = 5)
    @test t.options==(
        acquire_timeout = 5.0,
        spin_polls = 0,
        qos = :utility,
        pin = false,
        dispatch_cost = nothing,
    )
    A=FaerMatrix(reshape(sin.(1:(300*300)), 300, 300))
    for _ = 1:3
        @test Matrix(team_mul(t, A, A))≈Matrix(A)*Matrix(A)
    end
    d=team_diagnostics(t)
    @test d.width==min(3, Sys.CPU_THREADS)
    @test d.spin_polls==0
    @test !d.pin
    @test d.pin_skips==0
    @test all(>=(0), values(d))
    # copy builds an independent pool with the same options.
    c=copy(t)
    @test c.options==t.options
    @test team_diagnostics(c).spin_polls==0
    close(t)
    @test_throws ArgumentError team_diagnostics(t)
    @test Matrix(team_mul(c, A, A))≈Matrix(A)*Matrix(A)
    close(c)
    # Defaults keep threadteam's spin budget and pinning.
    FaerTeam(2) do t
        d=team_diagnostics(t)
        @test d.pin
        @test d.spin_polls>0
        @test t.options==(
            acquire_timeout = 30.0,
            spin_polls = nothing,
            qos = nothing,
            pin = true,
            dispatch_cost = nothing,
        )
        @test d.dispatch_cost>0
    end
    for q in (:default, :utility, :user_initiated, :user_interactive)
        FaerTeam(2; qos = q) do t
            @test Matrix(team_mul(t, A, A))≈Matrix(A)*Matrix(A)
        end
    end
    # A width-one team builds no pool.
    FaerTeam(1; spin_polls = 7) do t
        @test team_diagnostics(t)==(
            width = 1,
            spin_polls = 0,
            pin = false,
            resident_members = 0,
            pin_skips = 0,
            join_sleeps = 0,
            barrier_sleeps = 0,
            member_parks = 0,
            dispatch_cost = 0.0,
        )
        @test Matrix(lu(A; team = t).U)≈Matrix(lu(Matrix(A)).U)
    end
    # Options reach the team-aware factorizations.
    FaerTeam(2; spin_polls = 0, pin = false) do t
        F=lu(A; team = t)
        @test Matrix(F.L)*Matrix(F.U)≈Matrix(A)[F.p, :]
    end
    # Team dispatch costs feed faer's split decisions; results are unchanged.
    for (cost, w, expected) in (
        (2e-6, 2, 2e-6),
        ([2=>1e-6, 4=>3e-6], 2, 1e-6),
        ([2=>1e-6, 4=>3e-6], 3, 2e-6),
        (1e3, 2, 1e3),
        (0, 2, 0.0),
    )
        FaerTeam(w; dispatch_cost = cost) do t
            @test team_diagnostics(t).dispatch_cost≈expected
            @test Matrix(team_mul(t, A, A))≈Matrix(A)*Matrix(A)
            F=lu(A; team = t)
            @test Matrix(F.L)*Matrix(F.U)≈Matrix(A)[F.p, :]
            c=copy(t)
            @test team_diagnostics(c).dispatch_cost≈expected
            close(c)
        end
    end
    @test FaerTeam(2; dispatch_cost = [2=>1e-6]) do t
        t.options.dispatch_cost
    end==[2=>1e-6]
    for bad in ([1=>1e-6], [4=>1e-6, 2=>1e-6], [2=>-1.0], [2=>NaN], Pair{Int,Float64}[], :fast,
        [w=>1e-6 for w = 2:100])
        @test_throws ArgumentError FaerTeam(2; dispatch_cost = bad)
    end
    @test_throws ArgumentError FaerTeam(2; qos = :fast)
    @test_throws ArgumentError FaerTeam(2; acquire_timeout = 0)
    @test_throws ArgumentError FaerTeam(2; acquire_timeout = Inf)
    @test_throws ArgumentError FaerTeam(2; spin_polls = -1)
    @test_throws ArgumentError FaerTeam(2; spin_polls = 2^33)
    @test_throws MethodError FaerTeam(2; unknown = 1)
    @test_throws ArgumentError FaerTeam(0; pin = false)
    holder=Ref{Any}()
    @test_throws ErrorException FaerTeam(2; spin_polls = 0) do t
        holder[]=t
        error("exercise cleanup")
    end
    @test !isopen(holder[])
end
