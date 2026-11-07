using Test, Colmat, LinearAlgebra
function dqp_fixture(T, m, n, deficient)
    a=T[0.04sin(i+2j) for i = 1:m, j = 1:n]
    for i = 1:min(m, n)
        a[i, i]+=T(3+i)
    end
    T<:Complex && (a .+= T(0.03im) .* T[cos(i+3j) for i = 1:m, j = 1:n])
    if deficient && min(m, n)>0
        m>n ? (a[:, end].=a[:, 1]) : (a[end, :].=a[1, :])
    end
    a
end
dqp_tol(T, n) = 500eps(real(T))*max(n, 1)
@testset "saved dense rectangular/truncated QR actions" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        pivoted in (false, true),
        (m, n, deficient) in (
            (0, 3, false),
            (3, 0, false),
            (0, 0, false),
            (1, 4, false),
            (4, 1, false),
            (7, 3, false),
            (3, 7, false),
            (7, 3, true),
            (3, 7, true),
            (4, 4, true),
        ),
        S in (T, ComplexF64)

        a=dqp_fixture(T, m, n, deficient)
        source=FaerMatrix(a)
        f=pivoted ? qr(source, ColumnNorm()) : qr(source)
        rt=min(m, n)*eps(real(T))
        op=m==n ? FaerLinearOperator(f; rtol = rt) : FaerLinearOperator(f)
        close(source)
        close(f)
        reference=pinv(a; rtol = rt)
        @test size(op)==(n, m)
        P=promote_type(T, S)
        tol=dqp_tol(T, max(m, n))
        for (g, ref) in (
            (op, reference),
            (transpose(op), transpose(reference)),
            (adjoint(op), adjoint(reference)),
            (conj(op), conj(reference)),
        )
            b=S[0.1+i+2j for i = 1:size(ref, 2), j = 1:2]
            S<:Complex && (b .*= S(1+0.2im))
            gold=P.(ref)*P.(b)
            result=g*b
            @test eltype(result)===P
            @test Matrix(result)≈gold atol=tol*max(norm(gold), 1) rtol=tol
            close(result)
            team=FaerTeam(2)
            dest=zeros(P, size(ref, 1), 2)
            work=FaerScratch(mul!, dest, g, b; team)
            capacity=scratch_capacity(work)
            for _ = 1:2
                @test mul!(dest, g, b; scratch = work, team)===dest
                @test dest≈gold atol=tol*max(norm(gold), 1) rtol=tol
                @test scratch_capacity(work)==capacity
            end
            req=scratch_requirements(mul!, dest, g, b; team)
            if req.bytes>0
                short=FaerScratch(req.bytes-1; alignment = req.alignment)
                before=copy(dest)
                @test_throws ArgumentError mul!(dest, g, b; scratch = short, team)
                @test dest==before
                reserve!(short, mul!, dest, g, b; team)
                @test mul!(dest, g, b; scratch = short, team)===dest
                close(short)
            end
            close(work)
            close(team)
            g===op || close(g)
        end
        copied=deepcopy(op)
        close(op)
        @test Matrix(copied)≈reference atol=tol*max(norm(reference), 1) rtol=tol
        @test typeof(copied)===typeof(op)
        close(copied)
    end
end
@testset "QR cutoff, square compatibility and cached promotion" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), pivot in (false, true)
        a=T[1 0 0; 0 0.01 0]
        f=pivot ? qr(FaerMatrix(a), ColumnNorm()) : qr(FaerMatrix(a))
        for (at, rt) in ((0.0, 0.0), (0.02, 0.0), (0.0, 0.02), (Inf, 0.0), (0.0, Inf))
            op=FaerLinearOperator(f; atol = at, rtol = rt)
            @test Matrix(op)≈pinv(a; atol = at, rtol = rt) atol=dqp_tol(T, 3)
            close(op)
        end
        @test_throws ArgumentError FaerLinearOperator(f; atol = -1)
        @test_throws ArgumentError FaerLinearOperator(f; rtol = NaN)
        close(f)
        zero_f=pivot ? qr(FaerMatrix(zeros(T, 2, 2)), ColumnNorm()) :
               qr(FaerMatrix(zeros(T, 2, 2)))
        @test_throws Colmat.RustCall.RustError FaerLinearOperator(zero_f)
        op=FaerLinearOperator(zero_f; rtol = 0)
        @test Matrix(op)==zeros(T, 2, 2)
        close(op)
        close(zero_f)
    end
    a=Float32[1 0 0; 0 1e-9 0]
    f=qr(FaerMatrix(a), ColumnNorm())
    op=FaerLinearOperator(f)
    close(f)
    b=ones(ComplexF64, 2, 1)
    @test Matrix(op*b)≈ComplexF64[1; 0; 0;;]
    close(op)
end
@testset "blocked QR reflector actions and composition" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        (m, n) in ((129, 65), (33, 70)),
        pivot in (false, true)

        a=dqp_fixture(T, m, n, false)
        f=pivot ? qr(FaerMatrix(a), ColumnNorm()) : qr(FaerMatrix(a))
        op=FaerLinearOperator(f)
        close(f)
        ref=pinv(a)
        tol=dqp_tol(T, max(m, n))
        for (g, r) in (
            (op, ref),
            (transpose(op), transpose(ref)),
            (adjoint(op), adjoint(ref)),
            (conj(op), conj(ref)),
        )
            b=ones(T, size(r, 2), 3)
            @test Matrix(g*b)≈r*b atol=tol*max(norm(r*b), 1) rtol=tol
            g===op || close(g)
        end
        combined=FaerLinearOperator(I, n; T)*op
        close(op)
        @test Matrix(combined)≈ref atol=tol*max(norm(ref), 1) rtol=tol
        close(combined)
    end
end
@testset "zero truncation skips overflowing projections" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        a=T[4 1; 1 3; 0 1]
        f=qr(FaerMatrix(a))
        spectral=svd(FaerMatrix(a); native = true)
        for op in
            (FaerLinearOperator(f; rtol = Inf), FaerLinearOperator(spectral; rtol = Inf))
            for g in (op, transpose(op), adjoint(op), conj(op))
                b=fill(T(floatmax(real(T))), size(g, 2), 2)
                @test Matrix(g*b)==zeros(T, size(g, 1), 2)
                @test scratch_requirements(*, g, b).bytes==0
                g===op || close(g)
            end
            close(op)
        end
        close(f)
        close(spectral)
    end
    value=nextfloat(Float32(0.1))
    a=Float32[1 0 0; 0 value 0]
    f=qr(FaerMatrix(a))
    keep=FaerLinearOperator(f; rtol = Float64(value)-eps(value)/4)
    drop=FaerLinearOperator(f; rtol = value)
    @test Matrix(keep)≈pinv(a; rtol = Float64(value)-eps(value)/4)
    @test Matrix(drop)≈pinv(a; rtol = value)
    @test Matrix(keep)[2, 2]>0
    @test Matrix(drop)[2, 2]==0
    close(keep)
    close(drop)
    close(f)
end
@testset "QR snapshot guards, tasks and native destinations" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        a=dqp_fixture(T, 5, 3, false)
        f=qr(FaerMatrix(a), ColumnNorm())
        op=FaerLinearOperator(f)
        b=ones(T, 5, 2)
        native=FaerMatrix(zeros(T, 3, 2))
        @test mul!(native, op, b)===native
        @test Matrix(native)≈pinv(a)*b atol=dqp_tol(T, 5)
        team=FaerTeam(2)
        work=FaerScratch(mul!, native, op, b; team)
        dests=[zeros(T, 3, 2) for _ = 1:4]
        @sync for dest in dests
            @async mul!(dest, op, b; scratch = work, team)
        end
        @test all(x->x≈pinv(a)*b, dests)
        close(work)
        close(team)
        close(native)
        close(op)
        f.dims=(6, 3)
        @test_throws Colmat.RustCall.RustError FaerLinearOperator(f)
        f.dims=(5, 3)
        close(f)
        @test_throws ArgumentError FaerLinearOperator(f)
    end
end

@testset "wide QR with an empty square prefix" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), pivot in (false, true)
        a=T.(hcat(zeros(3, 3), diagm([1.0, 2, 3]), ones(3)))
        T<:Complex && (a[:, 7].+=T(0.2im))
        f=pivot ? qr(FaerMatrix(a), ColumnNorm()) : qr(FaerMatrix(a))
        op=FaerLinearOperator(f)
        close(f)
        tol=500eps(real(T))*7
        @test Matrix(op)≈pinv(a) atol=tol rtol=tol
        @test Matrix(op')≈pinv(a)' atol=tol rtol=tol
        close(op)
    end
end

@testset "pinv of saved dense QR factors matches pinv(A; atol, rtol)" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        kind in (:qr, :pivoted, :inplace),
        (m, n, deficient) in (
            (0, 3, false),
            (3, 0, false),
            (0, 0, false),
            (1, 4, false),
            (4, 1, false),
            (7, 3, false),
            (3, 7, false),
            (7, 3, true),
            (3, 7, true),
            (4, 4, false),
            (4, 4, true),
            (300, 4, false),
            (300, 4, true),
            (4, 120, true),
        )

        a=dqp_fixture(T, m, n, deficient)
        source=FaerMatrix(a)
        f=kind===:qr ? qr(source) :
          kind===:pivoted ? qr(source, ColumnNorm()) :
          qr!(source; scratch = FaerScratch(qr!, source))
        tol=dqp_tol(T, max(m, n))
        for kw in (
            (;),
            (; rtol = sqrt(eps(real(T)))),
            (; atol = real(T)(0.5)),
            (; atol = 0.1, rtol = 0.2),
            (; rtol = Inf),
        )
            reference=pinv(a; kw...)
            p=pinv(f; kw...)
            @test p isa FaerMatrix{T}
            @test size(p)==(n, m)
            @test Matrix(p)≈reference atol=tol*max(norm(reference), 1) rtol=tol
            close(p)
        end
        if kind!==:inplace
            team=FaerTeam(2)
            req=scratch_requirements(pinv, f; team)
            work=FaerScratch(pinv, f; team)
            reference=pinv(a)
            for _ = 1:2
                p=pinv(f; scratch = work, team)
                @test Matrix(p)≈reference atol=tol*max(norm(reference), 1) rtol=tol
                close(p)
            end
            if req.bytes>0
                short=FaerScratch(req.bytes-1; alignment = req.alignment)
                @test_throws ArgumentError pinv(f; scratch = short, team)
                reserve!(short, pinv, f; team)
                p=pinv(f; scratch = short, team)
                @test Matrix(p)≈reference atol=tol*max(norm(reference), 1) rtol=tol
                close(p)
                close(short)
            end
            close(work)
            close(team)
        else
            team=FaerTeam(2)
            p=pinv(f; team)
            @test Matrix(p)≈pinv(a) atol=tol*max(norm(pinv(a)), 1) rtol=tol
            close(p)
            close(team)
        end
        @test_throws ArgumentError pinv(f; atol = -1)
        @test_throws ArgumentError pinv(f; rtol = NaN)
        close(f)
        close(source)
    end
end
