using Test, LinearAlgebra, Random, Colmat

const dft_operations=(
    (lu, lu!, false),
    (full_piv_lu, full_piv_lu!, false),
    (qr, qr!, false),
    (
        (a; kwargs...)->qr(a, ColumnNorm(); kwargs...),
        (a; kwargs...)->qr!(a, ColumnNorm(); kwargs...),
        false,
    ),
    (cholesky, cholesky!, true),
    (ldlt, ldlt!, true),
    (bunchkaufman, bunchkaufman!, true),
)
dft_tol(T, n) = 200eps(typeof(real(zero(T))))*max(1, n)
function dft_check(f, a, tol)
    @test issuccess(f)
    @test Matrix(f)≈a rtol=tol atol=tol
    if size(a, 1)==size(a, 2) && !isempty(a)
        b=ones(eltype(a), size(a, 1), 2)
        x=Matrix(f\FaerMatrix(b))
        @test a*x≈b rtol=tol atol=tol
    end
end
@testset "Dense native construction with reusable teams" begin
    rng=MersenneTwister(3901)
    for T in (Float32, Float64, ComplexF32, ComplexF64), width in (1, 2)
        retained=[]
        team=FaerTeam(width)
        try
            for n in (0, 1, 5)
                raw=randn(rng, T, n, n)
                general=raw+T(n+1)*I
                positive=raw*raw'+T(n+1)*I
                for (fun, bang, selfadj) in dft_operations
                    a=Matrix(selfadj ? positive : general)
                    owner=FaerMatrix(a)
                    f=fun(owner; team)
                    dft_check(f, a, dft_tol(T, n))
                    @test Matrix(owner)==a
                    close(owner)
                    push!(retained, (f, a))
                    owner=FaerMatrix(a)
                    v=view(owner, :, :)
                    g=bang(owner; team)
                    dft_check(g, a, dft_tol(T, n))
                    @test Matrix(v)==Matrix(owner)
                    @test Matrix(owner)==Matrix(g.packed)
                    close(owner)
                    push!(retained, (g, a))
                end
            end
            # The selected triangle is authoritative, even when unused entries
            # contain NaNs. Upper writeback preserves the existing owner/view.
            for side in (:U, :L),
                (fun, bang) in
                ((cholesky, cholesky!), (ldlt, ldlt!), (bunchkaufman, bunchkaufman!))

                a=T[4 1; 1 3]
                owner=FaerMatrix(a)
                owner[side===:U ? 2 : 1, side===:U ? 1 : 2]=T(NaN)
                f=fun(Hermitian(owner, side); team)
                dft_check(f, a, dft_tol(T, 2))
                g=bang(Hermitian(owner, side); team)
                dft_check(g, a, dft_tol(T, 2))
                @test all(isfinite, Matrix(owner))
                close(owner)
                close(f)
                close(g)
            end
        finally
            close(team)
        end
        # No returned factor borrows the team or source matrix.
        GC.gc()
        for (f, a) in retained
            dft_check(f, a, dft_tol(T, size(a, 1)))
            g=copy(f)
            @test Matrix(g)≈a rtol=dft_tol(T, size(a, 1)) atol=dft_tol(T, size(a, 1))
            close(f)
            close(g)
        end
        for (fun, bang, _) in dft_operations
            a=FaerMatrix(Matrix{T}(I, 2, 2))
            saved=Matrix(a)
            @test_throws ArgumentError fun(a; team)
            @test_throws ArgumentError bang(a; team)
            @test_throws ArgumentError fun(a; team = 2)
            @test_throws ArgumentError bang(a; team = 2)
            @test Matrix(a)==saved
            close(a)
        end
    end
end
@testset "Rectangular QR and blocked dense team construction" begin
    rng=MersenneTwister(3902)
    FaerTeam(2) do team
        for T in (Float32, Float64, ComplexF32, ComplexF64),
            (m, n) in ((0, 4), (4, 0), (3, 7), (7, 3))

            a=randn(rng, T, m, n)
            for pivot in (NoPivot(), ColumnNorm())
                owner=FaerMatrix(a)
                f=qr(owner, pivot; team)
                @test size(f)==(m, n)
                @test Matrix(f)≈a rtol=dft_tol(T, max(m, n)) atol=dft_tol(T, max(m, n))
                g=qr!(owner, pivot; team)
                @test Matrix(g)≈a rtol=dft_tol(T, max(m, n)) atol=dft_tol(T, max(m, n))
                close(owner)
                close(f)
                close(g)
            end
        end
        for T in (Float32, Float64, ComplexF32, ComplexF64), (m, n) in ((4, 4), (6, 3))
            a=randn(rng, T, m, n)
            if m==n
                a+=T(5)*I
            end
            owner=FaerMatrix(a)
            f=factorize(owner; team)
            @test Matrix(f)≈a atol=dft_tol(T, max(m, n)) rtol=dft_tol(T, max(m, n))
            close(owner)
            close(f)
        end
        # Cross native blocked/parallel algorithm boundaries without timing claims.
        for T in (Float32, Float64, ComplexF32, ComplexF64)
            a=randn(rng, T, 129, 129)
            positive=a*a'+T(129)*I
            for (fun, selfadj) in
                ((lu, false), (full_piv_lu, false), (qr, false), (cholesky, true))
                data=Matrix(selfadj ? positive : a+T(129)*I)
                owner=FaerMatrix(data)
                f=fun(owner; team)
                dft_check(f, data, dft_tol(T, 129))
                close(f)
                close(owner)
            end
        end
        results=fetch.([@async begin
            a=FaerMatrix([4.0 1; 1 3])
            f=lu(a; team)
            close(a)
            f
        end for _ = 1:3])
        for f in results
            @test Matrix(f)≈[4.0 1; 1 3]
            close(f)
        end
    end
end
@testset "Dense team errors and current native input metadata" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        FaerTeam(2) do team
            for (fun, bang) in ((lu, lu!), (full_piv_lu, full_piv_lu!))
                a=FaerMatrix(T[1 0; 0 0])
                saved=Matrix(a)
                @test_throws SingularException bang(a; team)
                @test Matrix(a)==saved
                f=fun(a; check = false, team)
                @test !issuccess(f)
                close(f)
                close(a)
            end
            a=FaerMatrix(T[1 0; 0 -1])
            saved=Matrix(a)
            @test_throws PosDefException cholesky!(Hermitian(a); team)
            @test Matrix(a)==saved
            f=cholesky!(Hermitian(a); check = false, team)
            @test !issuccess(f)
            @test Matrix(a)==saved
            close(f)
            close(a)
            a=FaerMatrix(T[0 1; 1 0])
            f=bunchkaufman(Hermitian(a); team)
            dft_check(f, Matrix(a), dft_tol(T, 2))
            close(f)
            close(a)
            # Corrupt only cached Julia dimensions: native input must reject.
            a=FaerMatrix(Matrix{T}(I, 2, 2))
            a.dims=(3, 3)
            @test_throws Exception lu(a; team)
            @test_throws Exception qr!(a; team)
            a.dims=(2, 2)
            @test Matrix(a)==Matrix{T}(I, 2, 2)
            close(a)
        end
    end
    FaerTeam(2) do team
        a=FaerMatrix(Float32[2 0; 0 3])
        Colmat._native_call(
            Colmat.RustCrate.zeroes,
            a,
            UInt(2),
            UInt(2),
            Colmat._tag(ComplexF64),
        )
        @test_throws Exception lu(a; team)
        @test_throws Exception qr!(a; team)
        @test_throws Exception lu(a)  # same boundary check on default route
        close(a)
    end
end
