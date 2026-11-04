using Colmat, LinearAlgebra, SparseArrays, Test
@testset "borrowed sparse QR rank-certificate revisions" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        algorithm in (:auto, :simplicial, :supernodal),
        format in (:csc, :csr)

        A=T[4 0.02 0.01; 0.03 5 0.02; 0.01 0.02 6; 0.02 0.01 0.02; 0.01 0.02 0.03]
        T<:Complex&&(A .+= T(0.001im))
        a=FaerSparseMatrix(sparse(A); format)
        f=qr(a; algorithm, ordering = [3, 1, 2])
        g=copy(f)
        d=deepcopy(f)
        b=ones(T, 5, 2)
        full=scratch_requirements(ldiv!, f, b)
        work=FaerScratch(ldiv!, f, b)
        @test !Colmat._native_call(Colmat.RustCrate.sparse_qr_needs_spectral_solve, f)
        @test scratch_requirements(ldiv!, g, b)==full
        D=copy(A)
        D[:, 3]=D[:, 1]
        def=FaerSparseMatrix(sparse(D); format)
        @test qr!(f, def)===f
        @test Colmat._native_call(Colmat.RustCrate.sparse_qr_needs_spectral_solve, f)
        @test scratch_requirements(ldiv!, f, b).bytes>full.bytes
        @test scratch_requirements(ldiv!, g, b)==full
        @test scratch_requirements(ldiv!, d, b)==full
        x=fill(T(9), 3, 2)
        before=copy(x)
        @test_throws ArgumentError ldiv!(x, f, b; scratch = work)
        @test x==before
        reserve!(work, ldiv!, f, b)
        @test ldiv!(x, f, b; scratch = work)≈pinv(D)*b atol=1e-3
        @test Matrix(g\b)≈A\b atol=1e-3
        @test Matrix(d\b)≈A\b atol=1e-3
        @test qr!(f, a)===f
        @test !Colmat._native_call(Colmat.RustCrate.sparse_qr_needs_spectral_solve, f)
        @test scratch_requirements(ldiv!, f, b)==full
        close(f.source)
        close(f.plan)
        @test ldiv!(x, f, b; scratch = work)≈A\b atol=1e-3
        close(work)
        close(f)
        close(g)
        close(d)
        close(def)
        if algorithm!==:simplicial
            plan=FaerSparseSymbolic(a; kind = :qr, algorithm, ordering = [3, 1, 2])
            construction=FaerScratch(factorize, plan, a)
            h=factorize(plan, a; scratch = construction)
            @test !Colmat._native_call(Colmat.RustCrate.sparse_qr_needs_spectral_solve, h)
            @test scratch_requirements(ldiv!, h, b)==full
            close(h)
            close(construction)
            close(plan)
        end
        close(a)
    end
end
@testset "cached sparse QR source precision" begin
    for T in (Float32, ComplexF32), algorithm in (:auto, :simplicial, :supernodal)
        a=FaerSparseMatrix(sparse(T[1 0; 0 1e-8; 0 0]))
        f=qr(a; algorithm)
        @test Colmat._native_call(Colmat.RustCrate.sparse_qr_needs_spectral_solve, f)
        @test f\ones(T, 3)≈T[1, 0] atol=1e-4
        @test f\ones(ComplexF64, 3)≈ComplexF64[1, 0] atol=1e-4
        close(f)
        close(a)
    end
end
