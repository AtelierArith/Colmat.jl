using Test, LinearAlgebra, Libdl, Colmat
isdefined(@__MODULE__, :rook_factor!)||include("rook_helpers.jl")
@testset "ROOK singular factors, failed inverse and driver" begin
    reference=rook_reference()
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        uplo in ('L', 'U'),
        route in (:dll, :lbt),
        n in (1, 2, 5),
        rankone in (false, true),
        unblocked in (false, true)

        rankone && n==1 && continue
        original=fill(T(rankone ? 1 : 0), n, n)
        a=rook_stage(original, uplo)
        reference_a=copy(a)
        expected=rook_factor!(reference_a, Int64; uplo, unblocked, library=reference)
        result=rook_factor!(
            a,
            J;
            uplo,
            unblocked,
            pointer=rook_pointer(T, J, unblocked ? "tf2" : "trf", route),
        )
        @test 0<result.info==expected.info<=n
        @test result.diagnostic==expected.diagnostic==0
        @test all(p->1<=abs(p)<=n, result.piv)
        for factors in (:native, :reference)
            matrix=factors==:native ? a : reference_a
            piv=factors==:native ? result.piv : J.(expected.piv)
            inverse=copy(matrix)
            ref_inverse=copy(matrix)
            expected_inverse=rook_inverse!(
                ref_inverse, Int64.(piv); uplo, library=reference
            )
            failed=rook_inverse!(
                inverse, piv; uplo, pointer=rook_pointer(T, J, "tri", route)
            )
            @test 0<failed.info==expected_inverse.info<=n
            @test failed.diagnostic==expected_inverse.diagnostic==0
            @test isequal(inverse, matrix)
        end
        matrix=rook_stage(original, uplo)
        b=fill(T(19), n+2, 3)
        oldb=copy(b)
        driver=rook_driver!(matrix, b, J; uplo, pointer=rook_pointer(T, J, "sv", route))
        @test driver.info==expected.info && driver.diagnostic==0
        @test b==oldb
    end
end
@testset "ROOK argument and integer boundaries preserve payloads" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        route in (:dll, :lbt)

        a=rook_stage(Matrix{T}(I, 2, 2), 'L')
        old=copy(a)
        b=fill(T(19), 4, 2)
        oldb=copy(b)
        piv=J[1, 2]
        for unblocked in (false, true)
            pointer=rook_pointer(T, J, unblocked ? "tf2" : "trf", route)
            for (kwargs, arg) in ((; uplo=('?'))=>1, (; n=-1)=>2, (; lda=1)=>4)
                r=rook_factor!(a, J; unblocked, pointer, kwargs...)
                @test r.info==-arg && r.diagnostic==arg && isequal(a, old)
                @test all(==(rook_sentinel(J)), r.piv)
            end
            if !unblocked
                for work in (-2, 0)
                    r=rook_factor!(a, J; pointer, lwork=work)
                    @test r.info==-7 && r.diagnostic==7 && isequal(a, old)
                end
            end
            if J==Int64
                r=rook_factor!(a, J; unblocked, pointer, n=Int64(1)<<32, lda=1)
                @test r.info==-4 && r.diagnostic==4 && isequal(a, old)
                r=rook_factor!(
                    a, J; unblocked, pointer, n=typemax(Int64), lda=typemax(Int64)
                )
                @test r.info==-4 && r.diagnostic==4 && isequal(a, old)
            end
        end
        pointer=rook_pointer(T, J, "trs", route)
        for (kwargs, arg) in
            ((; uplo=('?'))=>1, (; n=-1)=>2, (; nrhs=-1)=>3, (; lda=1)=>5, (; ldb=1)=>8)
            r=rook_solve!(a, piv, b; pointer, kwargs...)
            @test r.info==-arg && r.diagnostic==arg && isequal(a, old) && b==oldb
        end
        if J==Int64
            r=rook_solve!(a, piv, b; pointer, nrhs=typemax(Int64))
            @test r.info==-8 && r.diagnostic==8 && b==oldb
        end
        pointer=rook_pointer(T, J, "tri", route)
        for (kwargs, arg) in ((; uplo=('?'))=>1, (; n=-1)=>2, (; lda=1)=>4)
            r=rook_inverse!(a, piv; pointer, kwargs...)
            @test r.info==-arg && r.diagnostic==arg && isequal(a, old)
        end
        pointer=rook_pointer(T, J, "sv", route)
        for (kwargs, arg) in (
            (; uplo=('?'))=>1,
            (; n=-1)=>2,
            (; nrhs=-1)=>3,
            (; lda=1)=>5,
            (; ldb=1)=>8,
            (; lwork=0)=>10,
            (; lwork=-2)=>10,
        )
            r=rook_driver!(a, b, J; pointer, kwargs...)
            @test r.info==-arg && r.diagnostic==arg && isequal(a, old) && b==oldb
        end
        # Zero right-hand sides must not inspect factor payloads, while SYSV
        # still factorizes its matrix even when the RHS has zero columns.
        empty_rhs=zeros(T, 4, 0)
        before=copy(a)
        r=rook_solve!(
            a,
            J[typemin(J), typemin(J)],
            empty_rhs;
            pointer=rook_pointer(T, J, "trs", route),
        )
        @test r.info==r.diagnostic==0 && isequal(a, before)
        r=rook_driver!(a, empty_rhs, J; pointer=rook_pointer(T, J, "sv", route))
        @test r.info==r.diagnostic==0 && all(p->1<=abs(p)<=2, r.piv)
    end
end
