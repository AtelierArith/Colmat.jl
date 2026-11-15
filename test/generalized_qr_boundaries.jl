using Test, LinearAlgebra, Libdl, Colmat
@testset "Generalized drivers exact argument diagnostics and workspace boundaries" begin
    ref=generalized_qr_reference()
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        lse in (false, true)

        base=lse ? (3, 4, 2) : (3, 2, 2)
        cases=[
            (-1, base[2], base[3], 5, 6, 20, 1),
            (base[1], -1, base[3], 5, 6, 20, 2),
            (base[1], base[2], -1, 5, 6, 20, 3),
            (base..., 0, 6, 20, 5),
            (base..., 5, 0, 20, 7),
            (base..., 5, 6, sum(base)-1, 12),
            (base..., 5, 6, -2, 12),
        ]
        if lse
            push!(cases, (3, 4, 5, 5, 6, 20, 3), (1, 4, 2, 5, 6, 20, 3))
        else
            push!(cases, (3, 4, 2, 5, 6, 20, 2), (4, 1, 2, 5, 6, 20, 3))
        end
        for (f, s, p, lda, ldb, lw, arg) in cases,
            (library, K) in ((Colmat.backend_library(), J), (ref, Int64))

            a=fill(T(777), 5, 4)
            b=fill(T(777), 6, 4)
            rhs=fill(T(777), 8)
            second=fill(T(777), 8)
            last=fill(T(777), 8)
            saved=deepcopy((a, b, rhs, second, last))
            result=direct_generalized_driver!(
                Val(lse), a, b, rhs, second, last, f, s, p, K; library, lda, ldb, lwork=lw
            )
            @test result.info == -arg
            @test isequal((a, b, rhs, second, last), saved)
        end
    end
end
@testset "Generalized drivers distinguish the two exact rank failures" begin
    ref=generalized_qr_reference()
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64), lse in (false, true),
        failure in (1, 2)

        for (library, K) in ((Colmat.backend_library(), J), (ref, Int64))
            if lse
                # B rank failure first; otherwise A's free-coordinate triangular factor fails.
                a=zeros(T, 4, 2)
                b=zeros(T, 3, 2)
                a[1, 1]=failure==1 ? 1 : 0
                a[2, 2]=1
                b[1, 2]=failure==2 ? 1 : 0
                firstdim, seconddim, p=2, 2, 1
                rhs=T[777, 1, 2, 777]
                second=T[777, 3, 777]
                last=fill(T(777), 4)
            else
                a=zeros(T, 4, 1)
                b=zeros(T, 4, 2)
                a[1, 1]=failure==1 ? 1 : 0
                b[1, 1]=1
                b[2, 2]=failure==2 ? 1 : 0
                firstdim, seconddim, p=2, 1, 2
                rhs=T[777, 1, 2, 777]
                second=fill(T(777), 3)
                last=fill(T(777), 4)
            end
            result=direct_generalized_driver!(
                Val(lse), a, b, rhs, second, last, firstdim, seconddim, p, K; library
            )
            @test result.info==failure
            if failure==1
                @test all(==(T(777)), last)
                !lse && @test all(==(T(777)), second)
            elseif !lse
                @test all(==(T(777)), second)
            end
        end
    end
end
