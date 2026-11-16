using Test, LinearAlgebra, Random, Libdl, Colmat
isdefined(@__MODULE__, :rook_factor!)||include("rook_helpers.jl")
@testset "ROOK compact factors interoperate with independent LAPACK" begin
    rng=MersenneTwister(372)
    reference=rook_reference()
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        uplo in ('L', 'U'),
        route in (:dll, :lbt),
        n in (0, 1, 2, 5, 8),
        unblocked in (false, true)

        @testset "$T/$J/$uplo/$route/$n/$unblocked" begin
            x=randn(rng, T, n, n)
            original=x+x'
            if n>=2
                original[1, 1]=original[2, 2]=0
                original[1, 2]=T<:Real ? T(3) : T(3+0.2im)
                original[2, 1]=conj(original[1, 2])
            end
            rhs=randn(rng, T, n, 3)
            tol=1000max(1, n)*eps(real(T))
            for producer in (:native, :reference)
                K=producer==:native ? J : Int64
                library=producer==:native ? Colmat.backend_library() : reference
                a=rook_stage(original, uplo)
                old=copy(a)
                if !unblocked
                    q=rook_factor!(
                        a,
                        K;
                        uplo,
                        lwork=-1,
                        library,
                        pointer=if producer==:native
                            rook_pointer(T, J, "trf", route)
                        else
                            nothing
                        end,
                    )
                    @test q.info==q.diagnostic==0 && real(q.work[1])>=1
                    @test isequal(a, old)
                    @test all(==(rook_sentinel(K)), q.piv)
                end
                factor=rook_factor!(
                    a,
                    K;
                    uplo,
                    unblocked,
                    library,
                    pointer=if producer==:native
                        rook_pointer(T, J, unblocked ? "tf2" : "trf", route)
                    else
                        nothing
                    end,
                )
                @test factor.info==factor.diagnostic==0
                rook_matrix_guards(a, old, uplo)
                @test norm(rook_reconstruct(a, factor.piv, uplo)-original)<=tol*max(
                    1, norm(original)
                )
                @test all(p->1<=abs(p)<=n, factor.piv)
                for consumer in (:native, :reference)
                    L=consumer==:native ? J : Int64
                    target=consumer==:native ? Colmat.backend_library() : reference
                    b=vcat(rhs, fill(T(777), 2, 3))
                    before=copy(a)
                    result=rook_solve!(
                        a,
                        L.(factor.piv),
                        b;
                        uplo,
                        library=target,
                        pointer=if consumer==:native
                            rook_pointer(T, J, "trs", route)
                        else
                            nothing
                        end,
                    )
                    @test result.info==result.diagnostic==0
                    @test isequal(a, before)
                    @test all(==(T(777)), b[(n + 1):end, :])
                    @test norm(original*b[1:n, :]-rhs)<=tol*max(
                        1, norm(original)*norm(b[1:n, :]), norm(rhs)
                    )
                    inverse=copy(a)
                    result=rook_inverse!(
                        inverse,
                        L.(factor.piv);
                        uplo,
                        library=target,
                        pointer=if consumer==:native
                            rook_pointer(T, J, "tri", route)
                        else
                            nothing
                        end,
                    )
                    @test result.info==result.diagnostic==0
                    rook_matrix_guards(inverse, a, uplo)
                    invmatrix=Matrix(Hermitian(inverse[1:n, :], Symbol(uplo)))
                    @test norm(original*invmatrix-I)<=tol*max(
                        1, norm(original)*norm(invmatrix)
                    )
                end
            end
            a=rook_stage(original, uplo)
            old=copy(a)
            b=vcat(rhs, fill(T(777), 2, 3))
            oldb=copy(b)
            q=rook_driver!(a, b, J; uplo, lwork=-1, pointer=rook_pointer(T, J, "sv", route))
            @test q.info==q.diagnostic==0 && real(q.work[1])>=1
            @test isequal(a, old) && b==oldb
            result=rook_driver!(a, b, J; uplo, pointer=rook_pointer(T, J, "sv", route))
            @test result.info==result.diagnostic==0
            rook_matrix_guards(a, old, uplo)
            @test all(==(T(777)), b[(n + 1):end, :])
            @test norm(original*b[1:n, :]-rhs)<=tol*max(
                1, norm(original)*norm(b[1:n, :]), norm(rhs)
            )
        end
    end
end

@testset "ROOK chain search moves both rows of a two-column pivot" begin
    reference=rook_reference()
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64), uplo in ('L', 'U'), route in (:dll, :lbt),
        unblocked in (false, true)

        n=6
        original=zeros(T, n, n)
        for i in 1:(n - 1)
            value=T<:Real ? T(exp2(i-1)) : T(exp2(i-1)*(1+0.2im))
            original[i + 1, i]=value
            original[i, i + 1]=conj(value)
        end
        uplo=='U' && (original=reverse(reverse(original; dims=1); dims=2))
        for producer in (:native, :reference)
            K=producer==:native ? J : Int64
            library=producer==:native ? Colmat.backend_library() : reference
            pointer=if producer==:native
                rook_pointer(T, J, unblocked ? "tf2" : "trf", route)
            else
                nothing
            end
            a=rook_stage(original, uplo)
            result=rook_factor!(a, K; uplo, unblocked, library, pointer)
            @test result.info==result.diagnostic==0
            k=uplo=='L' ? 1 : n
            other=uplo=='L' ? 2 : n-1
            @test result.piv[k]<0 && result.piv[other]<0
            @test abs(result.piv[k])!=k && abs(result.piv[other])!=other
            tol=1000n*eps(real(T))
            @test norm(rook_reconstruct(a, result.piv, uplo)-original)<=tol*norm(original)
            target=producer==:native ? reference : Colmat.backend_library()
            L=producer==:native ? Int64 : J
            pointer=producer==:native ? nothing : rook_pointer(T, J, "trs", route)
            b=vcat(Matrix{T}(I, n, n), fill(T(777), 2, n))
            solved=rook_solve!(a, L.(result.piv), b; uplo, library=target, pointer)
            @test solved.info==solved.diagnostic==0
            @test norm(original*b[1:n, :]-I)<=tol*max(1, norm(original)*norm(b[1:n, :]))
        end
    end
end
