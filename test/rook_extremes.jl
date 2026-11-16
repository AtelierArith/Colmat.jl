using Test, LinearAlgebra, Libdl, Colmat
isdefined(@__MODULE__, :rook_factor!)||include("rook_helpers.jl")
@testset "ROOK uniformly extreme two-by-two blocks" begin
    reference=rook_reference()
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64), uplo in ('L', 'U'),
        exponent in (real(T)==Float32 ? (0, 100, -100) : (0, 900, -900))

        @testset "$T/$J/$uplo/$exponent" begin
            coupling=T<:Real ? T(2) : T(2+3im)
            scale=real(T)(exp2(Float64(exponent)))
            original=T[1 coupling; conj(coupling) -1] .* scale
            desired=T[1 2; -2 1]
            rhs=original*desired
            tol=500eps(real(T))
            for producer in (:native, :reference)
                K=producer==:native ? J : Int64
                library=producer==:native ? Colmat.backend_library() : reference
                a=rook_stage(original, uplo)
                result=rook_factor!(a, K; uplo, library)
                @test result.info==result.diagnostic==0
                @test all(isfinite, diag(a))
                for consumer in (:native, :reference)
                    L=consumer==:native ? J : Int64
                    target=consumer==:native ? Colmat.backend_library() : reference
                    b=vcat(rhs, fill(T(777), 2, 2))
                    solved=rook_solve!(a, L.(result.piv), b; uplo, library=target)
                    @test solved.info==solved.diagnostic==0
                    @test all(isfinite, b) && norm(b[1:2, :]-desired)<=tol*norm(desired)
                    inverse=copy(a)
                    inverted=rook_inverse!(inverse, L.(result.piv); uplo, library=target)
                    @test inverted.info==inverted.diagnostic==0
                    matrix=Matrix(Hermitian(inverse[1:2, :], Symbol(uplo)))
                    @test all(isfinite, matrix) && norm(original*matrix-I)<=tol
                end
            end
        end
    end
end
