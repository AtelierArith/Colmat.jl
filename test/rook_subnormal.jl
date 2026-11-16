using Test, LinearAlgebra, Libdl, Colmat
isdefined(@__MODULE__, :rook_factor!)||include("rook_helpers.jl")
# Exact identity solutions, rather than a norm divided by the largest input
# block, ensure that a tiny block cannot be hidden by a normal block.
@testset "Subnormal BK blocks retain finite identity solutions" begin
    reference=rook_reference()
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        uplo in ('L', 'U'),
        route in (:dll, :lbt),
        classic in (false, true),
        n in (1, 2, 4, 6),
        mixed in (false, true)

        n<4 && mixed && continue
        @testset "$T/$J/$uplo/$route/$classic/$n/$mixed" begin
            tiny=real(T)(exp2(real(T)==Float32 ? -140 : -1040))
            @test tiny>0 && issubnormal(tiny)
            off=T<:Real ? T(tiny) : T(tiny, tiny/2)
            original=zeros(T, n, n)
            if n==1
                original[1, 1]=tiny
            else
                for k in 1:2:(n - 1)
                    value=mixed && k==n-1 ? (T<:Real ? T(1) : T(1, 0.5)) : off
                    original[k + 1, k]=value
                    original[k, k + 1]=conj(value)
                end
            end
            prefix=if T==Float32
                "s"
            elseif T==Float64
                "d"
            elseif T==ComplexF32
                "c"
            else
                "z"
            end
            kind=T<:Real ? "sy" : "he"
            stem=prefix*kind
            suffix=classic ? "" : "_rook"
            # OpenBLAS 0.3.30 classic TRF itself yields NaN factors for
            # multiple raw subnormal blocks. Its result is not an oracle there;
            # require exact reconstruction and the known identity solution.
            for producer in (classic && n>=4 ? (:native,) : (:native, :reference))
                K=producer==:native ? J : Int64
                library=producer==:native ? Colmat.backend_library() : reference
                pointer=if producer==:native
                    rook_pointer(T, J, "trf", route; classic)
                else
                    Libdl.dlsym(Libdl.dlopen(reference), stem*"trf"*suffix*"_64_")
                end
                a=rook_stage(original, uplo)
                factor=rook_factor!(a, K; uplo, library, pointer)
                @test factor.info==factor.diagnostic==0
                @test all(isfinite, diag(a))
                @test rook_reconstruct(a, factor.piv, uplo; classic)==original
                # Netlib's 1x1 inverse-times-RHS path can also overflow. The
                # exact-one invariant is the oracle for that case, not Netlib.
                for consumer in (n==1 ? (:native,) : (:native, :reference))
                    L=consumer==:native ? J : Int64
                    target=consumer==:native ? Colmat.backend_library() : reference
                    pointer=if consumer==:native
                        rook_pointer(T, J, "trs", route; classic)
                    else
                        Libdl.dlsym(Libdl.dlopen(reference), stem*"trs"*suffix*"_64_")
                    end
                    b=vcat(original, fill(T(777), 2, n))
                    solved=rook_solve!(a, L.(factor.piv), b; uplo, library=target, pointer)
                    @test solved.info==solved.diagnostic==0
                    @test all(isfinite, b[1:n, :])
                    @test norm(b[1:n, :]-I)<=200eps(real(T))*n
                end
            end
            a=rook_stage(original, uplo)
            b=vcat(original, fill(T(777), 2, n))
            result=rook_driver!(
                a, b, J; uplo, pointer=rook_pointer(T, J, "sv", route; classic)
            )
            @test result.info==result.diagnostic==0
            @test all(isfinite, b[1:n, :]) && norm(b[1:n, :]-I)<=200eps(real(T))*n
            # An inverse has entries beyond the scalar range here. This tests
            # representable solves, not an impossible finite inverse contract.
        end
    end
end

@testset "Nonzero L across subnormal congruence scales" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        uplo in ('L', 'U'),
        route in (:dll, :lbt),
        classic in (false, true),
        mixed in (false, true)

        tiny=real(T)(exp2(real(T)==Float32 ? -140 : -1040))
        off=T<:Real ? T(tiny) : T(tiny, tiny/2)
        original=T[4tiny off; conj(off) (mixed ? 4 : 4tiny)]
        a=rook_stage(original, uplo)
        factored=rook_factor!(a, J; uplo, pointer=rook_pointer(T, J, "trf", route; classic))
        @test factored.info==factored.diagnostic==0
        @test rook_reconstruct(a, factored.piv, uplo; classic)==original
        b=vcat(original, fill(T(777), 2, 2))
        solved=rook_solve!(
            a, factored.piv, b; uplo, pointer=rook_pointer(T, J, "trs", route; classic)
        )
        @test solved.info==solved.diagnostic==0
        @test all(isfinite, b) && norm(b[1:2, :]-I)<=200eps(real(T))
        @test all(==(T(777)), b[3:end, :])
        a=rook_stage(original, uplo)
        b=vcat(original, fill(T(777), 2, 2))
        result=rook_driver!(a, b, J; uplo, pointer=rook_pointer(T, J, "sv", route; classic))
        @test result.info==result.diagnostic==0
        @test all(isfinite, b) && norm(b[1:2, :]-I)<=200eps(real(T))
        @test all(==(T(777)), b[3:end, :])
    end
end
