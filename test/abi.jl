for I in (Int32, Int64)
    @eval function abi_gemm(f, ::Type{$I}, A, B, C; ta='N')
        ccall(
            f,
            Cvoid,
            (
                Ref{UInt8},
                Ref{UInt8},
                Ref{$I},
                Ref{$I},
                Ref{$I},
                Ref{Float64},
                Ptr{Float64},
                Ref{$I},
                Ptr{Float64},
                Ref{$I},
                Ref{Float64},
                Ptr{Float64},
                Ref{$I},
            ),
            ta,
            'N',
            2,
            2,
            2,
            1.0,
            A,
            2,
            B,
            2,
            0.0,
            C,
            2,
        )
    end
    @eval function abi_iamax(f, ::Type{$I}, n, x)
        ccall(f, $I, (Ref{$I}, Ptr{Float32}, Ref{$I}), n, x, 1)
    end
end
@testset "LP64 and ILP64 native ABI" begin
    handle = Libdl.dlopen(Colmat.backend_library())
    last_error = Libdl.dlsym(handle, :colmat_blas_last_error)
    for (IntT, suffix) in ((Int32, "_"), (Int64, "_64_"))
        # Calls use each ABI's actual integer width, independently of Julia BlasInt.
        f = Libdl.dlsym(handle, "dgemm" * suffix)
        A = [1.0 2; 3 4]
        B = [2.0 1; 1 3]
        C = zeros(2, 2)
        # ccall argument types must be literal: specialize through a method.
        abi_gemm(f, IntT, A, B, C)
        @test C == [4.0 7; 10 15]
        @test ccall(last_error, Cint, ()) == 0
        abi_gemm(f, IntT, A, B, C; ta='?')
        @test ccall(last_error, Cint, ()) == 1
        @test C == [4.0 7; 10 15]
        @test ccall(last_error, Cint, ()) == 0
        iamax = Libdl.dlsym(handle, "isamax" * suffix)
        @test abi_iamax(iamax, IntT, -1, Float32[2, 9, 1]) == 0
        @test abi_iamax(iamax, IntT, 3, Float32[2, 9, 1]) == 2
    end
end

for I in (Int32, Int64), T in (Float32, Float64, ComplexF32, ComplexF64)
    @eval function abi_product(
        f, ::Type{$I}, A::Matrix{$T}, B::Matrix{$T}, C::Matrix{$T}, ta, tb
    )
        m, n = size(C)
        k = ta == 'N' ? size(A, 2) : size(A, 1)
        ccall(
            f,
            Cvoid,
            (
                Ref{UInt8},
                Ref{UInt8},
                Ref{$I},
                Ref{$I},
                Ref{$I},
                Ref{$T},
                Ptr{$T},
                Ref{$I},
                Ptr{$T},
                Ref{$I},
                Ref{$T},
                Ptr{$T},
                Ref{$I},
            ),
            ta,
            tb,
            m,
            n,
            k,
            one($T),
            A,
            stride(A, 2),
            B,
            stride(B, 2),
            zero($T),
            C,
            stride(C, 2),
        )
    end
end
@testset "All scalar types in both integer ABIs" begin
    h = Libdl.dlopen(Colmat.backend_library())
    for (I, suffix) in ((Int32, "_"), (Int64, "_64_")),
        (T, prefix) in
        ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z"))

        f = Libdl.dlsym(h, prefix * "gemm" * suffix)
        for ta in ('N', 'T', 'C'), tb in ('N', 'T', 'C')
            A = randn(T, ta == 'N' ? (7, 4) : (4, 7))
            B = randn(T, tb == 'N' ? (4, 5) : (5, 4))
            C = fill(T(NaN), 7, 5)
            abi_product(f, I, A, B, C, ta, tb)
            @test C ≈ product(op(A, ta), op(B, tb))
        end
    end
end
