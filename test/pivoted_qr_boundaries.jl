using Test, Colmat, LinearAlgebra, Libdl
for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
    R=real(T)
    @eval function geqp3_boundary_probe(::Type{$T}, ::Type{$J}, p, m, n, lw)
        a=fill($T(777), 16)
        tau=fill($T(778), 8)
        pivot=fill($J(0), 8)
        work=fill($T(779), 32)
        rwork=fill($R(780), 16)
        info=fill($J(-777), 3)
        if $T<:Complex
            ccall(
                p,
                Cvoid,
                (
                    Ref{$J},
                    Ref{$J},
                    Ptr{$T},
                    Ref{$J},
                    Ptr{$J},
                    Ptr{$T},
                    Ptr{$T},
                    Ref{$J},
                    Ptr{$R},
                    Ptr{$J},
                ),
                m,
                n,
                a,
                max(1, m),
                pivot,
                tau,
                view(work, 2:31),
                lw,
                rwork,
                view(info, 2:2),
            )
        else
            ccall(
                p,
                Cvoid,
                (
                    Ref{$J},
                    Ref{$J},
                    Ptr{$T},
                    Ref{$J},
                    Ptr{$J},
                    Ptr{$T},
                    Ptr{$T},
                    Ref{$J},
                    Ptr{$J},
                ),
                m,
                n,
                a,
                max(1, m),
                pivot,
                tau,
                view(work, 2:31),
                lw,
                view(info, 2:2),
            )
        end
        (; a, tau, pivot, work, rwork, info)
    end
end
@testset "GEQP3 insufficient-work publication and query guards" begin
    native=Libdl.dlopen(Colmat.backend_library())
    external=Libdl.dlopen(
        only(
            filter(
                l->l.interface==:ilp64&&occursin("openblas", lowercase(l.libname)),
                BLAS.get_config().loaded_libs,
            ),
        ).libname,
    )
    error=Libdl.dlsym(native, :colmat_blas_last_error)
    for (T, prefix) in
        ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
        J in (Int32, Int64), (m, n) in ((1, 1), (0, 3), (3, 0)),
        lw in (0, -1)

        p=Libdl.dlsym(native, prefix*"geqp3"*(J==Int64 ? "_64_" : "_"))
        rp=Libdl.dlsym(external, prefix*"geqp3_64_")
        @test p!=rp
        own=geqp3_boundary_probe(T, J, p, m, n, lw)
        @test ccall(error, Cint, ())==(lw==0 ? 8 : 0)
        ref=geqp3_boundary_probe(T, Int64, rp, m, n, lw)
        @test ccall(error, Cint, ())==0
        for result in (own, ref)
            @test result.info==[-777, lw==0 ? -8 : 0, -777]
            minimum=min(m, n)==0 ? 1 : (T<:Complex ? n+1 : 3n+1)
            @test isreal(result.work[2])&&real(result.work[2])>=minimum
            @test result.work[1]==T(779)&&all(==(T(779)), result.work[3:end])
            @test result.a==fill(T(777), 16)&&result.tau==fill(T(778), 8)
            @test result.pivot==zeros(Int64, 8)&&result.rwork==fill(real(T)(780), 16)
            if min(m, n)==0
                @test result.work[2]==one(T)
            end
        end
    end
end
