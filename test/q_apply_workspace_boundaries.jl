using Test, Colmat, LinearAlgebra, Libdl
function q_apply_workspace_diagnostic()
    return ccall(
        Libdl.dlsym(Libdl.dlopen(Colmat.backend_library()), :colmat_blas_last_error),
        Cint,
        (),
    )
end
for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
    @eval function q_apply_workspace_probe(
        ::Type{$T}, ::Type{$J}, p, side, trans, m, n, k, lda, lwork
    )
        a=fill($T(NaN), 16)
        tau=fill($T(NaN), 4)
        c=fill($T(777), 64)
        work=fill($T(778), 16)
        info=fill($J(-777), 3)
        ccall(
            p,
            Cvoid,
            (
                Ref{UInt8},
                Ref{UInt8},
                Ref{$J},
                Ref{$J},
                Ref{$J},
                Ptr{$T},
                Ref{$J},
                Ptr{$T},
                Ptr{$T},
                Ref{$J},
                Ptr{$T},
                Ref{$J},
                Ptr{$J},
            ),
            UInt8(side),
            UInt8(trans),
            m,
            n,
            k,
            a,
            lda,
            tau,
            c,
            max(1, m),
            view(work, 2:15),
            lwork,
            view(info, 2:2),
        )
        (; a, tau, c, work, info)
    end
end
@testset "Q application empty and K=0 WORK contracts" begin
    native=Libdl.dlopen(Colmat.backend_library())
    reference=Libdl.dlopen(
        only(
            filter(
                l->l.interface==:ilp64&&occursin("openblas", lowercase(l.libname)),
                BLAS.get_config().loaded_libs,
            ),
        ).libname,
    )
    @test q_apply_workspace_diagnostic()==0
    for (T, prefix) in
        ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
        J in (Int32, Int64),
        orientation in ("qr", "lq", "ql", "rq"),
        (m, n) in ((0, 0), (0, 3), (3, 0), (2, 3)),
        side in ('L', 'R'),
        trans in ('N', T<:Complex ? 'C' : 'T')

        family=(T<:Complex ? "unm" : "orm")*orientation
        ownp=Libdl.dlsym(native, prefix*family*(J==Int64 ? "_64_" : "_"))
        refp=Libdl.dlsym(reference, prefix*family*"_64_")
        @test refp!=Libdl.dlsym(native, prefix*family*"_64_")
        minimum=max(1, side=='L' ? n : m)
        order=side=='L' ? m : n
        lda=orientation in ("lq", "rq") ? 1 : max(1, order)
        for lw in (-1, minimum-1, minimum, minimum+2)
            own=q_apply_workspace_probe(T, J, ownp, side, trans, m, n, 0, lda, lw)
            @test q_apply_workspace_diagnostic()==(lw==minimum-1 ? 12 : 0)
            ref=q_apply_workspace_probe(T, Int64, refp, side, trans, m, n, 0, lda, lw)
            @test q_apply_workspace_diagnostic()==0
            expected_info=lw==minimum-1 ? -12 : 0
            @test own.info==ref.info==[-777, expected_info, -777]
            @test all(isnan, own.a)&&isequal(own.a, ref.a)
            @test all(isnan, own.tau)&&isequal(own.tau, ref.tau)
            @test own.c==ref.c==fill(T(777), 64)
            @test own.work[[1, end]]==ref.work[[1, end]]==fill(T(778), 2)
            @test own.work[3:end]==ref.work[3:end]==fill(T(778), 14)
            if lw==-1
                @test real(own.work[2])>=minimum
                @test real(ref.work[2])>=1
                if orientation in ("ql", "rq")&&(m==0||n==0)
                    @test ref.work[2]==one(T)
                end
            elseif expected_info==0
                if m==0||n==0||orientation in ("qr", "lq")
                    @test own.work[2]==ref.work[2]==one(T)
                else
                    @test own.work[2]==T(minimum)
                    @test real(ref.work[2])>=minimum
                end
            end
        end
    end
end
