using Test, LinearAlgebra, Random, Libdl, Colmat

for (T, prefix, family) in (
        (Float32, "s", "syevx"),
        (Float64, "d", "syevx"),
        (ComplexF32, "c", "heevx"),
        (ComplexF64, "z", "heevx"),
    ),
    J in (Int32, Int64)

    R=real(T)
    symbol=Symbol(prefix*family*"_"*(J===Int64 ? "64_" : ""))
    @eval function direct_selected_selfadjoint!(
        a::Matrix{$T},
        ::Type{$J};
        n=size(a, 2),
        jobz=('V'),
        range=('A'),
        uplo=('L'),
        vl=($R(NaN)),
        vu=($R(NaN)),
        il=-777,
        iu=-777,
        abstol=zero($R),
        lda=size(a, 1),
        ldz=max(1, n+2),
        lwork=nothing,
        library=Colmat.backend_library(),
    )
        order=max(0, n)
        minimum=order<=1 ? 1 : ($T<:Complex ? 2order : 8order)
        lw=isnothing(lwork) ? minimum : lwork
        work=fill($T(777), max(1, lw)+2)
        rwork=fill($R(777), max(1, 7order)+2)
        iwork=fill($J(777), max(1, 5order)+2)
        failed=fill($J(777), max(1, order)+2)
        values=fill($R(777), max(1, order)+2)
        z=fill($T(777), max(1, ldz), max(1, order))
        count=Ref{$J}(-777)
        info=Ref{$J}(-777)
        wp=view(work, 2:(length(work) - 1))
        rp=view(rwork, 2:(length(rwork) - 1))
        ip=view(iwork, 2:(length(iwork) - 1))
        fp=view(failed, 2:(length(failed) - 1))
        sp=view(values, 2:(length(values) - 1))
        f=Libdl.dlsym(Libdl.dlopen(library), $(QuoteNode(symbol)))
        if $T<:Real
            ccall(
                f,
                Cvoid,
                (
                    Ref{UInt8},
                    Ref{UInt8},
                    Ref{UInt8},
                    Ref{$J},
                    Ptr{$T},
                    Ref{$J},
                    Ref{$R},
                    Ref{$R},
                    Ref{$J},
                    Ref{$J},
                    Ref{$R},
                    Ref{$J},
                    Ptr{$R},
                    Ptr{$T},
                    Ref{$J},
                    Ptr{$T},
                    Ref{$J},
                    Ptr{$J},
                    Ptr{$J},
                    Ref{$J},
                ),
                jobz,
                range,
                uplo,
                n,
                a,
                lda,
                vl,
                vu,
                il,
                iu,
                abstol,
                count,
                sp,
                z,
                ldz,
                wp,
                lw,
                ip,
                fp,
                info,
            )
        else
            ccall(
                f,
                Cvoid,
                (
                    Ref{UInt8},
                    Ref{UInt8},
                    Ref{UInt8},
                    Ref{$J},
                    Ptr{$T},
                    Ref{$J},
                    Ref{$R},
                    Ref{$R},
                    Ref{$J},
                    Ref{$J},
                    Ref{$R},
                    Ref{$J},
                    Ptr{$R},
                    Ptr{$T},
                    Ref{$J},
                    Ptr{$T},
                    Ref{$J},
                    Ptr{$R},
                    Ptr{$J},
                    Ptr{$J},
                    Ref{$J},
                ),
                jobz,
                range,
                uplo,
                n,
                a,
                lda,
                vl,
                vu,
                il,
                iu,
                abstol,
                count,
                sp,
                z,
                ldz,
                wp,
                lw,
                rp,
                ip,
                fp,
                info,
            )
        end
        (; info=info[], count=count[], values, z, work, rwork, iwork, failed)
    end
end

function selected_selfadjoint_storage(original, uplo)
    n=size(original, 1)
    T=eltype(original)
    a=fill(T(777), n+3, n)
    for j in 1:n, i in 1:n
        if i==j
            a[i, j]=T<:Complex ? T(real(original[i, j]), NaN) : original[i, j]
        elseif (uplo=='L' && i>j)||(uplo=='U' && i<j)
            a[i, j]=original[i, j]
        else
            a[i, j]=T(NaN)
        end
    end
    return a
end

@testset "Selected self-adjoint EVD four types and direct integer ABIs" begin
    reference=only(
        filter(
            l->l.interface==:ilp64 && occursin("openblas", lowercase(basename(l.libname))),
            BLAS.get_config().loaded_libs,
        ),
    ).libname
    rng=MersenneTwister(274)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        n in (0, 1, 4, 9),
        uplo in ('L', 'U'),
        jobz in ('N', 'V'),
        range in ('A', 'I', 'V'),
        abstol in (zero(real(T)), -one(real(T)), floatmin(real(T)))

        R=real(T)
        tolerance=R===Float32 ? 3e-4 : 5e-12
        raw=randn(rng, T, n, n)
        original=raw+raw'
        a=selected_selfadjoint_storage(original, uplo)
        prior=copy(a)
        il=n==0 ? 1 : min(2, n)
        iu=n==0 ? 0 : n
        result=direct_selected_selfadjoint!(
            a, J; n, uplo, jobz, range, il, iu, vl=R(-1), vu=R(2), abstol
        )
        external=direct_selected_selfadjoint!(
            copy(prior),
            Int64;
            n,
            uplo,
            jobz,
            range,
            il,
            iu,
            vl=R(-1),
            vu=R(2),
            abstol,
            library=reference,
        )
        @test result.info==external.info==0
        @test result.count==external.count
        count=result.count
        @test result.values[2:(count + 1)]≈external.values[2:(count + 1)] atol=tolerance
        @test result.values[[1, end]]==R[777, 777]
        @test result.failed[[1, end]]==J[777, 777]
        @test result.iwork[[1, end]]==J[777, 777]
        @test result.work[[1, end]]==T[777, 777]
        @test result.rwork[[1, end]]==R[777, 777]
        @test all(==(T(777)), a[(n + 1):end, :])
        if abstol<=0
            @test isequal(a, prior)
        end
        if jobz=='V'
            z=result.z[1:n, 1:count]
            @test original*z≈z*Diagonal(result.values[2:(count + 1)]) atol=tolerance
            @test z'*z≈Matrix{T}(I, count, count) atol=tolerance
            @test all(iszero, result.failed[2:(count + 1)])
            @test all(==(T(777)), result.z[(n + 1):end, :])
        else
            @test all(==(T(777)), result.z)
            @test all(==(J(777)), result.failed)
        end
    end
end

@testset "Selected self-adjoint EVD query, invalid arguments and integer width" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
        a=fill(T(NaN), 7, 4)
        before=copy(a)
        R=real(T)
        q=direct_selected_selfadjoint!(a, J; lwork=-1)
        @test q.info==0
        @test q.count==-777
        @test isequal(a, before)
        @test all(==(T(777)), q.z)
        @test all(==(J(777)), q.failed)
        @test all(==(J(777)), q.iwork)
        minimum=T<:Complex ? 8 : 32
        @test real(q.work[2])>=minimum
        @test direct_selected_selfadjoint!(a, J; lwork=minimum-1).info==-17
        for (options, status) in (
            ((; jobz=('?')), -1),
            ((; range=('?')), -2),
            ((; uplo=('?')), -3),
            ((; n=-1), -4),
            ((; lda=3), -6),
            ((; range=('V'), vl=1, vu=1), -8),
            ((; range=('I'), il=0, iu=1), -9),
            ((; range=('I'), il=1, iu=5), -10),
            ((; ldz=3), -15),
            ((; jobz=('N'), ldz=0), -15),
        )
            result=direct_selected_selfadjoint!(a, J; lwork=-1, options...)
            @test result.info==status
            @test result.count==-777
            @test isequal(a, before)
            @test all(==(T(777)), result.z)
            @test all(==(J(777)), result.failed)
        end
        if J===Int64
            result=direct_selected_selfadjoint!(a, J; lda=Int64(2)^32+1, lwork=-1)
            @test result.info==0
            @test result.count==-777
        end
        for (il, iu, status) in ((1, 0, 0), (0, 0, -9), (1, 1, -10))
            result=direct_selected_selfadjoint!(zeros(T, 1, 0), J; range=('I'), il, iu)
            @test result.info==status
            @test result.count==(status==0 ? 0 : -777)
        end
    end
end

@testset "Selected self-adjoint EVD interval endpoints and repeated eigenvalues" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        uplo in ('L', 'U')

        R=real(T)
        original=Matrix(Diagonal(T[-2, 0, 2, 2, 5]))
        for (range, vl, vu, il, iu, expected) in (
            ('A', R(NaN), R(NaN), -777, -777, R[-2, 0, 2, 2, 5]),
            ('I', R(NaN), R(NaN), 2, 4, R[0, 2, 2]),
            ('V', R(0), R(2), -777, -777, R[2, 2]),
            ('V', R(2), nextfloat(R(2)), -777, -777, R[]),
            ('V', prevfloat(R(2)), R(2), -777, -777, R[2, 2]),
        )
            result=direct_selected_selfadjoint!(
                selected_selfadjoint_storage(original, uplo), J; uplo, range, vl, vu, il, iu
            )
            count=length(expected)
            @test result.info==0
            @test result.count==count
            @test result.values[2:(count + 1)]≈expected
            @test all(==(R(777)), result.values[(count + 2):end])
            @test all(==(T(777)), result.z[:, (count + 1):end])
            @test all(iszero, result.failed[2:(count + 1)])
            @test all(==(J(777)), result.failed[(count + 2):end])
        end
        for (vl, vu, expected) in ((R(0), R(2), 1), (R(2), R(3), 0))
            result=direct_selected_selfadjoint!(
                fill(T(2), 3, 1), J; uplo, range=('V'), vl, vu
            )
            @test result.info==0
            @test result.count==expected
            @test result.z[1, 1]==one(T)
        end
    end
end

@testset "Selected self-adjoint EVD LBT registration and execution" begin
    for (T, prefix, family) in (
            (Float32, "s", "syevx"),
            (Float64, "d", "syevx"),
            (ComplexF32, "c", "heevx"),
            (ComplexF64, "z", "heevx"),
        ),
        J in (Int32, Int64)

        backing=BLAS.lbt_find_backing_library(prefix*family*"_", J===Int32 ? :lp64 : :ilp64)
        @test backing!==nothing
        @test realpath(backing.libname)==realpath(Colmat.backend_library())
        original=Matrix{T}(I, 3, 3)
        result=direct_selected_selfadjoint!(
            selected_selfadjoint_storage(original, 'L'), J; library=BLAS.libblastrampoline
        )
        @test result.info==0
        @test result.count==3
        @test result.values[2:4]≈ones(real(T), 3)
    end
end

@testset "Selected self-adjoint EVD extreme scales" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        uplo in ('L', 'U')

        R=real(T)
        tolerance=R===Float32 ? 1e-5 : 2e-13
        base=T<:Complex ? T[2 im; -im 2] : T[2 1; 1 2]
        for scale in (16floatmin(R), floatmax(R)/16)
            result=direct_selected_selfadjoint!(
                selected_selfadjoint_storage(base*scale, uplo), J; uplo
            )
            @test result.info==0
            @test result.count==2
            values=result.values[2:3]/scale
            @test values≈R[1, 3] rtol=tolerance
            z=result.z[1:2, 1:2]
            @test base*z≈z*Diagonal(values) atol=tolerance
            @test z'*z≈Matrix{T}(I, 2, 2) atol=tolerance
        end
    end
end
