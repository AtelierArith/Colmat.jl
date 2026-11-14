using Test, LinearAlgebra, Random, Libdl, Colmat

function selected_svd_workspace(T, m, n)
    k=min(m, n)
    k==0&&return 1
    tall=max(m, n)>=fld(8k, 5)
    return if T<:Complex
        (tall ? k*(k+5) : 3k+max(m, n))
    else
        (tall ? k*(3k+20) : max(k*(2k+19), 4k+max(m, n)))
    end
end
for (T, prefix) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    J in (Int32, Int64)

    R=typeof(real(zero(T)))
    symbol=Symbol(prefix*"gesvdx_"*(J===Int64 ? "64_" : ""))
    @eval function direct_selected_svd!(
        a::Matrix{$T},
        ::Type{$J};
        m=size(a, 1),
        n=size(a, 2),
        jobu=('V'),
        jobvt=('V'),
        range=('A'),
        vl=($R(NaN)),
        vu=($R(NaN)),
        il=-777,
        iu=-777,
        lda=max(1, size(a, 1)),
        ldu=max(1, m+2),
        ldvt=max(1, min(m, n)+2),
        lwork=nothing,
        library=Colmat.backend_library(),
    )
        k=max(0, min(m, n))
        minimum=selected_svd_workspace($T, max(0, m), max(0, n))
        lw=isnothing(lwork) ? minimum : lwork
        work=fill($T(777), max(1, lw)+2)
        rwork=fill($R(777), max(1, 17k*k)+2)
        iwork=fill($J(777), max(1, 12k)+2)
        values=fill($R(777), k+2)
        u=fill($T(777), max(1, ldu), max(1, k))
        vt=fill($T(777), max(1, ldvt), max(1, n))
        count=Ref{$J}(-777)
        info=Ref{$J}(-777)
        wp=view(work, 2:(length(work) - 1))
        rp=view(rwork, 2:(length(rwork) - 1))
        ip=view(iwork, 2:(length(iwork) - 1))
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
                    Ref{$J},
                    Ptr{$T},
                    Ref{$J},
                    Ref{$R},
                    Ref{$R},
                    Ref{$J},
                    Ref{$J},
                    Ref{$J},
                    Ptr{$R},
                    Ptr{$T},
                    Ref{$J},
                    Ptr{$T},
                    Ref{$J},
                    Ptr{$T},
                    Ref{$J},
                    Ptr{$J},
                    Ref{$J},
                ),
                jobu,
                jobvt,
                range,
                m,
                n,
                a,
                lda,
                vl,
                vu,
                il,
                iu,
                count,
                sp,
                u,
                ldu,
                vt,
                ldvt,
                wp,
                lw,
                ip,
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
                    Ref{$J},
                    Ptr{$T},
                    Ref{$J},
                    Ref{$R},
                    Ref{$R},
                    Ref{$J},
                    Ref{$J},
                    Ref{$J},
                    Ptr{$R},
                    Ptr{$T},
                    Ref{$J},
                    Ptr{$T},
                    Ref{$J},
                    Ptr{$T},
                    Ref{$J},
                    Ptr{$R},
                    Ptr{$J},
                    Ref{$J},
                ),
                jobu,
                jobvt,
                range,
                m,
                n,
                a,
                lda,
                vl,
                vu,
                il,
                iu,
                count,
                sp,
                u,
                ldu,
                vt,
                ldvt,
                wp,
                lw,
                rp,
                ip,
                info,
            )
        end
        (; info=info[], ns=count[], values, u, vt, work, rwork, iwork)
    end
end
function selected_storage(original)
    a=fill(eltype(original)(777), size(original, 1)+3, size(original, 2))
    a[1:size(original, 1), :]=original
    return a
end
@testset "Selected SVD four types, both ABIs and rectangular residuals" begin
    Random.seed!(273)
    reference=only(
        filter(
            l->l.interface==:ilp64&&occursin("openblas", lowercase(basename(l.libname))),
            BLAS.get_config().loaded_libs,
        ),
    ).libname
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        (m, n) in
        ((0, 0), (0, 4), (4, 0), (1, 1), (2, 5), (5, 2), (6, 6), (20, 4), (4, 20)),
        jobu in ('N', 'V'),
        jobvt in ('N', 'V')

        R=typeof(real(zero(T)))
        tolerance=R===Float32 ? 2e-4 : 2e-12
        original=randn(T, m, n)
        a=selected_storage(original)
        prior=copy(a)
        result=direct_selected_svd!(a, J; m, jobu, jobvt)
        external=direct_selected_svd!(copy(a), Int64; m, jobu, jobvt, library=reference)
        k=min(m, n)
        @test result.info==external.info==0
        @test result.ns==k
        @test result.values[2:(k + 1)] ≈ external.values[2:(k + 1)] atol=tolerance
        @test result.values[[1, end]]==R[777, 777]
        @test result.work[[1, end]]==T[777, 777]
        @test result.rwork[[1, end]]==R[777, 777]
        @test result.iwork[[1, end]]==J[777, 777]
        @test all(iszero, result.iwork[2:(k + 1)])
        @test a==prior
        u=result.u[1:m, 1:k]
        vt=result.vt[1:k, 1:n]
        values=result.values[2:(k + 1)]
        if jobu=='V'
            @test u'*u ≈ Matrix{T}(I, k, k) atol=tolerance
            @test all(==(T(777)), result.u[(m + 1):end, :])
        else
            @test all(==(T(777)), result.u)
        end
        if jobvt=='V'
            @test vt*vt' ≈ Matrix{T}(I, k, k) atol=tolerance
            @test all(==(T(777)), result.vt[(k + 1):end, :])
        else
            @test all(==(T(777)), result.vt)
        end
        if jobu=='V'&&jobvt=='V'
            @test original*vt' ≈ u*Diagonal(values) atol=tolerance
            @test original'*u ≈ vt'*Diagonal(values) atol=tolerance
        end
    end
end
@testset "Selected SVD intervals, repeated and zero singular values" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64), (m, n) in ((5, 7), (7, 5)),
        range in ('A', 'I', 'V')

        R=typeof(real(zero(T)))
        tolerance=R===Float32 ? 2e-4 : 2e-12
        original=zeros(T, m, n)
        sigma=R[5, 3, 3, 1, 0]
        for i in 1:5
            original[i, i]=T<:Complex&&isodd(i) ? T(0, sigma[i]) : T(sigma[i])
        end
        a=selected_storage(original)
        result=direct_selected_svd!(
            a, J; m, range, vl=R(2), vu=R(4), il=2, iu=4, ldvt=range=='I' ? 3 : 7
        )
        selected=if range=='A'
            collect(1:5)
        elseif range=='I'
            collect(2:4)
        else
            [2, 3]
        end
        count=length(selected)
        @test result.info==0
        @test result.ns==count
        @test result.values[2:(count + 1)] ≈ sigma[selected] atol=tolerance
        @test all(==(R(777)), result.values[(count + 2):end])
        u=result.u[1:m, 1:count]
        vt=result.vt[1:count, 1:n]
        values=result.values[2:(count + 1)]
        @test original*vt' ≈ u*Diagonal(values) atol=tolerance
        @test original'*u ≈ vt'*Diagonal(values) atol=tolerance
        @test u'*u ≈ Matrix{T}(I, count, count) atol=tolerance
        @test vt*vt' ≈ Matrix{T}(I, count, count) atol=tolerance
        @test all(==(T(777)), result.u[:, (count + 1):end])
        @test all(==(T(777)), result.vt[(count + 1):end, :])
    end
end
@testset "Selected SVD query and argument contracts" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
        a=fill(T(NaN), 7, 3)
        prior=copy(a)
        result=direct_selected_svd!(a, J; m=4, lwork=-1)
        @test result.info==0
        @test result.ns==0
        @test isequal(a, prior)
        @test all(==(T(777)), result.u)
        @test all(==(T(777)), result.vt)
        @test all(==(J(777)), result.iwork)
        @test real(result.work[2])>=selected_svd_workspace(T, 4, 3)
        @test direct_selected_svd!(a, J; m=4, lwork=selected_svd_workspace(T, 4, 3)-1).info==-19
        for (options, status) in (
            ((; jobu=('?')), -1),
            ((; jobvt=('?')), -2),
            ((; range=('?')), -3),
            ((; m=-1), -4),
            ((; n=-1), -5),
            ((; lda=3), -7),
            ((; range=('V'), vl=-1, vu=1), -8),
            ((; range=('V'), vl=1, vu=1), -9),
            ((; range=('I'), il=0, iu=1), -10),
            ((; range=('I'), il=1, iu=4), -11),
            ((; ldu=3), -15),
            ((; ldvt=2), -17),
        )
            @test direct_selected_svd!(a, J; m=4, options...).info==status
        end
    end
end

@testset "Selected SVD exact scalar interval endpoints" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
        R = typeof(real(zero(T)))
        for (lo, hi, expected) in (
            (R(1), R(3), 1),
            (R(3), R(4), 0),
            (prevfloat(R(3)), R(3), 1),
            (R(3), nextfloat(R(3)), 0),
        )
            result = direct_selected_svd!(fill(T(3), 1, 1), J; range=('V'), vl=lo, vu=hi)
            @test result.info == 0
            @test result.ns == expected
            if expected == 1
                @test result.values[2] == R(3)
                @test abs(result.u[1, 1]) == one(R)
                @test abs(result.vt[1, 1]) == one(R)
            else
                @test all(==(R(777)), result.values)
                @test all(==(T(777)), result.u)
                @test all(==(T(777)), result.vt)
            end
        end
    end
end

@testset "Selected SVD unused outputs and integer width" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
        a = selected_storage(Matrix{T}(I, 4, 3))
        result = direct_selected_svd!(a, J; m=4, jobu=('N'), jobvt=('N'), ldu=0, ldvt=0)
        @test result.info == 0
        @test result.ns == 3
        @test all(==(T(777)), result.u)
        @test all(==(T(777)), result.vt)
        @test result.iwork[1] == J(777)
        @test result.iwork[end] == J(777)
        @test result.work[1] == T(777)
        @test result.work[end] == T(777)
        if J === Int64
            # A 32-bit reader would see M=4 and return a successful query.
            wide = direct_selected_svd!(a, J; m=Int64(2)^32+4, ldu=1, lwork=-1)
            @test wide.info == -7
            @test wide.ns == 0
        end
    end
end

for (T, prefix) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    J in (Int32, Int64)

    R=typeof(real(zero(T)))
    symbol=Symbol(prefix*"gesvdq_"*(J===Int64 ? "64_" : ""))
    @eval function direct_preconditioned_svd!(
        a::Matrix{$T},
        ::Type{$J};
        m=size(a, 1),
        n=size(a, 2),
        joba=('H'),
        jobp=('N'),
        jobr=('N'),
        jobu=('S'),
        jobv=('A'),
        lda=size(a, 1),
        ldu=max(1, m+2),
        ldv=max(1, n+2),
        liwork=nothing,
        lwork=nothing,
        lrwork=nothing,
        library=Colmat.backend_library(),
    )
        u=fill($T(777), max(1, ldu), max(1, m))
        v=fill($T(777), max(1, ldv), max(1, n))
        values=fill($R(777), max(0, n))
        iw=fill($J(777), max(2, 3n+m+2))
        w=fill($T(777), max(2, 10n+2m+2))
        rw=fill($R(777), max(2, 5n+m+2))
        rank=Ref{$J}(-777)
        info=Ref{$J}(-777)
        f=Libdl.dlsym(Libdl.dlopen(library), $(QuoteNode(symbol)))
        ccall(
            f,
            Cvoid,
            (
                Ref{UInt8},
                Ref{UInt8},
                Ref{UInt8},
                Ref{UInt8},
                Ref{UInt8},
                Ref{$J},
                Ref{$J},
                Ptr{$T},
                Ref{$J},
                Ptr{$R},
                Ptr{$T},
                Ref{$J},
                Ptr{$T},
                Ref{$J},
                Ref{$J},
                Ptr{$J},
                Ref{$J},
                Ptr{$T},
                Ref{$J},
                Ptr{$R},
                Ref{$J},
                Ref{$J},
            ),
            joba,
            jobp,
            jobr,
            jobu,
            jobv,
            m,
            n,
            a,
            lda,
            values,
            u,
            ldu,
            v,
            ldv,
            rank,
            iw,
            isnothing(liwork) ? length(iw) : liwork,
            w,
            isnothing(lwork) ? length(w) : lwork,
            rw,
            isnothing(lrwork) ? length(rw) : lrwork,
            info,
        )
        (; u, v, values, rank=rank[], info=info[], iw, w, rw)
    end
end
@testset "QR preconditioned SVD direct ABI reconstruction" begin
    rng=MersenneTwister(273)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        (m, n) in ((3, 3), (7, 3)),
        joba in ('A', 'M', 'H'),
        jobp in ('N', 'P'),
        jobr in ('N', 'T'),
        jobu in ('A', 'S', 'R'),
        jobv in ('N', 'R', 'A')

        jobu=='R' && jobv=='A' && continue
        original=randn(rng, T, m, n)
        a=selected_storage(original)
        result=direct_preconditioned_svd!(a, J; m, joba, jobp, jobr, jobu, jobv)
        @test result.info==0
        @test result.rank==n
        tolerance=real(T)===Float32 ? 4e-4 : 3e-12
        @test result.values ≈ svdvals(original) rtol=tolerance
        u=result.u[1:m, 1:n]
        @test u'*u ≈ Matrix{T}(I, n, n) atol=tolerance
        @test all(==(T(777)), result.u[(m + 1):end, :])
        if jobv!='N'
            vt=result.v[1:n, 1:n]
            @test u*Diagonal(result.values)*vt ≈ original atol=tolerance
            @test vt*vt' ≈ Matrix{T}(I, n, n) atol=tolerance
        end
        @test all(==(T(777)), a[(m + 1):end, :])
        @test sort(result.iw[1:n])==J.(1:n)
    end
end

@testset "QR preconditioned SVD query and minimum workspace" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64), jobu in ('N', 'A', 'S', 'F'), jobv in ('N', 'A'),
        jobp in ('N', 'P')

        m, n=7, 3
        a=fill(T(NaN), m+2, n)
        before=copy(a)
        q=direct_preconditioned_svd!(a, J; m, jobu, jobv, jobp, lwork=-1)
        @test q.info==0
        @test q.rank==-777
        @test isequal(a, before)
        @test all(==(T(777)), q.u)
        @test all(==(T(777)), q.v)
        @test real(q.w[1])==real(q.w[2])
        for (arg, status) in ((:liwork, -17), (:lwork, -19), (:lrwork, -21))
            lengths=(; liwork=Int(q.iw[1]), lwork=Int(real(q.w[2])), lrwork=Int(q.rw[1]))
            options=merge(lengths, NamedTuple{(arg,)}((getproperty(lengths, arg)-1,)))
            bad=direct_preconditioned_svd!(a, J; m, jobu, jobv, jobp, options...)
            @test bad.info==status
            @test isequal(a, before)
        end
        original=Matrix{T}(I, m, n)
        result=direct_preconditioned_svd!(
            selected_storage(original),
            J;
            m,
            jobu,
            jobv,
            jobp,
            liwork=Int(q.iw[1]),
            lwork=Int(real(q.w[2])),
            lrwork=Int(q.rw[1]),
        )
        @test result.info==0
        @test result.rank==n
        @test result.values ≈ ones(real(T), n)
    end
end

@testset "QR preconditioned SVD rank, factored vectors and expert fallback" begin
    reference=only(
        filter(
            l->l.interface==:ilp64 && occursin("openblas", lowercase(basename(l.libname))),
            BLAS.get_config().loaded_libs,
        ),
    ).libname
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        joba in ('A', 'M', 'H'),
        jobp in ('N', 'P'),
        jobr in ('N', 'T'),
        spectrum in ((5, 2, 0), (0, 0, 0))

        R=real(T)
        m=6
        n=3
        original=zeros(T, m, n)
        for i in 1:n
            original[i + 2, i]=T<:Complex ? T(0, spectrum[i]) : T(spectrum[i])
        end
        a=selected_storage(original)
        result=direct_preconditioned_svd!(a, J; m, joba, jobp, jobr, jobu=('F'), jobv=('A'))
        @test result.info==0
        @test result.rank==count(!iszero, spectrum)
        @test result.values ≈ R[spectrum...] atol=20eps(R)
        pre_q=zeros(T, m, n)
        pre_q[1:n, :]=result.u[1:n, 1:n]
        LinearAlgebra.LAPACK.ormqr!('L', 'N', view(a, 1:m, :), result.w[1:n], pre_q)
        if jobp=='P'
            for i in (m - 1):-1:1
                p=result.iw[n + i]
                pre_q[[i, p], :]=pre_q[[p, i], :]
            end
        end
        vt=result.v[1:n, 1:n]
        @test pre_q*Diagonal(result.values)*vt ≈ original atol=200eps(R)
        @test pre_q'*pre_q ≈ Matrix{T}(I, n, n) atol=200eps(R)
        @test vt*vt' ≈ Matrix{T}(I, n, n) atol=200eps(R)
        @test all(==(T(777)), a[(m + 1):end, :])
    end
    for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
        original=reshape(T.(1:18), 6, 3)+Matrix{T}(I, 6, 3)
        a=selected_storage(original)
        native=direct_preconditioned_svd!(a, J; m=6, joba=('E'))
        external=direct_preconditioned_svd!(
            selected_storage(original), Int64; m=6, joba=('E'), library=reference
        )
        @test native.info==external.info==0
        @test native.rank==external.rank
        @test native.values==external.values
        @test native.rw[1]==external.rw[1]
        @test native.u[1:6, 1:3]*Diagonal(native.values)*native.v[1:3, 1:3] ≈ original rtol=200eps(
            real(T)
        )
    end
end

@testset "Advanced SVD LBT registration and numerical calls" begin
    for (T, prefix) in
        ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
        J in (Int32, Int64)

        abi=J===Int32 ? :lp64 : :ilp64
        for family in ("gesvdx", "gesvdq")
            backing=BLAS.lbt_find_backing_library(prefix*family*"_", abi)
            @test backing!==nothing
            @test realpath(backing.libname)==realpath(Colmat.backend_library())
        end
        original=Matrix{T}(I, 4, 3)
        x=direct_selected_svd!(copy(original), J; library=BLAS.libblastrampoline)
        q=direct_preconditioned_svd!(copy(original), J; library=BLAS.libblastrampoline)
        @test x.info==q.info==0
        @test x.ns==q.rank==3
        @test x.values[2:4]≈q.values
        for family in ("gejsv", "gesvj")
            backing=BLAS.lbt_find_backing_library(prefix*family*"_", abi)
            @test backing===nothing ||
                realpath(backing.libname)!=realpath(Colmat.backend_library())
        end
    end
end

@testset "QR preconditioned SVD invalid arguments and empty matrices" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
        a=fill(T(NaN), 7, 3)
        before=copy(a)
        for (options, status) in (
            ((; joba=('?')), -1),
            ((; jobp=('?')), -2),
            ((; jobr=('?')), -3),
            ((; jobu=('?')), -4),
            ((; jobv=('?')), -5),
            ((; jobu=('R'), jobv=('A')), -5),
            ((; m=-1), -6),
            ((; n=-1), -7),
            ((; n=5), -7),
            ((; lda=3), -9),
            ((; ldu=3), -12),
            ((; jobu=('F'), ldu=2), -12),
            ((; ldv=2), -14),
        )
            result=direct_preconditioned_svd!(a, J; m=4, lwork=-1, options...)
            @test result.info==status
            @test result.rank==-777
            @test isequal(a, before)
            @test all(==(T(777)), result.u)
            @test all(==(T(777)), result.v)
        end
        if J===Int64
            result=direct_preconditioned_svd!(a, J; m=4, lda=Int64(2)^32+1, lwork=-1)
            @test result.info==0
            @test result.rank==-777
        end
        for (m, n) in ((0, 0), (4, 0))
            a=fill(T(777), max(1, m+2), n)
            result=direct_preconditioned_svd!(a, J; m, n)
            @test result.info==0
            @test result.rank==-777
            @test all(==(T(777)), result.u)
            @test all(==(T(777)), result.v)
            @test all(==(J(777)), result.iw)
        end
    end
end

@testset "QR preconditioned SVD accuracy-dependent rank thresholds" begin
    reference=only(
        filter(
            l->l.interface==:ilp64 && occursin("openblas", lowercase(basename(l.libname))),
            BLAS.get_config().loaded_libs,
        ),
    ).libname
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64), joba in ('A', 'M', 'H'),
        jobr in ('N', 'T')

        R=real(T)
        e=eps(R)
        original=zeros(T, 5, 3)
        original[1, 1]=one(T)
        original[2, 2]=T(3e/4)
        original[3, 3]=T(e*e/8)
        result=direct_preconditioned_svd!(
            selected_storage(original), J; m=5, joba, jobr, jobu=('R'), jobv=('R')
        )
        external=direct_preconditioned_svd!(
            selected_storage(original),
            Int64;
            m=5,
            joba,
            jobr,
            jobu=('R'),
            jobv=('R'),
            library=reference,
        )
        expected=if joba=='A'
            1
        elseif joba=='M'
            2
        else
            3
        end
        @test result.info==external.info==0
        @test result.rank==external.rank==expected
        @test result.values[(expected + 1):end]==zeros(R, 3-expected)
        @test result.values[1:expected] ≈ external.values[1:expected] rtol=100eps(R)
        @test all(==(T(777)), result.u[:, (expected + 1):end])
        @test all(==(T(777)), result.v[(expected + 1):end, :])
    end
end
