using Test, LinearAlgebra, Random, Libdl, Colmat

function structured_evd_minima(T, n, kind, jobz)
    n <= 1 && return (1, 1, 1)
    band = startswith(string(kind), "b")
    if endswith(string(kind), "d")
        if jobz == 'N'
            return (T <: Complex ? n : 2n, n, 1)
        elseif T <: Complex
            return (band ? 2n^2 : 2n, 1 + 5n + 2n^2, 3 + 5n)
        else
            return (band ? 1 + 5n + 2n^2 : 1 + 6n + n^2, 1, 3 + 5n)
        end
    end
    return (T <: Complex ? (band ? n : 2n) : (band ? 7n : 8n), 7n, 5n)
end

# Generate concrete external signatures, so LP64 and ILP64 execute with their
# actual integer widths rather than a runtime-selected integer alias.
for (T, prefix) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    J in (Int32, Int64),
    kind in (:bev, :bevd, :bevx, :pev, :pevd, :pevx)

    R = real(T)
    band = startswith(string(kind), "b")
    selected = endswith(string(kind), "x")
    divided = endswith(string(kind), "d")
    family = (T <: Complex ? "h" : "s") * string(kind)
    symbol = Symbol(prefix * family * "_" * (J === Int64 ? "64_" : ""))
    signature = Any[Ref{UInt8}]
    arguments = Any[:jobz]
    if selected
        push!(signature, Ref{UInt8})
        push!(arguments, :range)
    end
    append!(signature, [Ref{UInt8}, Ref{J}])
    append!(arguments, [:uplo, :n])
    if band
        push!(signature, Ref{J})
        push!(arguments, :kd)
    end
    push!(signature, Ptr{T})
    push!(arguments, :ap)
    if band
        push!(signature, Ref{J})
        push!(arguments, :ldab)
        if selected
            append!(signature, [Ptr{T}, Ref{J}])
            append!(arguments, [:qp, :ldq])
        end
    end
    if selected
        append!(signature, [Ref{R}, Ref{R}, Ref{J}, Ref{J}, Ref{R}, Ptr{J}])
        append!(arguments, [:(($R)(vl)), :(($R)(vu)), :il, :iu, :(($R)(abstol)), :mp])
    end
    append!(signature, [Ptr{R}, Ptr{T}, Ref{J}, Ptr{T}])
    append!(arguments, [:wp, :zp, :ldz, :workp])
    if divided
        push!(signature, Ref{J})
        push!(arguments, :lw)
    end
    if T <: Complex
        push!(signature, Ptr{R})
        push!(arguments, :rworkp)
        if divided
            push!(signature, Ref{J})
            push!(arguments, :lr)
        end
    end
    if divided || selected
        push!(signature, Ptr{J})
        push!(arguments, :iworkp)
        if divided
            push!(signature, Ref{J})
            push!(arguments, :li)
        else
            push!(signature, Ptr{J})
            push!(arguments, :fp)
        end
    end
    push!(signature, Ptr{J})
    push!(arguments, :infop)
    @eval function call_structured_evd(
        a::Vector{$T},
        ::Type{$J},
        ::Val{$(QuoteNode(kind))};
        n,
        kd=1,
        uplo=('L'),
        jobz=('V'),
        range=('A'),
        vl=NaN,
        vu=NaN,
        il=-777,
        iu=-777,
        abstol=zero($R),
        ldab=max(1, kd+3),
        ldq=max(1, n+2),
        ldz=max(1, n+2),
        lwork=nothing,
        lrwork=nothing,
        liwork=nothing,
        library=Colmat.backend_library(),
    )
        order = max(0, n)
        minimum = structured_evd_minima($T, order, $(QuoteNode(kind)), jobz)
        lw = isnothing(lwork) ? minimum[1] : lwork
        lr = isnothing(lrwork) ? minimum[2] : lrwork
        li = isnothing(liwork) ? minimum[3] : liwork
        input = vcat($T(777), a, $T(777))
        qrows = max(1, ldq)
        zrows = max(1, ldz)
        columns = max(1, order)
        qbuffer = fill($T(777), qrows*columns+2)
        zbuffer = fill($T(777), zrows*columns+2)
        values = fill($R(777), max(1, order)+2)
        work = fill($T(777), max(1, lw)+2)
        rwork = fill($R(777), max(1, lr)+2)
        iwork = fill($J(777), max(1, li)+2)
        failed = fill($J(777), max(1, order)+2)
        count = fill($J(-777), 3)
        info = fill($J(-777), 3)
        ap=view(input, 2:(length(input) - 1))
        qp=view(qbuffer, 2:(length(qbuffer) - 1))
        zp=view(zbuffer, 2:(length(zbuffer) - 1))
        wp=view(values, 2:(length(values) - 1))
        workp=view(work, 2:(length(work) - 1))
        rworkp=view(rwork, 2:(length(rwork) - 1))
        iworkp=view(iwork, 2:(length(iwork) - 1))
        fp=view(failed, 2:(length(failed) - 1))
        mp=view(count, 2:2)
        infop=view(info, 2:2)
        f = Libdl.dlsym(Libdl.dlopen(library), $(QuoteNode(symbol)))
        ccall(f, Cvoid, $(Expr(:tuple, signature...)), $(arguments...))
        diagnostic=ccall(
            Libdl.dlsym(Libdl.dlopen(Colmat.backend_library()), :colmat_blas_last_error),
            Cint,
            (),
        )
        return (;
            input,
            a=ap,
            q=reshape(qp, qrows, columns),
            z=reshape(zp, zrows, columns),
            qbuffer,
            zbuffer,
            values,
            work,
            rwork,
            iwork,
            failed,
            diagnostic,
            count=count[2],
            info=info[2],
            count_guard=count[[1, 3]],
            info_guard=info[[1, 3]],
        )
    end
end

function structured_evd_input(a, kind, uplo, kd, ldab)
    n = size(a, 1)
    T=eltype(a)
    if startswith(string(kind), "b")
        storage=fill(T(NaN), ldab, n)
        storage[(kd + 2):end, :].=T(777)
        for j in 1:n, i in max(1, j - kd):min(n, j + kd)
            if uplo=='L' && i>=j
                storage[1 + i - j, j]=a[i, j]
            elseif uplo=='U' && i<=j
                storage[kd + 1 + i - j, j]=a[i, j]
            end
        end
        return vec(storage)
    end
    return if uplo=='L'
        [a[i, j] for j in 1:n for i in j:n]
    else
        [a[i, j] for j in 1:n for i in 1:j]
    end
end

function structured_evd_guards(r, T, J)
    R=real(T)
    @test r.input[[1, end]]==T[777, 777]
    @test r.qbuffer[[1, end]]==T[777, 777]
    @test r.zbuffer[[1, end]]==T[777, 777]
    @test r.values[[1, end]]==R[777, 777]
    @test r.work[[1, end]]==T[777, 777]
    @test r.rwork[[1, end]]==R[777, 777]
    @test r.iwork[[1, end]]==J[777, 777]
    @test r.failed[[1, end]]==J[777, 777]
    @test r.count_guard==J[-777, -777]
    @test r.info_guard==J[-777, -777]
    @test r.diagnostic==max(0, -r.info)
end

@testset "Structured EVD four types and both direct integer ABIs" begin
    reference=only(
        filter(
            l->l.interface==:ilp64&&occursin("openblas", lowercase(basename(l.libname))),
            BLAS.get_config().loaded_libs,
        ),
    ).libname
    rng=MersenneTwister(286)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        kind in (:bev, :bevd, :bevx, :pev, :pevd, :pevx),
        n in (0, 1, 4, 7),
        uplo in ('L', 'U'),
        jobz in ('N', 'V'),
        kd in (0, 1, 9)

        R=real(T)
        tolerance=R===Float32 ? 3e-4 : 3e-11
        band=startswith(string(kind), "b")
        selected=endswith(string(kind), "x")
        !band&&kd!=1&&continue
        raw=randn(rng, T, n, n)
        original=Matrix(Hermitian(raw+raw'))
        if band
            for j in 1:n, i in 1:n
                abs(i-j)>kd&&(original[i, j]=zero(T))
            end
        end
        ldab=kd+3
        input=structured_evd_input(original, kind, uplo, kd, ldab)
        for range in (selected ? ('A', 'I', 'V') : ('A',)),
            abstol in (selected ? (zero(R), floatmin(R)) : (zero(R),))

            il=n==0 ? 1 : min(2, n)
            iu=n==0 ? 0 : n
            result=call_structured_evd(
                input,
                J,
                Val(kind);
                n,
                kd,
                ldab,
                uplo,
                jobz,
                range,
                il,
                iu,
                vl=-1,
                vu=2,
                abstol,
            )
            external=call_structured_evd(
                input,
                Int64,
                Val(kind);
                n,
                kd,
                ldab,
                uplo,
                jobz,
                range,
                il,
                iu,
                vl=-1,
                vu=2,
                abstol,
                library=reference,
            )
            @test result.info==external.info==0
            count=selected ? result.count : n
            @test count==(selected ? external.count : n)
            if n==1&&band&&kind==:bevd&&uplo=='U'&&kd>0
                # Reference BEVD's singleton quick return reads AB(1,1)
                # rather than the documented upper diagonal AB(KD+1,1).
                # Keep NaN in unused input and use the exact scalar oracle.
                # See reproducers/openblas-bevd-singleton-upper.jl.
                @test result.values[2]≈real(original[1, 1]) atol=tolerance rtol=tolerance
            else
                @test result.values[2:(count + 1)]≈external.values[2:(count + 1)] atol=tolerance rtol=tolerance
            end
            structured_evd_guards(result, T, J)
            @test all(==(R(777)), result.values[(count + 2):end])
            if jobz=='V'
                z=result.z[1:n, 1:count]
                @test original*z≈z*Diagonal(result.values[2:(count + 1)]) atol=tolerance rtol=tolerance
                @test z'*z≈Matrix{T}(I, count, count) atol=tolerance rtol=tolerance
                @test all(==(T(777)), result.z[(n + 1):end, :])
                if selected&&n>1
                    @test all(iszero, result.failed[2:(count + 1)])
                    @test all(==(J(777)), result.failed[(count + 2):end])
                    if band
                        q=result.q[1:n, 1:n]
                        @test q'*q≈Matrix{T}(I, n, n) atol=tolerance rtol=tolerance
                        stored=reshape(result.a, ldab, n)
                        diagonal=real.(stored[uplo=='L' ? 1 : kd+1, :])
                        offdiagonal=if kd==0
                            zeros(R, n-1)
                        else
                            real.(uplo=='L' ? stored[2, 1:(n - 1)] : stored[kd, 2:n])
                        end
                        reduced=Matrix(SymTridiagonal(diagonal, offdiagonal))
                        @test q*reduced*q'≈original atol=tolerance rtol=tolerance
                        @test all(==(T(777)), result.q[(n + 1):end, :])
                    end
                end
            else
                @test all(==(T(777)), result.z)
                @test all(==(J(777)), result.failed)
                @test all(==(T(777)), result.q)
            end
            if band
                stored=reshape(result.a, ldab, n)
                @test all(==(T(777)), stored[(kd + 2):end, :])
            end
        end
    end
end

@testset "Structured EVD workspace query and invalid arguments" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        kind in (:bev, :bevd, :bevx, :pev, :pevd, :pevx)

        R=real(T)
        band=startswith(string(kind), "b")
        selected=endswith(string(kind), "x")
        divided=endswith(string(kind), "d")
        # Both unused triangles and all mathematical entries can remain NaN
        # during a query or an argument error.
        n=4
        kd=1
        ldab=4
        a=fill(T(NaN), band ? ldab*n : n*(n+1)÷2)
        if divided
            for jobz in ('N', 'V'), which in (T<:Complex ? (1, 2, 3) : (1, 2))
                lw=which==1 ? -1 : 0
                li=which==2 ? -1 : 0
                lr=which==3 ? -1 : 0
                r=call_structured_evd(
                    a, J, Val(kind); n, kd, ldab, jobz, lwork=lw, liwork=li, lrwork=lr
                )
                minimum=structured_evd_minima(T, n, kind, jobz)
                @test r.info==0
                @test real(r.work[2])>=minimum[1]
                @test r.iwork[2]==minimum[3]
                T<:Complex&&(@test r.rwork[2]>=minimum[2])
                @test all(isnan, r.a)
                @test all(==(R(777)), r.values)
                @test all(==(T(777)), r.z)
                @test all(==(T(777)), r.q)
                @test all(==(J(777)), r.failed)
                @test r.count==-777
                structured_evd_guards(r, T, J)
            end
            minimum=structured_evd_minima(T, n, kind, 'V')
            for (options, arg) in (
                ((; lwork=minimum[1]-1), band ? 11 : 9),
                (
                    (; liwork=minimum[3]-1),
                    band ? (T<:Complex ? 15 : 13) : (T<:Complex ? 13 : 11),
                ),
            )
                r=call_structured_evd(a, J, Val(kind); n, kd, ldab, options...)
                @test r.info == -arg
                @test all(isnan, r.a)
                @test all(==(R(777)), r.values)
                @test all(==(T(777)), r.z)
                structured_evd_guards(r, T, J)
            end
            if T<:Complex
                @test call_structured_evd(
                    a, J, Val(kind); n, kd, ldab, lrwork=minimum[2]-1
                ).info==-(band ? 13 : 11)
            end
        end
        cases=if selected
            [
                ((; jobz=('?')), -1),
                ((; range=('?')), -2),
                ((; uplo=('?')), -3),
                ((; n=-1), -4),
                ((; range=('V'), vl=2, vu=1), -(band ? 11 : 7)),
                ((; range=('I'), il=0, iu=1), -(band ? 12 : 8)),
                ((; range=('I'), il=1, iu=5), -(band ? 13 : 9)),
                ((; ldz=3), -(band ? 18 : 14)),
                ((; jobz=('N'), ldz=0), -(band ? 18 : 14)),
            ]
        else
            [
                ((; jobz=('?')), -1),
                ((; uplo=('?')), -2),
                ((; n=-1), -3),
                ((; ldz=3), -(band ? 9 : 7)),
                ((; jobz=('N'), ldz=0), -(band ? 9 : 7)),
            ]
        end
        if band
            push!(cases, ((; kd=-1), selected ? -5 : -4))
            push!(cases, ((; ldab=1), selected ? -7 : -6))
            selected&&push!(cases, ((; ldq=3), -9))
        end
        for (options, status) in cases
            opts=(; n, kd, ldab, options...)
            r=call_structured_evd(a, J, Val(kind); opts...)
            @test r.info==status
            @test r.count==-777
            @test all(isnan, r.a)
            @test all(==(R(777)), r.values)
            @test all(==(T(777)), r.z)
            @test all(==(T(777)), r.q)
            @test all(==(J(777)), r.failed)
            structured_evd_guards(r, T, J)
        end
    end
end

@testset "Structured EVD repeated eigenvalues, exact endpoints and singleton outputs" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64), kind in (:bevx, :pevx), uplo in ('L', 'U'),
        jobz in ('N', 'V')

        R=real(T)
        band=kind==:bevx
        n=4
        kd=0
        ldab=3
        original=Matrix(Diagonal(T[-2, 1, 1, 4]))
        stored_matrix=copy(original)
        if T<:Complex
            for i in 1:n
                stored_matrix[i, i]=complex(real(original[i, i]), R(NaN))
            end
        end
        input=structured_evd_input(stored_matrix, kind, uplo, kd, ldab)
        for (vl, vu, expected) in (
                (-2, 1, R[1, 1]),
                (1, 4, R[4]),
                (4, 5, R[]),
                (-3, -2, R[-2]),
                (-3, 5, R[-2, 1, 1, 4]),
            ),
            abstol in (zero(R), floatmin(R))

            result=call_structured_evd(
                input, J, Val(kind); n, kd, ldab, uplo, jobz, range=('V'), vl, vu, abstol
            )
            @test result.info==0
            @test result.count==length(expected)
            @test result.values[2:(length(expected) + 1)]≈expected
            structured_evd_guards(result, T, J)
            if jobz=='V'
                z=result.z[1:n, 1:length(expected)]
                @test original*z≈z*Diagonal(expected)
                @test z'*z≈Matrix{T}(I, length(expected), length(expected))
            end
        end
        n=1
        kd=2
        ldab=5
        single=reshape(T[2], 1, 1)
        input=structured_evd_input(single, kind, uplo, kd, ldab)
        for included in (false, true)
            result=call_structured_evd(
                input,
                J,
                Val(kind);
                n,
                kd,
                ldab,
                uplo,
                jobz,
                range=('V'),
                vl=included ? 1 : 2,
                vu=included ? 2 : 3,
            )
            @test result.info==0
            @test result.count==Int(included)
            @test result.values[2]==R(included ? 2 : 777)
            @test isequal(result.a, input)
            @test all(==(T(777)), result.q)
            @test all(==(J(777)), result.failed)
            if jobz=='V'
                @test result.z[1, 1]==T(included||!band ? 1 : 777)
            else
                @test all(==(T(777)), result.z)
            end
            structured_evd_guards(result, T, J)
        end
    end
end

@testset "Structured EVD all 24 numerical LBT routes and reload" begin
    Colmat.load_backend()
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        kind in (:bev, :bevd, :bevx, :pev, :pevd, :pevx)

        prefix=if T===Float32
            "s"
        elseif T===Float64
            "d"
        elseif T===ComplexF32
            "c"
        else
            "z"
        end
        symbol=prefix*(T<:Complex ? "h" : "s")*string(kind)*"_"
        provider=BLAS.lbt_find_backing_library(symbol, J===Int32 ? :lp64 : :ilp64)
        @test provider!==nothing
        @test realpath(provider.libname)==realpath(Colmat.backend_library())
        original=Matrix(Diagonal(T[1, 2, 3]))
        n=3
        kd=0
        ldab=3
        input=structured_evd_input(original, kind, 'L', kd, ldab)
        result=call_structured_evd(
            input, J, Val(kind); n, kd, ldab, library=BLAS.libblastrampoline
        )
        @test result.info==0
        @test result.values[2:4]≈real(T)[1, 2, 3]
        @test original*result.z[1:3, 1:3]≈result.z[1:3, 1:3]*Diagonal(result.values[2:4])
        structured_evd_guards(result, T, J)
    end
    group=Cint(Colmat._fallback_group("band/packed selected EVD").group)
    getter=Libdl.dlsym(
        Libdl.dlopen(Colmat.backend_library()), :colmat_fallback_has
    )
    for id in 0:7
        @test ccall(getter, Cint, (Cint, Cint), group, id)==1
    end
end
