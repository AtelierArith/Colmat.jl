using Test, LinearAlgebra, Libdl, Colmat
isdefined(@__MODULE__, :direct_ordinary_trexc!) || include("ordinary_schur_reorder.jl")
const ordinary_geesx_calls=Ref(0)
for (T, prefix) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    J in (Int32, Int64)

    R=real(T)
    low=Symbol("ordinary_geesx_low_", prefix, "_", J)
    positive=Symbol("ordinary_geesx_imag_", prefix, "_", J)
    lowptr=Symbol(low, "_pointer")
    positiveptr=Symbol(positive, "_pointer")
    if T<:Real
        @eval begin
            function $low(re::Ptr{$T}, im::Ptr{$T})::$J
                ordinary_geesx_calls[]+=1
                $J(unsafe_load(re)<1.25)
            end
            function $positive(re::Ptr{$T}, im::Ptr{$T})::$J
                ordinary_geesx_calls[]+=1
                $J(unsafe_load(im)>0)
            end
        end
        @eval begin
            const $lowptr=@cfunction($low, $J, (Ptr{$T}, Ptr{$T}))
            const $positiveptr=@cfunction($positive, $J, (Ptr{$T}, Ptr{$T}))
        end
    else
        @eval begin
            function $low(value::Ptr{$T})::$J
                ordinary_geesx_calls[]+=1
                $J(real(unsafe_load(value))<1.25)
            end
            function $positive(value::Ptr{$T})::$J
                ordinary_geesx_calls[]+=1
                $J(imag(unsafe_load(value))>0)
            end
        end
        @eval begin
            const $lowptr=@cfunction($low, $J, (Ptr{$T},))
            const $positiveptr=@cfunction($positive, $J, (Ptr{$T},))
        end
    end
    @eval ordinary_geesx_callback(::Type{$T}, ::Type{$J}, filter) =
        filter===:imag ? $positiveptr : $lowptr
    symbol=Symbol(prefix*"geesx_"*(J===Int64 ? "64_" : ""))
    signature=Any[
        Ref{UInt8},
        Ref{UInt8},
        Ptr{Cvoid},
        Ref{UInt8},
        Ref{J},
        Ptr{T},
        Ref{J},
        Ptr{J},
        Ptr{T},
    ]
    args=Any[
        :job,
        :sort,
        :(sort=='S' ? selection : C_NULL),
        :sense,
        :n,
        :a,
        :lda,
        :(view(count, 2:2)),
        :(view(w, 2:(length(w) - 1))),
    ]
    if T<:Real
        push!(signature, Ptr{T})
        push!(args, :(view(wi, 2:(length(wi) - 1))))
    end
    append!(signature, [Ptr{T}, Ref{J}, Ptr{R}, Ptr{R}, Ptr{T}, Ref{J}])
    append!(
        args,
        [
            :vs,
            :ldvs,
            :(view(rconde, 2:2)),
            :(view(rcondv, 2:2)),
            :(view(work, 2:(length(work) - 1))),
            :(query===:work ? -1 : lwork),
        ],
    )
    if T<:Real
        append!(signature, [Ptr{J}, Ref{J}])
        append!(
            args, [:(view(iwork, 2:(length(iwork) - 1))), :(query===:integer ? -1 : liwork)]
        )
    else
        push!(signature, Ptr{R})
        push!(args, :(view(rwork, 2:(length(rwork) - 1))))
    end
    append!(signature, [Ptr{J}, Ptr{J}, Clong, Clong, Clong])
    append!(args, [:(view(bwork, 2:(length(bwork) - 1))), :(view(info, 2:2)), 1, 1, 1])
    @eval function direct_ordinary_geesx!(
        a::Matrix{$T},
        vs::Matrix{$T},
        ::Type{$J};
        job=('V'),
        sort=('S'),
        sense=('N'),
        n=size(a, 2),
        lda=size(a, 1),
        ldvs=size(vs, 1),
        lwork=max(1, 4max(0, n)^2+4max(0, n)+16),
        liwork=max(1, max(0, n)^2+16),
        query=:none,
        selection=ordinary_geesx_callback($T, $J, :low),
        pointer=nothing,
    )
        f=if isnothing(pointer)
            Libdl.dlsym(Libdl.dlopen(Colmat.backend_library()), $(QuoteNode(symbol)))
        else
            pointer
        end
        w=fill($T(777), max(0, n)+2)
        wi=fill($R(777), max(0, n)+2)
        count=fill($J(-777), 3)
        info=copy(count)
        rconde=fill($R(777), 3)
        rcondv=copy(rconde)
        work=fill($T(777), (query===:none ? max(1, lwork) : 1)+2)
        iwork=fill($J(-777), (query===:none ? max(1, liwork) : 1)+2)
        rwork=fill($R(777), max(1, n)+2)
        bwork=fill($J(-777), max(1, n)+2)
        ordinary_geesx_calls[]=0
        ccall(f, Cvoid, $(Expr(:tuple, signature...)), $(args...))
        calls=ordinary_geesx_calls[]
        @test count[[1, 3]]==info[[1, 3]]==[$J(-777), $J(-777)]
        @test w[[1, end]]==[$T(777), $T(777)]&&wi[[1, end]]==[$R(777), $R(777)]
        @test rconde[[1, 3]]==rcondv[[1, 3]]==[$R(777), $R(777)]
        @test work[[1, end]]==[$T(777), $T(777)]&&rwork[[1, end]]==[$R(777), $R(777)]
        @test iwork[[1, end]]==bwork[[1, end]]==[$J(-777), $J(-777)]
        values=$T<:Real ? complex.(w[2:(end - 1)], wi[2:(end - 1)]) : w[2:(end - 1)]
        (;
            info=info[2],
            diagnostic=ordinary_schur_diagnostic(),
            count=count[2],
            values,
            rconde=rconde[2],
            rcondv=rcondv[2],
            work=work[2:(end - 1)],
            iwork=iwork[2:(end - 1)],
            bwork=bwork[2:(end - 1)],
            calls,
        )
    end
end
ordinary_geesx_take(value, filter) = filter===:imag ? imag(value)>0 : real(value)<1.25
function ordinary_geesx_cluster(values, ::Type{T}, filter) where {T}
    selected=falses(length(values))
    j=1
    while j<=length(values)
        pair=T<:Real&&j<length(values)&&!iszero(imag(values[j]))
        len=pair ? 2 : 1
        selected[j:(j + len - 1)].=any(
            v->ordinary_geesx_take(v, filter), values[j:(j + len - 1)]
        )
        j+=len
    end
    return selected
end
@testset "GEESX native sorting and original expert conditions both integer ABIs" begin
    Random.seed!(3123)
    lib=Libdl.dlopen(
        only(
            filter(
                l->l.interface==:ilp64&&occursin(
                    "openblas", lowercase(basename(l.libname))
                ),
                BLAS.get_config().loaded_libs,
            ),
        ).libname,
    )
    for (T, prefix) in
        ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
        J in (Int32, Int64),
        n in (0, 1, 3, 6),
        form in (:dense, :pairs, :repeated, :zero),
        job in ('N', 'V'),
        sort in ('N', 'S'),
        sense in (sort=='S' ? ('N', 'E', 'V', 'B') : ('N',)),
        filter in (sort=='S' ? (:low, :imag) : (:low,))

        tol=T in (Float32, ComplexF32) ? 4e-3 : 5e-10
        initial=if form===:dense
            randn(T, n, n)
        elseif form===:zero
            zeros(T, n, n)
        elseif form===:repeated
            -2Matrix{T}(I, n, n)
        else
            t=ordinary_schur_fixture(T, n, :pairs)
            if n>0
                for (j, len) in ordinary_schur_layout(t)
                    for k in j:(j + len - 1)
                        t[k, k]=T(j==1 ? -3 : 0.5+j)+(T<:Complex ? T(0, 0.1j) : zero(T))
                    end
                end
            end
            basis=n==0 ? zeros(T, 0, 0) : Matrix(qr(randn(T, n, n)).Q)
            basis*t*basis'
        end
        a=vcat(initial, fill(T(777), 3, n))
        ar=copy(a)
        old=copy(a)
        vs=fill(T(777), job=='V' ? n+3 : 1, job=='V' ? n : 1)
        vr=copy(vs)
        oldvs=copy(vs)
        selection=ordinary_geesx_callback(T, J, filter)
        result=direct_ordinary_geesx!(a, vs, J; job, sort, sense, selection)
        oracle=direct_ordinary_geesx!(
            ar,
            vr,
            Int64;
            job,
            sort,
            sense,
            selection=ordinary_geesx_callback(T, Int64, filter),
            pointer=Libdl.dlsym(lib, Symbol(prefix*"geesx_64_")),
        )
        @test result.info==result.diagnostic==oracle.info==0
        @test result.count==oracle.count
        @test Base.sort(result.values; by=v->(real(v), imag(v)))≈Base.sort(
            oracle.values; by=v->(real(v), imag(v))
        ) atol=tol rtol=tol
        @test result.values≈ordinary_schur_values(a) atol=tol rtol=tol
        @test result.rconde≈oracle.rconde atol=tol rtol=tol
        @test result.rcondv≈oracle.rcondv atol=tol rtol=tol
        @test a[(n + 1):end, :]==ar[(n + 1):end, :]==old[(n + 1):end, :]
        if sense=='N'
            @test result.rconde==result.rcondv==777
        end
        if sort=='S'&&n>0
            @test result.calls>=n&&oracle.calls>=n
            selected=ordinary_geesx_cluster(result.values, T, filter)
            @test count(selected)==result.count
            @test all(selected[1:result.count])&&!any(selected[(result.count + 1):end])
        else
            @test result.calls==oracle.calls==0
            @test all(==(-777), result.bwork)&&all(==(-777), oracle.bwork)
        end
        if job=='V'
            @test vs[(n + 1):end, :]==vr[(n + 1):end, :]==oldvs[(n + 1):end, :]
            @test vs[1:n, :]*a[1:n, :]*vs[1:n, :]'≈initial atol=tol rtol=tol
            @test vr[1:n, :]*ar[1:n, :]*vr[1:n, :]'≈initial atol=tol rtol=tol
            @test vs[1:n, :]'*vs[1:n, :]≈Matrix{T}(I, n, n) atol=tol rtol=tol
        else
            @test vs==vr==oldvs
        end
    end
end
@testset "GEESX queries preserve data and do not invoke SELECT" begin
    lib=Libdl.dlopen(
        only(
            filter(
                l->l.interface==:ilp64&&occursin(
                    "openblas", lowercase(basename(l.libname))
                ),
                BLAS.get_config().loaded_libs,
            ),
        ).libname,
    )
    for (T, prefix) in
        ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
        J in (Int32, Int64), n in (0, 1, 5), sense in ('N', 'E', 'V', 'B'),
        query in (T<:Real ? (:work, :integer) : (:work,))

        a=fill(T(777), n+3, n)
        old=copy(a)
        ar=copy(a)
        vs=fill(T(777), 1, 1)
        vr=copy(vs)
        result=direct_ordinary_geesx!(
            a, vs, J; sense, query, ldvs=max(1, n), lwork=-9, liwork=-9, selection=C_NULL
        )
        oracle=direct_ordinary_geesx!(
            ar,
            vr,
            Int64;
            sense,
            query,
            ldvs=max(1, n),
            lwork=-9,
            liwork=-9,
            selection=C_NULL,
            pointer=Libdl.dlsym(lib, Symbol(prefix*"geesx_64_")),
        )
        @test result.info==result.diagnostic==oracle.info==0
        @test result.calls==oracle.calls==0
        @test result.count==oracle.count==-777
        @test isequal(a, old)&&isequal(ar, old)&&vs==vr==fill(T(777), 1, 1)
        @test result.rconde==result.rcondv==oracle.rconde==oracle.rcondv==777
        @test all(==(-777), result.bwork)&&all(==(-777), oracle.bwork)
        minimum=max(1, (T<:Real ? 3 : 2)*n)
        @test real(result.work[1])>=minimum&&real(oracle.work[1])>=minimum
        if sense=='N'
            @test real(result.work[1])==minimum
        else
            @test result.work[1]==oracle.work[1]
        end
        if T<:Real
            @test result.iwork[1]==oracle.iwork[1]
        end
    end
end
