using Test, LinearAlgebra, Libdl, Colmat
isdefined(@__MODULE__, :direct_ordinary_geesx!) || include("ordinary_schur_expert.jl")
for (T, prefix) in ((Float32, "s"), (Float64, "d")), J in (Int32, Int64)
    choose=Symbol("ordinary_rejected_gees_select_", prefix, "_", J)
    callback=Symbol(choose, "_pointer")
    limit=T===Float32 ? Float32(1+2.0^-15) : 1.25
    @eval function $choose(re::Ptr{$T}, im::Ptr{$T})::$J
        $J(unsafe_load(re)>$limit)
    end
    @eval const $callback=@cfunction($choose, $J, (Ptr{$T}, Ptr{$T}))
    @eval ordinary_rejected_gees_callback(::Type{$T}, ::Type{$J}) = $callback
    symbol=Symbol(prefix*"gees_"*(J===Int64 ? "64_" : ""))
    @eval function direct_ordinary_gees!(
        a::Matrix{$T},
        vs::Matrix{$T},
        ::Type{$J};
        job=('V'),
        sort=('S'),
        selection=ordinary_rejected_gees_callback($T, $J),
        pointer=nothing,
    )
        n=size(a, 2)
        f=if isnothing(pointer)
            Libdl.dlsym(Libdl.dlopen(Colmat.backend_library()), $(QuoteNode(symbol)))
        else
            pointer
        end
        wr=fill($T(777), n+2)
        wi=copy(wr)
        work=fill($T(777), max(1, 4n*n+4n+16)+2)
        count=fill($J(-777), 3)
        info=copy(count)
        bwork=fill($J(-777), max(1, n)+2)
        ccall(
            f,
            Cvoid,
            (
                Ref{UInt8},
                Ref{UInt8},
                Ptr{Cvoid},
                Ref{$J},
                Ptr{$T},
                Ref{$J},
                Ptr{$J},
                Ptr{$T},
                Ptr{$T},
                Ptr{$T},
                Ref{$J},
                Ptr{$T},
                Ref{$J},
                Ptr{$J},
                Ptr{$J},
                Clong,
                Clong,
            ),
            job,
            sort,
            sort=='S' ? selection : C_NULL,
            n,
            a,
            size(a, 1),
            view(count, 2:2),
            view(wr, 2:(n + 1)),
            view(wi, 2:(n + 1)),
            vs,
            size(vs, 1),
            view(work, 2:(length(work) - 1)),
            length(work)-2,
            view(bwork, 2:(length(bwork) - 1)),
            view(info, 2:2),
            1,
            1,
        )
        @test count[[1, 3]]==info[[1, 3]]==[$J(-777), $J(-777)]
        @test wr[[1, end]]==wi[[1, end]]==work[[1, end]]==[$T(777), $T(777)]
        @test bwork[[1, end]]==[$J(-777), $J(-777)]
        (;
            info=info[2],
            diagnostic=ordinary_schur_diagnostic(),
            count=count[2],
            values=complex.(wr[2:(end - 1)], wi[2:(end - 1)]),
        )
    end
end
@testset "GEES and GEESX report the full selected dimension on reorder refusal" begin
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
    for (T, prefix) in ((Float32, "s"), (Float64, "d")),
        J in (Int32, Int64), job in ('N', 'V'), family in (:gees, :geesx),
        sense in (family===:gees ? ('N',) : ('N', 'E', 'V', 'B'))

        initial=ordinary_rejected_window(T)
        n=4
        a=vcat(initial, fill(T(777), 3, n))
        ar=copy(a)
        old=copy(a)
        vs=fill(T(777), job=='V' ? n+3 : 1, job=='V' ? n : 1)
        vr=copy(vs)
        oldvs=copy(vs)
        before=copy(a)
        before_ref=copy(ar)
        before_vs=copy(vs)
        before_vsr=copy(vr)
        selection=ordinary_rejected_gees_callback(T, J)
        reference_selection=ordinary_rejected_gees_callback(T, Int64)
        if family===:gees
            raw=direct_ordinary_gees!(before, before_vs, J; job, sort=('N'))
            raw_ref=direct_ordinary_gees!(
                before_ref,
                before_vsr,
                Int64;
                job,
                sort=('N'),
                pointer=Libdl.dlsym(lib, Symbol(prefix*"gees_64_")),
            )
            result=direct_ordinary_gees!(a, vs, J; job, selection)
            oracle=direct_ordinary_gees!(
                ar,
                vr,
                Int64;
                job,
                selection=reference_selection,
                pointer=Libdl.dlsym(lib, Symbol(prefix*"gees_64_")),
            )
        else
            raw=direct_ordinary_geesx!(before, before_vs, J; job, sort=('N'))
            raw_ref=direct_ordinary_geesx!(
                before_ref,
                before_vsr,
                Int64;
                job,
                sort=('N'),
                pointer=Libdl.dlsym(lib, Symbol(prefix*"geesx_64_")),
            )
            result=direct_ordinary_geesx!(a, vs, J; job, sense, selection)
            oracle=direct_ordinary_geesx!(
                ar,
                vr,
                Int64;
                job,
                sense,
                selection=reference_selection,
                pointer=Libdl.dlsym(lib, Symbol(prefix*"geesx_64_")),
            )
            @test result.rconde==oracle.rconde&&result.rcondv==oracle.rcondv
        end
        if family===:geesx&&sense!='N'
            # Expert modes run the saved original decomposition as well as
            # its sorting; compare the refusal state with that same backend.
            before=copy(before_ref)
            before_vs=copy(before_vsr)
        end
        @test raw.info==raw.diagnostic==raw_ref.info==0
        @test result.info==oracle.info==n+1&&result.diagnostic==0
        @test result.count==oracle.count==2
        @test isequal(a, before)&&isequal(vs, before_vs)
        @test isequal(ar, before_ref)&&isequal(vr, before_vsr)
        @test a[(n + 1):end, :]==ar[(n + 1):end, :]==old[(n + 1):end, :]
        if job=='N'
            @test vs==vr==oldvs
        end
    end
end
