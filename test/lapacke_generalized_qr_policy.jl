using Test, Colmat, LinearAlgebra, Libdl
isdefined(@__MODULE__, :lapacke288_gpair) || include("lapacke_generalized_qr_pairs.jl")
isdefined(@__MODULE__, :lapacke288_gdriver) || include("lapacke_generalized_qr_drivers.jl")
@testset "Generalized QR C header order, query preservation and NaN policy" begin
    native=Libdl.dlopen(Colmat.backend_library())
    diagnostic=Libdl.dlsym(native, :colmat_blas_last_error)
    getter=Libdl.dlsym(native, :LAPACKE_get_nancheck)
    setter=Libdl.dlsym(native, :LAPACKE_set_nancheck)
    old=ccall(getter, Cint, ())
    ccall(setter, Cvoid, (Cint,), 1)
    try
        for T in (Float32, Float64, ComplexF32, ComplexF64),
            J in (Int32, Int64), family in (:ggqrf, :ggrqf, :gglse, :ggglm),
            route in (:dll, :lbt_trampoline)

            pair=family in (:ggqrf, :ggrqf)
            rq=family==:ggrqf
            lse=family==:gglse
            a=fill(T(777), 40)
            b=fill(T(778), 40)
            ta=fill(T(779), 8)
            tb=fill(T(780), 8)
            c=fill(T(781), 8)
            d=fill(T(782), 8)
            x=fill(T(783), 8)
            w=fill(T(784), 12)
            av=view(a, 2:39)
            bv=view(b, 2:39)
            tav=view(ta, 2:7)
            tbv=view(tb, 2:7)
            cv=view(c, 2:7)
            dv=view(d, 2:7)
            xv=view(x, 2:7)
            wv=view(w, 2:11)
            function call(high, l, f, s, p, la, lb, lw; null=false)
                aa=null ? Ptr{T}(C_NULL) : av
                bb=null ? Ptr{T}(C_NULL) : bv
                if pair
                    t1=null ? Ptr{T}(C_NULL) : tav
                    t2=null ? Ptr{T}(C_NULL) : tbv
                    lapacke288_gpair(
                        T, J, rq, high, l, f, s, p, aa, la, t1, bb, lb, t2, wv, lw; route
                    )
                else
                    cc=null ? Ptr{T}(C_NULL) : cv
                    dd=null ? Ptr{T}(C_NULL) : dv
                    xx=null ? Ptr{T}(C_NULL) : xv
                    lapacke288_gdriver(
                        T,
                        J,
                        lse,
                        high,
                        l,
                        f,
                        s,
                        p,
                        aa,
                        la,
                        bb,
                        lb,
                        cc,
                        dd,
                        xx,
                        wv,
                        lw;
                        route,
                    )
                end
            end
            saved=deepcopy((a, b, ta, tb, c, d, x))
            for layout in (101, 102)
                # Valid square headers; queries and insufficient WORK cannot
                # read any numerical input or output pointer.
                @test call(false, layout, 1, 1, 1, 3, 4, -1; null=true)==0
                query=w[2]
                @test real(query)>=1
                w[2]=T(784)
                @test call(false, layout, 1, 1, 1, 3, 4, 0; null=true)==(pair ? -12 : -13)
                @test w[2]==query
                @test ccall(diagnostic, Cint, ())==(pair ? 11 : 12)
                @test isequal((a, b, ta, tb, c, d, x), saved)
            end
            @test call(false, 0, -1, -1, -1, 0, 0, -1; null=true)==-1
            @test call(false, 101, -1, 2, 2, 0, 0, -1; null=true)==-6
            @test call(false, 101, -1, 2, 2, 2, 0, -1; null=true)==(pair ? -9 : -8)
            @test isequal((a, b, ta, tb, c, d, x), saved)
            for layout in (101, 102)
                w[2]=T(784)
                @test call(false, layout, -1, 1, 1, 3, 4, -1; null=true)==-2
                @test ccall(diagnostic, Cint, ())==1
                @test pair ? real(w[2])>=1&&w[2]!=T(784) : w[2]==T(784)
                w[2]=T(784)
                @test call(false, layout, 1, -1, 1, 3, 4, -1; null=true)==-3
                @test ccall(diagnostic, Cint, ())==2
                @test pair ? real(w[2])>=1&&w[2]!=T(784) : w[2]==T(784)
                w[2]=T(784)
                @test call(false, layout, 1, 1, -1, 3, 4, -1; null=true)==-4
                @test ccall(diagnostic, Cint, ())==3
                @test pair ? real(w[2])>=1&&w[2]!=T(784) : w[2]==T(784)
            end
            if J===Int64
                @test call(false, 102, -4294967293, 0, 0, 1, 1, -1; null=true)==-2
                @test ccall(diagnostic, Cint, ())==1
                if pair
                    f, s, p=0, 4294967296, 0
                    @test call(false, 102, f, s, p, 1, rq ? s : 1, -1; null=true)==0
                    @test real(w[2])>=4294967296
                elseif lse
                    @test call(
                        false, 102, 0, 4294967296, 4294967296, 1, 4294967296, -1; null=true
                    )==0
                    @test real(w[2])>=8589934592
                else
                    @test call(false, 102, 0, 0, 4294967296, 1, 1, -1; null=true)==0
                    @test real(w[2])>=1
                end
            end
            @test w[[1, end]]==T[784, 784]
            for layout in (101, 102)
                # One logical element inside NaN padding. TAU and solution
                # vectors are output-only; their initial NaNs are ignored.
                fill!(a, T(NaN))
                fill!(b, T(NaN))
                fill!(ta, T(NaN))
                fill!(tb, T(NaN))
                fill!(c, T(NaN))
                fill!(d, T(NaN))
                fill!(x, T(NaN))
                a[2]=T(2)
                b[2]=T(3)
                c[2]=T(4)
                if lse
                    d[2]=T(6)
                end
                @test call(true, layout, 1, 1, 1, 3, 4, 0)==0
                a[2]=T(NaN)
                b[2]=T(NaN)
                c[2]=T(NaN)
                if lse
                    d[2]=T(NaN)
                end
                before=deepcopy((a, b, ta, tb, c, d, x))
                @test call(true, layout, 1, 1, 1, 3, 4, 0)==-5
                @test isequal((a, b, ta, tb, c, d, x), before)
                a[2]=T(2)
                @test call(true, layout, 1, 1, 1, 3, 4, 0)==(pair ? -8 : -7)
                b[2]=T(3)
                if !pair
                    @test call(true, layout, 1, 1, 1, 3, 4, 0)==-9
                    c[2]=T(4)
                    if lse
                        @test call(true, layout, 1, 1, 1, 3, 4, 0)==-10
                        d[2]=T(6)
                    end
                end
                @test call(true, layout, 1, 1, 1, 3, 4, 0)==0
                @test ccall(diagnostic, Cint, ())==0
            end
            if !pair
                reference=only(
                    filter(
                        l->l.interface==:ilp64&&occursin("openblas", lowercase(l.libname)),
                        BLAS.get_config().loaded_libs,
                    ),
                ).libname
                for failure in (1, 2), layout in (101, 102), high in (false, true)
                    f, s, p=lse ? (2, 2, 1) : (2, 1, 1)
                    ao=zeros(T, f, s)
                    bo=zeros(T, lse ? p : f, lse ? s : p)
                    if failure==1
                        for j in 1:s
                            ao[j, j]=one(T)
                        end
                    else
                        bo[end, end]=one(T)
                    end
                    co=ones(T, f)
                    d0=ones(T, lse ? p : s)
                    x0=zeros(T, lse ? s : p)
                    ar=vcat(ao, zeros(T, 1, s))
                    br=vcat(bo, zeros(T, 1, size(bo, 2)))
                    expected=lapacke288_gdriver_fortran(
                        T,
                        Int64,
                        lse,
                        f,
                        s,
                        p,
                        ar,
                        br,
                        copy(co),
                        copy(d0),
                        copy(x0),
                        reference,
                    )
                    @test expected==failure
                    fill!(a, T(777))
                    fill!(b, T(778))
                    fill!(c, T(781))
                    fill!(d, T(782))
                    fill!(x, T(783))
                    la=(layout==101 ? size(ao, 2) : size(ao, 1))+2
                    lb=(layout==101 ? size(bo, 2) : size(bo, 1))+2
                    for j in axes(ao, 2), i in axes(ao, 1)
                        a[1 + (layout == 101 ? (i - 1) * la + j : (j - 1) * la + i)]=ao[
                            i, j
                        ]
                    end
                    for j in axes(bo, 2), i in axes(bo, 1)
                        b[1 + (layout == 101 ? (i - 1) * lb + j : (j - 1) * lb + i)]=bo[
                            i, j
                        ]
                    end
                    c[2:(f + 1)]=co
                    d[2:(length(d0) + 1)]=d0
                    @test call(high, layout, f, s, p, la, lb, 10)==expected
                    @test ccall(diagnostic, Cint, ())==0
                    @test a[[1, end]]==T[777, 777] && b[[1, end]]==T[778, 778]
                    @test c[[1, end]]==T[781, 781] &&
                        d[[1, end]]==T[782, 782] &&
                        x[[1, end]]==T[783, 783]
                end
            end
            ccall(setter, Cvoid, (Cint,), 0)
            @test call(true, 102, 0, 0, 0, 1, 1, 0)==0
            ccall(setter, Cvoid, (Cint,), 1)
        end
    finally
        ccall(setter, Cvoid, (Cint,), old)
    end
end
