using Test, Colmat, LinearAlgebra, Libdl
isdefined(@__MODULE__, :lapacke288_selected_entry) || include("lapacke_calls.jl")

for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
    R=typeof(real(zero(T)))
    tail=T<:Complex ? [Ptr{R}] : []
    base=[Cint, Cchar, J, J, Ptr{T}, J, Ptr{R}, Ptr{T}, J, Ptr{T}, J]
    vdhigh=Expr(:tuple, Cint, Cchar, Cchar, base[3:end]..., Ptr{R})
    ddhigh=Expr(:tuple, base...)
    vdwork=Expr(:tuple, Cint, Cchar, Cchar, base[3:end]..., Ptr{T}, J, tail...)
    ddwork=Expr(:tuple, base..., Ptr{T}, J, tail..., Ptr{J})
    vdtail=T<:Complex ? [:rwork] : []
    @eval function lapacke288_svd_call(
        ::Type{$T},
        ::Type{$J},
        pointer,
        dd,
        high,
        layout,
        ju,
        jv,
        m,
        n,
        a,
        lda,
        s,
        u,
        ldu,
        vt,
        ldvt,
        workspace,
        lwork,
        rwork,
        iwork,
        superb,
        name,
        route,
    )
        lapacke288_selected_entry(name, $J, route) do
            if dd
                if high
                    ccall(
                        pointer, $J, $ddhigh, layout, ju, m, n, a, lda, s, u, ldu, vt, ldvt
                    )
                else
                    ccall(
                        pointer,
                        $J,
                        $ddwork,
                        layout,
                        ju,
                        m,
                        n,
                        a,
                        lda,
                        s,
                        u,
                        ldu,
                        vt,
                        ldvt,
                        workspace,
                        lwork,
                        $(vdtail...),
                        iwork,
                    )
                end
            else
                if high
                    ccall(
                        pointer,
                        $J,
                        $vdhigh,
                        layout,
                        ju,
                        jv,
                        m,
                        n,
                        a,
                        lda,
                        s,
                        u,
                        ldu,
                        vt,
                        ldvt,
                        superb,
                    )
                else
                    ccall(
                        pointer,
                        $J,
                        $vdwork,
                        layout,
                        ju,
                        jv,
                        m,
                        n,
                        a,
                        lda,
                        s,
                        u,
                        ldu,
                        vt,
                        ldvt,
                        workspace,
                        lwork,
                        $(vdtail...),
                    )
                end
            end
        end
    end
end

for T in (Float32, Float64, ComplexF32, ComplexF64)
    R=typeof(real(zero(T)))
    args=[
        Ref{UInt8},
        Ref{UInt8},
        Ref{Int64},
        Ref{Int64},
        Ptr{T},
        Ref{Int64},
        Ptr{R},
        Ptr{T},
        Ref{Int64},
        Ptr{T},
        Ref{Int64},
        Ptr{T},
        Ref{Int64},
    ]
    tail=T<:Complex ? [Ptr{R}] : []
    signature=Expr(:tuple, args..., tail..., Ref{Int64})
    values=T<:Complex ? [:rwork] : []
    @eval function lapacke288_svd_reference(::Type{$T}, pointer, original)
        m, n=size(original)
        a=copy(original)
        s=zeros($R, min(m, n))
        dummy=zeros($T, 1)
        rwork=zeros($R, max(1, 5min(m, n)))
        query=zeros($T, 1)
        info=Ref{Int64}(0)
        ccall(
            pointer,
            Cvoid,
            $signature,
            UInt8('N'),
            UInt8('N'),
            m,
            n,
            a,
            max(1, m),
            s,
            dummy,
            1,
            dummy,
            1,
            query,
            -1,
            $(values...),
            info,
        )
        @test info[]==0
        workspace=zeros($T, max(1, Int(real(query[1]))))
        ccall(
            pointer,
            Cvoid,
            $signature,
            UInt8('N'),
            UInt8('N'),
            m,
            n,
            a,
            max(1, m),
            s,
            dummy,
            1,
            dummy,
            1,
            workspace,
            length(workspace),
            $(values...),
            info,
        )
        @test info[]==0
        s
    end
end

function lapacke288_svd(::Type{T}, ::Type{J}) where {T,J}
    R=typeof(real(zero(T)))
    prefix=Dict(Float32=>"s", Float64=>"d", ComplexF32=>"c", ComplexF64=>"z")[T]
    suffix=J===Int64 ? "64_" : ""
    native=Libdl.dlopen(Colmat.backend_library())
    trampoline=Libdl.dlopen(BLAS.libblastrampoline)
    reference=only(
        filter(
            l->l.interface==:ilp64 && occursin("openblas", lowercase(l.libname)),
            BLAS.get_config().loaded_libs,
        ),
    )
    reference_pointer=Libdl.dlsym(Libdl.dlopen(reference.libname), prefix*"gesvd_64_")
    @test reference_pointer!=Libdl.dlsym(native, prefix*"gesvd_64_")
    for (m, n) in ((3, 2), (2, 3), (0, 2), (2, 0))
        original=T[complex(3(i+1)+j*j, T<:Complex ? 0.3i-0.6j : 0) for i in 1:m, j in 1:n]
        expected=lapacke288_svd_reference(T, reference_pointer, original)
        r=min(m, n)
        for dd in (false, true),
            high in (false, true),
            route in (:dll, :lbt_trampoline),
            layout in (101, 102),
            ju in ('A', 'S', 'N', 'O'),
            jv in (dd ? (ju,) : ('A', 'S', 'N', 'O'))

            !dd && ju=='O' && jv=='O' && continue
            name="LAPACKE_"*prefix*(dd ? "gesdd" : "gesvd")*(high ? "" : "_work")
            @test BLAS.lbt_get_forward(name, J===Int64 ? :ilp64 : :lp64)==Libdl.dlsym(
                native, name*suffix
            )
            pointer=Libdl.dlsym(route===:dll ? native : trampoline, name*suffix)
            uc=ju=='A'||(dd && ju=='O' && m<n) ? m : r
            vr=jv=='A'||(dd && ju=='O' && m>=n) ? n : r
            use_u=ju in ('A', 'S')||(dd && ju=='O' && m<n)
            use_v=jv in ('A', 'S')||(dd && ju=='O' && m>=n)
            lda=(layout==101 ? n : m)+2
            ldu=(layout==101 ? uc : m)+2
            ldvt=(layout==101 ? n : vr)+2
            offset(i, j, ld) = layout==101 ? (i-1)*ld+j : i+(j-1)*ld
            poison=T(789)
            a=fill(poison, lda*(max(m, n)+2))
            u=fill(poison, ldu*(m+2))
            vt=fill(poison, ldvt*(n+2))
            for i in 1:m, j in 1:n
                a[offset(i, j, lda)]=original[i, j]
            end
            before=copy(a)
            s=zeros(R, max(1, r))
            superb=fill(R(-77), max(1, r))
            w=zeros(T, 1)
            rw=zeros(R, 256)
            iw=zeros(J, 8max(1, r))
            # Null outputs are legal for jobs that do not return them separately.
            up=use_u ? u : Ptr{T}(C_NULL)
            vp=use_v ? vt : Ptr{T}(C_NULL)
            call(w, lw) = lapacke288_svd_call(
                T,
                J,
                pointer,
                dd,
                high,
                layout,
                ju,
                jv,
                m,
                n,
                a,
                lda,
                s,
                up,
                ldu,
                vp,
                ldvt,
                w,
                lw,
                rw,
                iw,
                superb,
                name,
                route,
            )
            if !high
                @test call(w, -1)==0
                @test a==before
                @test all(==(poison), u) && all(==(poison), vt)
                w=zeros(T, Int(real(w[1])))
            end
            @test call(w, length(w))==0
            tolerance=R===Float32 ? 5e-5 : 2e-12
            @test s[1:r]≈expected rtol=tolerance atol=tolerance
            @test all(>=(zero(R)), s[1:r])
            @test issorted(s[1:r]; rev=true)
            has_u=ju!='N'
            has_v=jv!='N'
            left=ju=='O' && (!dd || m>=n) ? a : u
            right=(!dd && jv=='O')||(dd && ju=='O' && m<n) ? a : vt
            ll=left===a ? lda : ldu
            lr=right===a ? lda : ldvt
            U=T[left[offset(i, j, ll)] for i in 1:m, j in 1:uc]
            V=T[right[offset(i, j, lr)] for i in 1:vr, j in 1:n]
            if has_u && has_v
                @test U[:, 1:r]*Diagonal(s[1:r])*V[1:r, :]≈original rtol=tolerance atol=tolerance
            end
            if has_u
                @test U'*U≈Matrix{T}(I, uc, uc) rtol=tolerance atol=tolerance
            end
            if has_v
                @test V*V'≈Matrix{T}(I, vr, vr) rtol=tolerance atol=tolerance
            end
            amask=falses(length(a))
            umask=falses(length(u))
            vmask=falses(length(vt))
            for i in 1:m, j in 1:n
                amask[offset(i, j, lda)]=true
            end
            if use_u
                for i in 1:m, j in 1:uc
                    umask[offset(i, j, ldu)]=true
                end
            end
            if use_v
                for i in 1:vr, j in 1:n
                    vmask[offset(i, j, ldvt)]=true
                end
            end
            @test a[.!amask]==before[.!amask]
            @test all(==(poison), u[.!umask])
            @test all(==(poison), vt[.!vmask])
            @test superb[end]==R(-77)
        end
    end
end

@testset "LAPACKE SVD: independent reference, layouts, jobs and integer ABIs" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
        lapacke288_svd(T, J)
    end
end
