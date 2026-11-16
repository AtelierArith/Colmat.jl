using Test, Colmat, LinearAlgebra, Libdl
isdefined(@__MODULE__, :lapacke288_selected_entry) || include("lapacke_calls.jl")
isdefined(@__MODULE__, :range_boundary_probe) || include("selfadjoint_range_boundaries.jl")

for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
    R=typeof(real(zero(T)))
    base=[
        Cint,
        Cchar,
        Cchar,
        Cchar,
        J,
        Ptr{T},
        J,
        R,
        R,
        J,
        J,
        R,
        Ptr{J},
        Ptr{R},
        Ptr{T},
        J,
        Ptr{J},
    ]
    high_signature=Expr(:tuple, base...)
    work_signature=Expr(
        :tuple, base..., Ptr{T}, J, (T<:Complex ? [Ptr{R}, J] : [])..., Ptr{J}, J
    )
    extra=T<:Complex ? [:rwork, :lr] : []
    @eval function lapacke288_range_call(
        ::Type{$T},
        ::Type{$J},
        p,
        high,
        layout,
        job,
        mode,
        uplo,
        n,
        a,
        ld,
        vl,
        vu,
        il,
        iu,
        abstol,
        m,
        w,
        z,
        lz,
        support,
        work,
        lw,
        rwork,
        lr,
        iwork,
        li,
        name,
        route,
    )
        lapacke288_selected_entry(name, $J, route) do
            if high
                ccall(
                    p,
                    $J,
                    $high_signature,
                    layout,
                    job,
                    mode,
                    uplo,
                    n,
                    a,
                    ld,
                    vl,
                    vu,
                    il,
                    iu,
                    abstol,
                    m,
                    w,
                    z,
                    lz,
                    support,
                )
            else
                ccall(
                    p,
                    $J,
                    $work_signature,
                    layout,
                    job,
                    mode,
                    uplo,
                    n,
                    a,
                    ld,
                    vl,
                    vu,
                    il,
                    iu,
                    abstol,
                    m,
                    w,
                    z,
                    lz,
                    support,
                    work,
                    lw,
                    $(extra...),
                    iwork,
                    li,
                )
            end
        end
    end
end

function lapacke288_range(::Type{T}, ::Type{J}) where {T,J}
    R=typeof(real(zero(T)))
    prefix=Dict(Float32=>"s", Float64=>"d", ComplexF32=>"c", ComplexF64=>"z")[T]
    family=T<:Complex ? "heevr" : "syevr"
    suffix=J===Int64 ? "64_" : ""
    native=Libdl.dlopen(Colmat.backend_library())
    trampoline=Libdl.dlopen(BLAS.libblastrampoline)
    reference=only(
        filter(
            l->l.interface==:ilp64 && occursin("openblas", lowercase(l.libname)),
            BLAS.get_config().loaded_libs,
        ),
    )
    refp=Libdl.dlsym(Libdl.dlopen(reference.libname), prefix*family*"_64_")
    @test refp!=Libdl.dlsym(native, prefix*family*"_64_")
    for n in (0, 1, 3),
        job in ('N', 'V'), mode in ('A', 'I', 'V'), uplo in ('L', 'U', 'l', 'u'),
        abstol in (0.0, 1e-4)

        original=if n==0
            zeros(T, 0, 0)
        elseif n==1
            fill(T(T<:Complex ? 2 : 4), 1, 1)
        elseif T<:Complex
            T[2 1+im 0; 1-im 2 1+im; 0 1-im 2]
        else
            T[2 2 0; 2 2 0; 0 0 6]
        end
        vl=if T<:Complex
            2.0
        elseif n==1
            4.0
        else
            3.0
        end
        il=n>1 ? 2 : 1
        iu=n
        expected=range_boundary_probe(
            T, Int64, refp, original, job, mode, uppercase(uplo), vl, 10.0, il, iu, abstol
        )
        for high in (false, true),
            route in (:dll, :lbt_trampoline), layout in (101, 102),
            imaginary_nan in (T<:Complex && !high ? (false, true) : (false,))

            name="LAPACKE_"*prefix*family*(high ? "" : "_work")
            @test BLAS.lbt_get_forward(name, J===Int64 ? :ilp64 : :lp64)==Libdl.dlsym(
                native, name*suffix
            )
            p=Libdl.dlsym(route===:dll ? native : trampoline, name*suffix)
            ld=max(1, n)+2
            lz=max(1, n)+2
            offset(i, j, stride) = layout==101 ? (i-1)*stride+j : i+(j-1)*stride
            a=fill(T(NaN), ld*max(1, n))
            amask=falses(length(a))
            for i in 1:n, j in 1:n
                selected=uppercase(uplo)=='L' ? i>=j : i<=j
                if selected
                    a[offset(i, j, ld)]=if imaginary_nan && i==j
                        T(complex(real(original[i, j]), NaN))
                    else
                        original[i, j]
                    end
                    amask[offset(i, j, ld)]=true
                end
            end
            before=copy(a)
            z=fill(T(-91), lz*max(1, n))
            w=fill(R(-91), n+2)
            support=fill(J(-91), 2max(1, n)+2)
            m=Ref{J}(-91)
            work=zeros(T, 1)
            rwork=zeros(R, max(1, 24n))
            iwork=fill(J(-91), max(1, 10n))
            zp=job=='V' ? z : Ptr{T}(C_NULL)
            call(work, lw) = lapacke288_range_call(
                T,
                J,
                p,
                high,
                layout,
                job,
                mode,
                uplo,
                n,
                a,
                ld,
                vl,
                10.0,
                il,
                iu,
                abstol,
                m,
                w,
                zp,
                lz,
                support,
                work,
                lw,
                rwork,
                length(rwork),
                iwork,
                length(iwork),
                name,
                route,
            )
            if !high
                @test call(work, -1)==0
                @test m[]==-91 && isequal(a, before)
                @test all(==(R(-91)), w) && all(==(T(-91)), z)
                work=zeros(T, Int(real(work[1])))
                @test call(work, length(work)-1)==-19
                @test m[]==-91 && isequal(a, before)
            end
            @test call(work, length(work))==0
            @test m[]==expected.count
            tol=R===Float32 ? 5e-5 : 2e-12
            @test w[1:m[]]≈expected.values rtol=tol atol=tol
            @test all(==(R(-91)), w[(m[] + 1):end])
            @test support[(end - 1):end]==fill(J(-91), 2)
            @test isequal(a[.!amask], before[.!amask])
            columns=job=='V' ? (n==1 ? 1 : Int(m[])) : 0
            zmask=falses(length(z))
            for i in 1:n, j in 1:columns
                zmask[offset(i, j, lz)]=true
            end
            @test all(==(T(-91)), z[.!zmask])
            if job=='V'
                vectors=T[z[offset(i, j, lz)] for i in 1:n, j in 1:Int(m[])]
                @test original*vectors≈vectors*Diagonal(w[1:m[]]) rtol=tol atol=max(
                    tol, abstol
                )
                @test vectors'*vectors≈Matrix{T}(I, Int(m[]), Int(m[])) rtol=tol atol=tol
                if n==1
                    @test z[1]==one(T) && support[1:2]==J[1, 1]
                end
            end
        end
    end
end

@testset "LAPACKE SYEVR/HEEVR: interval selection, layouts and integer ABIs" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
        lapacke288_range(T, J)
    end
end
