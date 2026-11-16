using Test, Colmat, LinearAlgebra, Libdl

for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
    R=typeof(real(zero(T)))
    signature=Expr(
        :tuple,
        Ref{UInt8},
        Ref{UInt8},
        Ref{UInt8},
        Ref{J},
        Ptr{T},
        Ref{J},
        Ref{R},
        Ref{R},
        Ref{J},
        Ref{J},
        Ref{R},
        Ref{J},
        Ptr{R},
        Ptr{T},
        Ref{J},
        Ptr{J},
        Ptr{T},
        Ref{J},
        (T<:Complex ? [Ptr{R}, Ref{J}] : [])...,
        Ptr{J},
        Ref{J},
        Ref{J},
    )
    extra=T<:Complex ? [:rwork, :lr] : []
    @eval function range_boundary_probe(
        ::Type{$T}, ::Type{$J}, p, original, job, mode, uplo, vl, vu, il, iu, abstol
    )
        n=size(original, 1)
        ld=max(1, n)+2
        lz=max(1, n)+1
        a=fill($T(NaN), ld*max(1, n))
        z=fill($T(-91), lz*max(1, n))
        w=fill($R(-91), n+2)
        for i in 1:n, j in 1:n
            if uplo=='L' ? i>=j : i<=j
                a[i + (j - 1) * ld]=original[i, j]
            end
        end
        before=copy(a)
        support=fill($J(-91), 2max(1, n)+2)
        iw=fill($J(-91), max(1, 10n))
        work=zeros($T, 1)
        rwork=zeros($R, max(1, 24n))
        m=Ref{$J}(-91)
        info=Ref{$J}(-91)
        lr=length(rwork)
        li=length(iw)
        zp=job=='V' ? z : Ptr{$T}(C_NULL)
        ccall(
            p,
            Cvoid,
            $signature,
            UInt8(job),
            UInt8(mode),
            UInt8(uplo),
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
            zp,
            lz,
            support,
            work,
            -1,
            $(extra...),
            iw,
            li,
            info,
        )
        @test info[]==0
        @test isequal(a, before) && m[]==-91
        work=zeros($T, max(1, Int(real(work[1]))))
        lw=length(work)
        # Scalar and empty calls accept the reference's one-element minimum.
        if n<=1
            work=zeros($T, 1)
            rwork=zeros($R, 1)
            iw=fill($J(-91), 1)
            lw=lr=li=1
        end
        ccall(
            p,
            Cvoid,
            $signature,
            UInt8(job),
            UInt8(mode),
            UInt8(uplo),
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
            zp,
            lz,
            support,
            work,
            lw,
            $(extra...),
            iw,
            li,
            info,
        )
        @test info[]==0
        @test 0<=m[]<=n
        @test all(==($R(-91)), w[(m[] + 1):end])
        @test support[(end - 1):end]==fill($J(-91), 2)
        if job=='V'
            columns=n==1 ? 1 : Int(m[])
            for j in 1:max(1, n), i in 1:lz
                if i>n || j>columns
                    @test z[i + (j - 1) * lz]==$T(-91)
                end
            end
            if n==1
                @test z[1]==one($T) && support[1:2]==$J[1, 1]
            end
        end
        vectors=$T[z[i + (j - 1) * lz] for i in 1:n, j in 1:Int(m[])]
        (count=Int(m[]), values=w[1:Int(m[])], vectors=vectors)
    end
end

@testset "Self-adjoint range endpoints, tolerances and scalar workspace boundaries" begin
    native=Libdl.dlopen(Colmat.backend_library())
    reference=only(
        filter(
            l->l.interface==:ilp64 && occursin("openblas", lowercase(l.libname)),
            BLAS.get_config().loaded_libs,
        ),
    )
    external=Libdl.dlopen(reference.libname)
    for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
        R=typeof(real(zero(T)))
        prefix=Dict(Float32=>"s", Float64=>"d", ComplexF32=>"c", ComplexF64=>"z")[T]
        routine=prefix*(T<:Complex ? "heevr" : "syevr")*"_"
        own=Libdl.dlsym(native, routine*(J===Int64 ? "64_" : ""))
        refp=Libdl.dlsym(external, routine*"64_")
        @test own!=refp
        for n in (0, 1, 3),
            job in ('N', 'V'), mode in ('A', 'I', 'V'), uplo in ('L', 'U'),
            abstol in (-1.0, 0.0, 1e-4)

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
            actual=range_boundary_probe(
                T, J, own, original, job, mode, uplo, vl, 10.0, il, iu, abstol
            )
            expected=range_boundary_probe(
                T, Int64, refp, original, job, mode, uplo, vl, 10.0, il, iu, abstol
            )
            @test actual.count==expected.count
            tol=R===Float32 ? 5e-5 : 2e-12
            @test actual.values≈expected.values rtol=tol atol=tol
            if T<:Complex && n==3 && mode=='V'
                @test actual.count==1
            end
            if job=='V'
                @test original*actual.vectors≈actual.vectors*Diagonal(actual.values) rtol=tol atol=max(
                    tol, max(0.0, abstol)
                )
                @test actual.vectors'*actual.vectors≈Matrix{T}(
                    I, actual.count, actual.count
                ) rtol=tol atol=tol
            end
        end
    end
end
