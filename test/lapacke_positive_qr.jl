using Test, Colmat, LinearAlgebra, Libdl
isdefined(@__MODULE__, :lapacke288_qr_call)||include("lapacke_qr.jl")
@testset "LAPACKE GEQRFP nonnegative diagonal and independent Q consumers" begin
    native=Libdl.dlopen(Colmat.backend_library())
    trampoline=Libdl.dlopen(BLAS.libblastrampoline)
    reference=Libdl.dlopen(
        only(
            filter(
                l->l.interface==:ilp64&&occursin("openblas", lowercase(l.libname)),
                BLAS.get_config().loaded_libs,
            ),
        ).libname,
    )
    shapes=((0, 0), (0, 3), (3, 0), (1, 1), (2, 4), (4, 2), (3, 3), (7, 5), (5, 7))
    for (T, prefix) in
        ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
        J in (Int32, Int64),
        (m, n) in shapes,
        blocked in (true,),
        deficient in (false, true),
        exponent in (-1, 0, 1)

        R=real(T)
        scale=R(2)^(exponent*(R==Float32 ? 40 : 400))
        tol=R(500)*eps(R)
        k=min(m, n)
        minimum=m==0||n==0 ? 1 : n
        original=T[cos(i+2j)+(i==j ? 2 : 0) for i in 1:m, j in 1:n]
        T<:Complex&&(original .+= T(im) .* T[sin(2i+j)/5 for i in 1:m, j in 1:n])
        deficient&&n>0&&(original[:, 1].=0)
        original .*= scale
        family="geqrfp"
        refp=Libdl.dlsym(reference, prefix*family*"_64_")
        qname=prefix*(T<:Complex ? "ungqr" : "orgqr")*"_64_"
        qref=Libdl.dlsym(reference, qname)
        qown=Libdl.dlsym(native, qname)
        @test refp!=Libdl.dlsym(native, prefix*family*"_64_")&&qref!=qown
        expected=lapacke288_qr_reference(T, refp, blocked, original)
        independent_q=lapacke288_qr_form_q(T, qown, expected.a, expected.tau)
        denom=max(norm(original), floatmin(R))
        @test norm(independent_q*triu(expected.a)-original)/denom<=tol*20
        @test independent_q'*independent_q≈Matrix{T}(I, m, m) atol=tol
        for route in (:dll, :lbt_trampoline), layout in (101, 102), high in (false, true)
            name="LAPACKE_"*prefix*family*(high ? "" : "_work")
            suffix=J==Int64 ? "64_" : ""
            @test BLAS.lbt_get_forward(name, J==Int64 ? :ilp64 : :lp64)==Libdl.dlsym(
                native, name*suffix
            )
            p=Libdl.dlsym(route==:dll ? native : trampoline, name*suffix)
            ld=layout==101 ? n+2 : m+2
            at(i, j) = layout==101 ? (i-1)*ld+j : i+(j-1)*ld
            a=fill(T(NaN), layout==101 ? max(1, m)*ld : max(1, n)*ld)
            mask=falses(length(a))
            for i in 1:m, j in 1:n
                a[at(i, j)]=original[i, j]
                mask[at(i, j)]=true
            end
            before=copy(a)
            tau=fill(T(777), k+2)
            work=fill(T(777), max(1, n)+2)
            call(lw) = lapacke288_qr_call(
                T,
                J,
                p,
                blocked,
                high,
                layout,
                m,
                n,
                a,
                ld,
                view(tau, 2:(k + 1)),
                view(work, 2:(length(work) - 1)),
                lw,
                name,
                route,
            )
            if blocked&&!high
                @test call(-1)==0
                @test isequal(a, before)&&all(==(T(777)), tau)
                @test real(work[2])>=minimum
                if m==0||n==0
                    @test work[2]==one(T)
                end
                @test call(minimum-1)==-8
                @test isequal(a, before)&&all(==(T(777)), tau)
            end
            @test call(minimum)==0
            @test isequal(a[.!mask], before[.!mask])
            @test tau[[1, end]]==work[[1, end]]==fill(T(777), 2)
            if blocked&&!high&&(m==0||n==0)
                @test work[2]==one(T)
            end
            factors=T[a[at(i, j)] for i in 1:m, j in 1:n]
            @test all(i->isreal(factors[i, i])&&real(factors[i, i])>=0, 1:k)
            @test work[minimum + 2]==T(777)
            q=lapacke288_qr_form_q(T, qref, factors, tau[2:(k + 1)])
            @test norm(q*triu(factors)-original)/denom<=tol*20
            @test q'*q≈Matrix{T}(I, m, m) atol=tol
        end
    end
end
