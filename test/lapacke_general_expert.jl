using Test, Colmat, LinearAlgebra, Libdl
isdefined(@__MODULE__, :lapacke288_selected_entry) || include("lapacke_calls.jl")
isdefined(@__MODULE__, :direct_geevx!) || include("geevx_expert_helpers.jl")

for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
    R=real(T)
    signature=Any[Cint, Cchar, Cchar, Cchar, Cchar, J, Ptr{T}, J, Ptr{T}]
    wiargs=T<:Real ? [:(view(x.wi, 2:(n + 1)))] : []
    T<:Real && push!(signature, Ptr{R})
    append!(
        signature, [Ptr{T}, J, Ptr{T}, J, Ptr{J}, Ptr{J}, Ptr{R}, Ptr{R}, Ptr{R}, Ptr{R}]
    )
    highsig=Expr(:tuple, signature...)
    worksig=Expr(:tuple, signature..., Ptr{T}, J, T<:Real ? Ptr{J} : Ptr{R})
    aux=if T<:Real
        :(view(x.iwork, 2:(length(x.iwork) - 1)))
    else
        :(view(x.rwork, 2:(length(x.rwork) - 1)))
    end
    @eval function lapacke288_geevx_call(
        ::Type{$T},
        ::Type{$J},
        p,
        high,
        layout,
        bal,
        jl,
        jr,
        sense,
        n,
        x,
        ld,
        lwork,
        name,
        route,
    )
        lapacke288_selected_entry(name, $J, route) do
            if high
                ccall(
                    p,
                    $J,
                    $highsig,
                    layout,
                    bal,
                    jl,
                    jr,
                    sense,
                    n,
                    x.a,
                    ld,
                    view(x.w, 2:(n + 1)),
                    $(wiargs...),
                    x.left,
                    ld,
                    x.right,
                    ld,
                    view(x.ilo, 2:2),
                    view(x.ihi, 2:2),
                    view(x.scale, 2:(n + 1)),
                    view(x.abnrm, 2:2),
                    view(x.rconde, 2:(n + 1)),
                    view(x.rcondv, 2:(n + 1)),
                )
            else
                ccall(
                    p,
                    $J,
                    $worksig,
                    layout,
                    bal,
                    jl,
                    jr,
                    sense,
                    n,
                    x.a,
                    ld,
                    view(x.w, 2:(n + 1)),
                    $(wiargs...),
                    x.left,
                    ld,
                    x.right,
                    ld,
                    view(x.ilo, 2:2),
                    view(x.ihi, 2:2),
                    view(x.scale, 2:(n + 1)),
                    view(x.abnrm, 2:2),
                    view(x.rconde, 2:(n + 1)),
                    view(x.rcondv, 2:(n + 1)),
                    view(x.work, 2:(length(x.work) - 1)),
                    lwork,
                    $aux,
                )
            end
        end
    end
end

@testset "LAPACKE GEEVX balancing, condition outputs and integer ABIs" begin
    native=Libdl.dlopen(Colmat.backend_library())
    trampoline=Libdl.dlopen(BLAS.libblastrampoline)
    reference=geevx_reference()
    for (T, prefix) in
        ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
        J in (Int32, Int64),
        n in (0, 1, 3, 5),
        bal in ('N', 'P', 'S', 'B'),
        sense in ('N', 'E', 'V', 'B'),
        jl in ('N', 'V'),
        jr in ('N', 'V')

        sense in ('E', 'B') && (jl!='V'||jr!='V') && continue
        R=real(T)
        ld=n+2
        original=if n==0
            zeros(T, 0, 0)
        elseif n==1
            fill(T(7), 1, 1)
        elseif n==3
            T[1 -2 1; 2 1 2; 0 0 5]
        else
            T[sin(i+2j)+(i==j ? i : 0) for i in 1:n, j in 1:n]
        end
        T<:Complex && n>0 && (original .+= T(im)*I(n))
        refmatrix=zeros(T, ld, n)
        refmatrix[1:n, :].=original
        refp=Libdl.dlsym(Libdl.dlopen(reference), prefix*"geevx_64_")
        @test refp!=Libdl.dlsym(native, prefix*"geevx_64_")
        expected=direct_geevx!(
            refmatrix, Int64; bal, left=jl, right=jr, sense, pointer=refp
        )
        @test expected.info==0
        expected_values=if T<:Real
            complex.(expected.values, expected.imaginary)
        else
            expected.values
        end
        tol=R(500)*eps(R)
        for route in (:dll, :lbt_trampoline), layout in (101, 102), high in (false, true)
            name="LAPACKE_"*prefix*"geevx"*(high ? "" : "_work")
            suffix=J==Int64 ? "64_" : ""
            @test BLAS.lbt_get_forward(name, J==Int64 ? :ilp64 : :lp64)==Libdl.dlsym(
                native, name*suffix
            )
            p=Libdl.dlsym(route==:dll ? native : trampoline, name*suffix)
            offset(i, j) = layout==101 ? (i-1)*ld+j : i+(j-1)*ld
            a=fill(T(777), ld*max(1, n))
            for i in 1:n, j in 1:n
                a[offset(i, j)]=original[i, j]
            end
            minimum=geevx_minimum(T, n, jl, jr, sense)
            x=(;
                a,
                left=fill(T(777), length(a)),
                right=fill(T(777), length(a)),
                w=fill(T(777), n+2),
                wi=fill(R(777), n+2),
                ilo=fill(J(-777), 3),
                ihi=fill(J(-777), 3),
                scale=fill(R(777), n+2),
                abnrm=fill(R(777), 3),
                rconde=fill(R(777), n+2),
                rcondv=fill(R(777), n+2),
                work=fill(T(777), minimum+2),
                iwork=fill(J(777), max(1, 2n-2)+2),
                rwork=fill(R(777), max(1, 2n)+2),
            )
            call(lw) = lapacke288_geevx_call(
                T,
                J,
                p,
                high,
                layout,
                high ? lowercase(bal) : bal,
                high ? lowercase(jl) : jl,
                high ? lowercase(jr) : jr,
                high ? lowercase(sense) : sense,
                n,
                x,
                ld,
                lw,
                name,
                route,
            )
            saved=deepcopy(x)
            if !high
                @test call(-1)==0
                for field in (
                    :a,
                    :left,
                    :right,
                    :w,
                    :wi,
                    :ilo,
                    :ihi,
                    :scale,
                    :abnrm,
                    :rconde,
                    :rcondv,
                    :iwork,
                )
                    @test getproperty(x, field)==getproperty(saved, field)
                end
                @test call(minimum-1)==(T<:Complex ? -21 : -22)
                @test a==saved.a
            end
            @test call(minimum)==0
            @test x.ilo[2]==expected.ilo && x.ihi[2]==expected.ihi
            @test x.scale[2:(n + 1)]≈expected.scale rtol=tol atol=tol
            @test x.abnrm[2]≈expected.abnrm rtol=tol atol=tol
            values=T<:Real ? complex.(x.w[2:(n + 1)], x.wi[2:(n + 1)]) : x.w[2:(n + 1)]
            unmatched=collect(1:n)
            for j in 1:n
                k=findfirst(
                    k->abs(values[j]-expected_values[k])<=tol*max(
                        1, abs(expected_values[k])
                    ),
                    unmatched,
                )
                @test k!==nothing
                k===nothing && continue
                index=popat!(unmatched, k)
                sense in ('E', 'B') &&
                    @test x.rconde[j + 1]≈expected.rconde[index] rtol=tol atol=tol
                sense in ('V', 'B') &&
                    @test x.rcondv[j + 1]≈expected.rcondv[index] rtol=tol atol=tol
            end
            @test isempty(unmatched)
            if jl=='V'||jr=='V'
                schur=T[x.a[offset(i, j)] for i in 1:n, j in 1:n]
                for j in 1:n, i in (j + (T <: Complex ? 1 : 2)):n
                    @test abs(schur[i, j])<=tol
                end
                @test tr(schur)≈sum(values) rtol=tol*10 atol=tol*10
                @test det(schur)≈prod(values) rtol=tol*10 atol=tol*10
            end
            for (job, data, left) in ((jl, x.left, true), (jr, x.right, false))
                if job=='N'
                    @test all(==(T(777)), data)
                    continue
                end
                matrix=T[data[offset(i, j)] for i in 1:n, j in 1:n]
                vectors=geevx_unpack(matrix, x.wi[2:(n + 1)])
                @test (left ? original' : original)*vectors≈vectors*Diagonal(
                    left ? conj.(values) : values
                ) atol=tol*20
                for j in 1:n
                    @test norm(vectors[:, j])≈1 atol=tol
                    largest=maximum(abs, view(vectors, :, j))
                    @test any(
                        v->abs(abs(v)-largest)<=tol && abs(imag(v))<=tol,
                        view(vectors, :, j),
                    )
                end
            end
            for field in (:w, :wi, :scale, :rconde, :rcondv, :work, :iwork, :rwork)
                @test getproperty(x, field)[[1, end]]==getproperty(saved, field)[[1, end]]
            end
            for field in (:ilo, :ihi, :abnrm)
                @test getproperty(x, field)[[1, 3]]==getproperty(saved, field)[[1, 3]]
            end
            for k in eachindex(a)
                if k>ld*n||(k-1)%ld>=n
                    @test x.a[k]==saved.a[k] && x.left[k]==T(777) && x.right[k]==T(777)
                end
            end
            sense in ('N', 'V') && @test x.rconde==saved.rconde
            sense in ('N', 'E') && @test x.rcondv==saved.rcondv
        end
    end
end
