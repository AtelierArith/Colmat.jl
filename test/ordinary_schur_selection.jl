using Test, LinearAlgebra, Libdl, Colmat
isdefined(@__MODULE__, :direct_ordinary_trexc!) || include("ordinary_schur_reorder.jl")
function ordinary_trsen_minimum(::Type{T}, job, n, m) where {T}
    nn=m*(n-m)
    lw=if job=='N'
        (T<:Real ? max(1, n) : 1)
    elseif job=='E'
        max(1, nn)
    else
        max(1, 2nn)
    end
    li=T<:Real&&job in ('V', 'B') ? max(1, nn) : 1
    return lw, li
end
for (T, prefix) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    J in (Int32, Int64)

    R=real(T)
    symbol=Symbol(prefix*"trsen_"*(J===Int64 ? "64_" : ""))
    signature=Any[
        Ref{UInt8}, Ref{UInt8}, Ptr{J}, Ref{J}, Ptr{T}, Ref{J}, Ptr{T}, Ref{J}, Ptr{T}
    ]
    args=Any[
        :job,
        :comp,
        :(view(mask, 2:(length(mask) - 1))),
        :n,
        :t,
        :ldt,
        :q,
        :ldq,
        :(view(w, 2:(length(w) - 1))),
    ]
    if T<:Real
        push!(signature, Ptr{T})
        push!(args, :(view(wi, 2:(length(wi) - 1))))
    end
    append!(signature, [Ptr{J}, Ptr{R}, Ptr{R}, Ptr{T}, Ref{J}])
    append!(
        args,
        [
            :(view(m, 2:2)),
            :(view(condition, 2:2)),
            :(view(sep, 2:2)),
            :(view(work, 2:(length(work) - 1))),
            :lwork,
        ],
    )
    if T<:Real
        append!(signature, [Ptr{J}, Ref{J}])
        append!(args, [:(view(iwork, 2:(length(iwork) - 1))), :liwork])
    end
    append!(signature, [Ptr{J}, Clong, Clong])
    append!(args, [:(view(info, 2:2)), 1, 1])
    @eval function direct_ordinary_trsen!(
        t::Matrix{$T},
        q::Matrix{$T},
        mask::Vector{$J},
        ::Type{$J};
        job=('N'),
        comp=('V'),
        n=size(t, 2),
        ldt=size(t, 1),
        ldq=size(q, 1),
        lwork=max(1, 4max(0, n)^2+4max(0, n)+16),
        liwork=max(1, max(0, n)^2+4max(0, n)+16),
        pointer=nothing,
    )
        f=if isnothing(pointer)
            Libdl.dlsym(Libdl.dlopen(Colmat.backend_library()), $(QuoteNode(symbol)))
        else
            pointer
        end
        query=lwork==-1
        w=fill($T(777), max(0, n)+2)
        wi=fill($R(777), max(0, n)+2)
        m=fill($J(-777), 3)
        info=copy(m)
        condition=fill($R(777), 3)
        sep=copy(condition)
        work=fill($T(777), (query ? 1 : max(1, lwork))+2)
        iwork=fill($J(-777), (query ? 1 : max(1, liwork))+2)
        ccall(f, Cvoid, $(Expr(:tuple, signature...)), $(args...))
        @test m[[1, 3]]==info[[1, 3]]==mask[[1, end]]==[$J(-777), $J(-777)]
        @test w[[1, end]]==[$T(777), $T(777)]&&wi[[1, end]]==[$R(777), $R(777)]
        @test condition[[1, 3]]==sep[[1, 3]]==[$R(777), $R(777)]
        @test work[[1, end]]==[$T(777), $T(777)]&&iwork[[1, end]]==[$J(-777), $J(-777)]
        values=$T<:Real ? complex.(w[2:(end - 1)], wi[2:(end - 1)]) : w[2:(end - 1)]
        (;
            info=info[2],
            diagnostic=ordinary_schur_diagnostic(),
            count=m[2],
            values,
            condition=condition[2],
            sep=sep[2],
            work=work[2:(end - 1)],
            iwork=iwork[2:(end - 1)],
        )
    end
end
@testset "TRSEN stable selection exact workspaces and expert estimates" begin
    Random.seed!(3122)
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
        n in (0, 1, 2, 5, 8),
        form in (:pairs, :triangular, :repeated, :zero),
        comp in ('N', 'V'),
        job in ('N', 'E', 'V', 'B')

        tol=T in (Float32, ComplexF32) ? 3e-3 : 5e-10
        initial=ordinary_schur_fixture(T, n, form)
        layout=ordinary_schur_layout(initial)
        basis=n==0 ? zeros(T, 0, 0) : Matrix(qr(randn(T, n, n)).Q)
        a=basis*initial*basis'
        eigenvalues=ordinary_schur_values(initial)
        for pattern in (:none, :all, :alternate, :second)
            flags=J[
                pattern===:all||pattern===:alternate&&isodd(j)||pattern===:second&&iseven(
                    j
                ) for j in 1:n
            ]
            chosen=[(j, len) for (j, len) in layout if any(!iszero, flags[j:(j + len - 1)])]
            others=[(j, len) for (j, len) in layout if all(iszero, flags[j:(j + len - 1)])]
            order=reduce(
                vcat,
                [collect(j:(j + len - 1)) for (j, len) in vcat(chosen, others)];
                init=Int[],
            )
            count=sum((len for (j, len) in chosen); init=0)
            lw, li=ordinary_trsen_minimum(T, job, n, count)
            t=vcat(initial, fill(T(777), 3, n))
            tr=copy(t)
            old=copy(t)
            q=comp=='V' ? vcat(basis, fill(T(777), 3, n)) : fill(T(777), 1, 1)
            qref=copy(q)
            oldq=copy(q)
            mask=vcat(J(-777), flags, J(-777))
            oldmask=copy(mask)
            mr=Int64.(mask)
            result=direct_ordinary_trsen!(t, q, mask, J; job, comp, lwork=lw, liwork=li)
            oracle=direct_ordinary_trsen!(
                tr,
                qref,
                mr,
                Int64;
                job,
                comp,
                lwork=lw,
                liwork=li,
                pointer=Libdl.dlsym(lib, Symbol(prefix*"trsen_64_")),
            )
            @test result.info==result.diagnostic==oracle.info==0
            @test result.count==oracle.count==count
            @test mask==mr==oldmask
            @test result.values≈oracle.values atol=tol rtol=tol
            @test result.values≈eigenvalues[order] atol=tol rtol=tol
            @test result.condition≈oracle.condition atol=tol rtol=tol
            @test result.sep≈oracle.sep atol=tol rtol=tol
            @test t[(n + 1):end, :]==tr[(n + 1):end, :]==old[(n + 1):end, :]
            if job=='N'
                @test result.condition==result.sep==777
            end
            if count==0||count==n
                @test isequal(t, old)&&isequal(tr, old)&&isequal(q, oldq)&&isequal(
                    qref, oldq
                )
            end
            if comp=='V'
                @test q[(n + 1):end, :]==qref[(n + 1):end, :]==oldq[(n + 1):end, :]
                @test q[1:n, :]*t[1:n, :]*q[1:n, :]'≈a atol=tol rtol=tol
                @test qref[1:n, :]*tr[1:n, :]*qref[1:n, :]'≈a atol=tol rtol=tol
                @test q[1:n, :]'*q[1:n, :]≈Matrix{T}(I, n, n) atol=tol rtol=tol
            else
                @test q==qref==oldq
            end
        end
    end
end
@testset "TRSEN query and insufficient workspace preserve untouched fields" begin
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
        J in (Int32, Int64), n in (0, 1, 5),
        job in ('N', 'E', 'V', 'B')

        initial=ordinary_schur_fixture(T, n, :pairs)
        flags=J.(iseven.(1:n))
        layout=ordinary_schur_layout(initial)
        count=sum(
            (len for (j, len) in layout if any(!iszero, flags[j:(j + len - 1)])); init=0
        )
        lw, li=ordinary_trsen_minimum(T, job, n, count)
        cases=if T<:Real
            ((-1, -9, 0), (lw-1, li, -15), (lw, li-1, -17), (lw, -1, -17))
        else
            ((-1, -9, 0), (lw-1, li, -14))
        end
        for (lwork, liwork, expected) in cases
            t=vcat(initial, fill(T(777), 3, n))
            tr=copy(t)
            old=copy(t)
            q=fill(T(777), n+3, n)
            qref=copy(q)
            oldq=copy(q)
            mask=vcat(J(-777), flags, J(-777))
            mr=Int64.(mask)
            oldmask=copy(mask)
            result=direct_ordinary_trsen!(t, q, mask, J; job, lwork, liwork)
            oracle=direct_ordinary_trsen!(
                tr,
                qref,
                mr,
                Int64;
                job,
                lwork,
                liwork,
                pointer=Libdl.dlsym(lib, Symbol(prefix*"trsen_64_")),
            )
            @test result.info==oracle.info==expected
            @test result.diagnostic==-expected
            @test result.count==oracle.count==count
            @test mask==mr==oldmask
            @test isequal(t, old)&&isequal(tr, old)&&isequal(q, oldq)&&isequal(qref, oldq)
            @test result.condition==result.sep==oracle.condition==oracle.sep==777
            if expected==0
                @test real(result.work[1])==real(oracle.work[1])==lw
                if T<:Real
                    @test result.iwork[1]==oracle.iwork[1]==li
                end
            end
        end
    end
end
