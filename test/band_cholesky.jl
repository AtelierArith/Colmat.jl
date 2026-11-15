using Test, LinearAlgebra, Random, Libdl, Colmat

function pb_reference()
    return only(
        filter(
            l->l.interface==:ilp64 && occursin("openblas", lowercase(basename(l.libname))),
            BLAS.get_config().loaded_libs,
        ),
    ).libname
end
function pb_last_error()
    return ccall(
        Libdl.dlsym(Libdl.dlopen(Colmat.backend_library()), :colmat_blas_last_error),
        Cint,
        (),
    )
end
for (T, prefix) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    J in (Int32, Int64)

    suffix=J===Int64 ? "64_" : ""
    @eval function direct_pb_factor!(
        ab::Matrix{$T},
        ::Type{$J};
        n=size(ab, 2),
        kd,
        u=('U'),
        family="pbtrf",
        ld=size(ab, 1),
        library=Colmat.backend_library(),
    )
        info=fill($J(777), 3)
        f=Libdl.dlsym(Libdl.dlopen(library), Symbol($prefix*family*"_"*$suffix))
        ccall(
            f,
            Cvoid,
            (Ref{UInt8}, Ref{$J}, Ref{$J}, Ptr{$T}, Ref{$J}, Ptr{$J}),
            u,
            n,
            kd,
            ab,
            ld,
            view(info, 2:2),
        )
        @test info[[1, 3]]==[777, 777]
        info[2]
    end
    @eval function direct_pb_solve!(
        ab::Matrix{$T},
        b::Matrix{$T},
        ::Type{$J};
        n=size(ab, 2),
        kd,
        u=('U'),
        nrhs=size(b, 2),
        ld=size(ab, 1),
        ldb=size(b, 1),
        family="pbtrs",
        library=Colmat.backend_library(),
    )
        info=fill($J(777), 3)
        f=Libdl.dlsym(Libdl.dlopen(library), Symbol($prefix*family*"_"*$suffix))
        ccall(
            f,
            Cvoid,
            (
                Ref{UInt8},
                Ref{$J},
                Ref{$J},
                Ref{$J},
                Ptr{$T},
                Ref{$J},
                Ptr{$T},
                Ref{$J},
                Ptr{$J},
            ),
            u,
            n,
            kd,
            nrhs,
            ab,
            ld,
            view(b, 2:size(b, 1), :),
            ldb,
            view(info, 2:2),
        )
        @test info[[1, 3]]==[777, 777]
        info[2]
    end
end

function pb_matrix(T, n, kd, rng)
    upper=zeros(T, n, n)
    for j in 1:n, i in max(1, j - kd):j
        upper[i, j]=i==j ? T(2+i/16) : randn(rng, T)/8
    end
    # An independent scalar product, without routing the oracle through BLAS.
    a=zeros(T, n, n)
    for j in 1:n, i in 1:n, k in 1:n
        a[i, j]+=conj(upper[k, i])*upper[k, j]
    end
    return a
end
function pb_storage(a, kd, u; poison=true)
    T=eltype(a)
    n=size(a, 1)
    ab=fill(T(NaN), kd+3, n)
    ab[(end - 1):end, :].=777
    for j in 1:n
        indices=u=='U' ? (max(1, j - kd):j) : (j:min(n, j + kd))
        for i in indices
            row=u=='U' ? kd+1+i-j : 1+i-j
            ab[row, j]=poison && T<:Complex && i==j ? T(real(a[i, j]), NaN) : a[i, j]
        end
    end
    return ab
end
function pb_factor_matrix(ab, n, kd, u; split=false)
    T=eltype(ab)
    s=zeros(T, n, n)
    middle=min(n, (n+kd)÷2)
    for j in 1:n
        indices=u=='U' ? (max(1, j - kd):j) : (j:min(n, j + kd))
        for i in indices
            value=ab[u=='U' ? kd+1+i-j : 1+i-j, j]
            if !split
                s[i, j]=value
            elseif u=='U'
                if j>middle
                    s[j, i]=conj(value)
                else
                    s[i, j]=value
                end
            elseif i>middle
                s[i, j]=value
            else
                s[j, i]=conj(value)
            end
        end
    end
    return s
end
function pb_reconstruct(s, u, split)
    n=size(s, 1)
    a=zeros(eltype(s), n, n)
    for j in 1:n, i in 1:n, k in 1:n
        a[i, j]+=split||u=='U' ? conj(s[k, i])*s[k, j] : s[i, k]*conj(s[j, k])
    end
    return a
end
function pb_assert_unused(ab, before, n, kd, u)
    @test all(==(777), ab[(end - 1):end, :])
    for j in 1:n, row in 1:(kd + 1)
        i=u=='U' ? row+j-kd-1 : row+j-1
        if !(1<=i<=n)
            @test isequal(ab[row, j], before[row, j])
        end
    end
end

@testset "Band and split Cholesky reconstruct independent positive matrices" begin
    reference=pb_reference()
    rng=MersenneTwister(283)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64), n in 0:9, kd in (0, 1, 2, 5, 12), u in ('U', 'L'),
        family in ("pbtrf", "pbtf2", "pbstf")

        a=pb_matrix(T, n, kd, rng)
        ab=pb_storage(a, kd, u)
        before=copy(ab)
        @test direct_pb_factor!(ab, J; kd, u, family)==0
        R=typeof(real(zero(T)))
        tol=500eps(R)
        s=pb_factor_matrix(ab, n, kd, u; split=family=="pbstf")
        @test pb_reconstruct(s, u, family=="pbstf")≈a rtol=tol atol=tol
        for j in 1:n
            @test imag(ab[u=='U' ? kd+1 : 1, j])==0
        end
        pb_assert_unused(ab, before, n, kd, u)
        # Reference PBSTF uses m=(n+kd)/2 without checking m<=n. Avoid its
        # out-of-bounds column accesses for redundant widths; Colmat's clipped
        # split is still independently reconstructed above.
        if family!="pbstf" || n==0 || (n+kd)÷2<=n
            external=copy(before)
            @test direct_pb_factor!(external, Int64; kd, u, family, library=reference)==0
            @test isapprox(ab, external; rtol=tol, atol=tol, nans=true)
        end
    end
end

@testset "Band Cholesky solve and driver interoperate with external factors" begin
    reference=pb_reference()
    rng=MersenneTwister(1283)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        n in (0, 1, 3, 8, 35),
        kd in (0, 1, 4, 38),
        u in ('U', 'L'),
        nrhs in (0, 1, 3)

        a=pb_matrix(T, n, kd, rng)
        initial=pb_storage(a, kd, u)
        ab=copy(initial)
        external=copy(initial)
        @test direct_pb_factor!(ab, J; kd, u)==0
        @test direct_pb_factor!(external, Int64; kd, u, library=reference)==0
        x=randn(rng, T, n, nrhs)
        rhs=zeros(T, n, nrhs)
        for c in 1:nrhs, i in 1:n, j in 1:n
            rhs[i, c]+=a[i, j]*x[j, c]
        end
        before=fill(T(777), n+3, nrhs)
        before[2:(n + 1), :]=rhs
        R=typeof(real(zero(T)))
        tol=1500eps(R)
        saved=copy(ab)
        for (factors, provider) in (
            (ab, Colmat.backend_library()),
            (external, Colmat.backend_library()),
            (ab, reference),
            (external, reference),
        )
            b=copy(before)
            K=provider==reference ? Int64 : J
            @test direct_pb_solve!(factors, b, K; kd, u, library=provider)==0
            @test b[2:(n + 1), :]≈x rtol=tol atol=tol
            @test all(==(777), b[[1, n+2, n+3], :])
        end
        @test isequal(ab, saved)
        b=copy(before)
        ab=copy(initial)
        @test direct_pb_solve!(ab, b, J; kd, u, family="pbsv")==0
        @test b[2:(n + 1), :]≈x rtol=tol atol=tol
        @test all(==(777), b[[1, n+2, n+3], :])
    end
end

@testset "Nonpositive band and split pivots preserve partial factor contracts" begin
    reference=pb_reference()
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        n in (1, 3, 8),
        kd in (0, 1, 2),
        u in ('U', 'L'),
        position in 1:n,
        bad in (0, -1),
        family in ("pbtrf", "pbtf2", "pbstf")

        a=Matrix{T}(I, n, n)
        a[position, position]=T(bad)
        ab=pb_storage(a, kd, u)
        before=copy(ab)
        external=copy(ab)
        @test direct_pb_factor!(ab, J; kd, u, family)==position
        @test direct_pb_factor!(external, Int64; kd, u, family, library=reference)==position
        @test isequal(ab, external)
        pb_assert_unused(ab, before, n, kd, u)
        if family=="pbtrf"
            b=fill(T(777), n+3, 2)
            ab=copy(before)
            @test direct_pb_solve!(ab, b, J; kd, u, family="pbsv")==position
            @test all(==(777), b)
        end
    end
end

@testset "All 20 band Cholesky symbols compute through both LBT interfaces" begin
    rng=MersenneTwister(2283)
    for (T, prefix) in
        ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
        J in (Int32, Int64)

        abi=J===Int64 ? :ilp64 : :lp64
        suffix=J===Int64 ? "64_" : ""
        for family in ("pbtrf", "pbtf2", "pbstf", "pbtrs", "pbsv")
            pointer=Libdl.dlsym(
                Libdl.dlopen(Colmat.backend_library()), Symbol(prefix*family*"_"*suffix)
            )
            @test BLAS.lbt_get_forward(prefix*family*"_", abi)==pointer
        end
        n=5
        kd=2
        u='L'
        a=pb_matrix(T, n, kd, rng)
        R=typeof(real(zero(T)))
        tol=1000eps(R)
        for family in ("pbtrf", "pbtf2", "pbstf")
            ab=pb_storage(a, kd, u)
            @test direct_pb_factor!(ab, J; kd, u, family, library=BLAS.libblastrampoline)==0
            s=pb_factor_matrix(ab, n, kd, u; split=family=="pbstf")
            @test pb_reconstruct(s, u, family=="pbstf")≈a rtol=tol atol=tol
        end
        x=ones(T, n, 2)
        rhs=zeros(T, n, 2)
        for c in 1:2, i in 1:n, j in 1:n
            rhs[i, c]+=a[i, j]*x[j, c]
        end
        for family in ("pbtrs", "pbsv")
            ab=pb_storage(a, kd, u)
            b=fill(T(777), n+3, 2)
            b[2:(n + 1), :]=rhs
            if family=="pbtrs"
                @test direct_pb_factor!(ab, J; kd, u)==0
            end
            @test direct_pb_solve!(
                ab, b, J; kd, u, family, library=BLAS.libblastrampoline
            )==0
            @test b[2:(n + 1), :]≈x rtol=tol atol=tol
        end
    end
end

@testset "Band Cholesky diagnostics integer width and unused RHS" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64), J in (Int32, Int64)
        ab=fill(T(777), 5, 3)
        b=fill(T(777), 6, 1)
        for family in ("pbtrf", "pbtf2", "pbstf"),
            (changes, arg) in
            (((; u=('X'), n=-1), 1), ((; n=-1), 2), ((; kd=-1), 3), ((; ld=1), 5))

            args=merge((; n=3, kd=1, u=('U'), ld=5), changes)
            @test direct_pb_factor!(ab, J; args..., family)==-arg
            @test pb_last_error()==arg
        end
        for family in ("pbtrs", "pbsv"),
            (changes, arg) in (
                ((; u=('X'), n=-1), 1),
                ((; n=-1), 2),
                ((; kd=-1), 3),
                ((; nrhs=-1), 4),
                ((; ld=1), 6),
                ((; ldb=2), 8),
            )

            args=merge((; n=3, kd=1, u=('U'), ld=5, nrhs=1, ldb=6), changes)
            @test direct_pb_solve!(ab, b, J; args..., family)==-arg
            @test pb_last_error()==arg
        end
        @test all(==(777), ab)
        @test all(==(777), b)
        if J===Int64
            large=(Int64(1)<<32)+1
            negative=-(Int64(1)<<32)+1
            @test direct_pb_factor!(ab, J; n=negative, kd=1)==-2
            @test pb_last_error()==2
            @test direct_pb_solve!(ab, b, J; n=large, kd=1, nrhs=0, ldb=large)==0
            @test all(==(777), b)
        end
        for u in ('U', 'L')
            a=Matrix{T}(I, 3, 3)
            native=pb_storage(a, 1, u)
            before=copy(b)
            @test direct_pb_solve!(native, b, J; kd=1, u, nrhs=0, family="pbsv")==0
            @test isequal(b, before)
        end
    end
end

@testset "PBSTF redundant widths never access columns beyond N" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        u in ('U', 'L')

        kd=4
        ab=zeros(T, 7, 2)
        row=u=='U' ? kd+1 : 1
        ab[row, 1]=T<:Complex ? T(4, NaN) : T(4)
        ab[row, 2]=T(-1)
        before=copy(ab[:, 2])
        @test direct_pb_factor!(ab, J; n=1, kd, u, family="pbstf")==0
        @test ab[row, 1]==2
        @test isequal(ab[:, 2], before)
    end
end

@testset "NaN real diagonals propagate through ordinary and split Cholesky" begin
    reference=pb_reference()
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64), u in ('U', 'L'), position in 1:3,
        family in ("pbtrf", "pbtf2", "pbstf")

        a=Matrix{T}(I, 3, 3)
        a[position, position]=T(NaN)
        ab=pb_storage(a, 1, u)
        external=copy(ab)
        expected=0
        @test direct_pb_factor!(ab, J; kd=1, u, family)==expected
        @test direct_pb_factor!(external, Int64; kd=1, u, family, library=reference)==expected
        @test isapprox(ab, external; nans=true)
    end
end
