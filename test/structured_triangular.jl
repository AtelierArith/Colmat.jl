using Test, LinearAlgebra, Libdl, Colmat

triangular_blas_pointer(a::Matrix) = a
triangular_blas_pointer(a::Vector) = view(a, 2:(length(a) - 1))
for (T, p) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    J in (Int32, Int64)

    suffix=J===Int64 ? "64_" : ""
    @eval function direct_triangular_blas!(
        family,
        a::Array{$T},
        x::Vector{$T},
        ::Type{$J};
        n,
        k=0,
        u,
        tr,
        d,
        inc,
        library=Colmat.backend_library(),
    )
        f=Libdl.dlsym(Libdl.dlopen(library), Symbol($p*family*"_"*$suffix))
        source=triangular_blas_pointer(a)
        if startswith(family, "tb")
            ccall(
                f,
                Cvoid,
                (
                    Ref{UInt8},
                    Ref{UInt8},
                    Ref{UInt8},
                    Ref{$J},
                    Ref{$J},
                    Ptr{$T},
                    Ref{$J},
                    Ptr{$T},
                    Ref{$J},
                ),
                u,
                tr,
                d,
                n,
                k,
                source,
                size(a, 1),
                pointer(x, 2),
                inc,
            )
        else
            ccall(
                f,
                Cvoid,
                (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{$J}, Ptr{$T}, Ptr{$T}, Ref{$J}),
                u,
                tr,
                d,
                n,
                source,
                pointer(x, 2),
                inc,
            )
        end
    end
end
function triangular_blas_reference()
    return only(
        filter(
            l->l.interface==:ilp64 && occursin("openblas", lowercase(basename(l.libname))),
            BLAS.get_config().loaded_libs,
        ),
    ).libname
end
function triangular_blas_indices(n, inc)
    return if n==0
        Int[]
    else
        collect(inc>0 ? (2:inc:(2 + (n - 1) * inc)) : ((2 + (n - 1) * abs(inc)):inc:2))
    end
end
function triangular_blas_fixture(
    T, family, n, k, u, d; masked=false, finite_reference=false
)
    band=startswith(family, "tb")
    a=band ? fill(T(NaN), k+3, n+1) : fill(T(NaN), n*(n+1)÷2+2)
    if !band
        a[1]=a[end]=T(777)
    end
    dense=zeros(T, n, n)
    cursor=2
    for j in 1:n
        rows=u=='U' ? (1:j) : (j:n)
        for i in rows
            if band && abs(i-j)>k
                continue
            end
            value=if i==j
                (T <: Complex ? T(2+j/8, (j-2)/16) : T(2+j/8))
            else
                (T <: Complex ? T((-1)^(i+j)*(i+2j)/64, (i-j)/64) : T((-1)^(i+j)*(i+2j)/64))
            end
            dense[i, j]=i==j && d=='U' ? one(T) : value
            stored=i==j && d=='U' ? (T <: Complex ? T(NaN, NaN) : T(NaN)) : value
            if masked
                stored=if j==2&&!finite_reference
                    T(NaN)
                elseif i==j
                    T(2)
                else
                    T(0.5)
                end
                dense[i, j]=if j==2
                    zero(T)
                elseif i==j
                    T(2)
                else
                    T(0.5)
                end
            end
            if band
                a[u=='U' ? k+1+i-j : 1+i-j, j]=stored
            else
                a[cursor]=stored
                cursor+=1
            end
        end
    end
    return a, dense
end

@testset "Native triangular band/packed multiply and solve" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        family in ("tbmv", "tbsv", "tpmv", "tpsv"),
        n in 0:8,
        k in (startswith(family, "tb") ? (0, 1, 3, 10) : (0,)),
        u in ('U', 'L'),
        tr in ('N', 'T', 'C'),
        d in ('N', 'U'),
        inc in (-3, -1, 1, 2)

        R=typeof(real(zero(T)))
        a, dense=triangular_blas_fixture(T, family, n, k, u, d)
        indices=triangular_blas_indices(n, inc)
        x=fill(T(NaN), max(1, (n-1)*abs(inc)+1)+2)
        x[1]=x[end]=T(777)
        for (i, pos) in enumerate(indices)
            x[pos]=T <: Complex ? T((i-3)/4, (i+1)/16) : T((i-3)/4)
        end
        before=copy(x)
        beforea=copy(a)
        op=if tr=='N'
            dense
        elseif tr=='T'
            transpose(dense)
        else
            adjoint(dense)
        end
        direct_triangular_blas!(family, a, x, J; n, k, u, tr, d, inc)
        @test isequal(a, beforea)
        ignored=setdiff(eachindex(x), indices)
        @test isequal(x[ignored], before[ignored])
        if endswith(family, "sv")
            reconstructed=T[sum(op[i, j]*x[indices[j]] for j in 1:n) for i in 1:n]
            @test isapprox(reconstructed, before[indices]; rtol=500eps(R), atol=500eps(R))
        else
            expected=T[sum(op[i, j]*before[indices[j]] for j in 1:n) for i in 1:n]
            @test isapprox(x[indices], expected; rtol=200eps(R), atol=200eps(R))
        end
        if J===Int64
            external=copy(before)
            direct_triangular_blas!(
                family,
                a,
                external,
                J;
                n,
                k,
                u,
                tr,
                d,
                inc,
                library=triangular_blas_reference(),
            )
            @test isapprox(x[indices], external[indices]; rtol=500eps(R), atol=500eps(R))
            @test isequal(external[ignored], before[ignored])
        end
    end
end

@testset "Triangular no-transpose zero columns do not read stored NaNs" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        family in ("tbmv", "tbsv", "tpmv", "tpsv"),
        u in ('U', 'L'),
        inc in (-2, 1, 3)

        R=typeof(real(zero(T)))
        n=3
        k=1
        a, dense=triangular_blas_fixture(T, family, n, k, u, 'N'; masked=true)
        indices=triangular_blas_indices(n, inc)
        solution=T[1, 0, 2]
        original=if endswith(family, "sv")
            T[sum(dense[i, j]*solution[j] for j in 1:n) for i in 1:n]
        else
            solution
        end
        x=fill(T(777), max(1, (n-1)*abs(inc)+1)+2)
        x[indices]=original
        before=copy(x)
        direct_triangular_blas!(family, a, x, J; n, k, u, tr=('N'), d=('N'), inc)
        expected=if endswith(family, "sv")
            solution
        else
            T[sum(dense[i, j]*solution[j] for j in 1:n) for i in 1:n]
        end
        @test x[indices]==expected
        external=copy(before)
        # Netlib's no-transpose algorithms skip zero columns. Optimized
        # providers can evaluate zero*NaN instead. Compare an equivalent
        # finite matrix whose column 2 still multiplies a zero solution.
        # The original NaN matrix remains in the Colmat call above.
        reference_a, _=triangular_blas_fixture(
            T, family, n, k, u, 'N'; masked=true, finite_reference=true
        )
        direct_triangular_blas!(
            family,
            reference_a,
            external,
            Int64;
            n,
            k,
            u,
            tr=('N'),
            d=('N'),
            inc,
            library=triangular_blas_reference(),
        )
        @test isequal(x, external)
        a.=T(NaN)
        x[indices].=T <: Complex ? T(-0.0, -0.0) : T(-0.0)
        before=copy(x)
        direct_triangular_blas!(family, a, x, J; n, k, u, tr=('N'), d=('N'), inc)
        @test isequal(x, before)
    end
end

@testset "Unit diagonal-only transforms preserve every component bit" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        family in ("tbmv", "tbsv", "tpmv", "tpsv"),
        tr in ('N', 'T', 'C'),
        inc in (-2, 1)

        n=startswith(family, "tb") ? 5 : 1
        a, _=triangular_blas_fixture(T, family, n, 0, 'L', 'U')
        R=typeof(real(zero(T)))
        values=R[NaN, Inf, -0.0, 0.0, 2]
        indices=triangular_blas_indices(n, inc)
        x=fill(T(777), max(1, (n-1)*abs(inc)+1)+2)
        for (i, pos) in enumerate(indices)
            x[pos]=T <: Complex ? T(values[i], -values[i]) : T(values[i])
        end
        before=copy(x)
        direct_triangular_blas!(family, a, x, J; n, k=0, u=('L'), tr, d=('U'), inc)
        W=R===Float32 ? UInt32 : UInt64
        @test reinterpret(W, x)==reinterpret(W, before)
    end
end

@testset "All 16 triangular storage symbols numerically through LBT" begin
    for (T, p) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
        J in (Int32, Int64),
        family in ("tbmv", "tbsv", "tpmv", "tpsv")

        a, dense=triangular_blas_fixture(T, family, 4, 1, 'L', 'U')
        x=T[777, 1, 2, 3, 4, 777]
        before=copy(x)
        backing=BLAS.lbt_find_backing_library(p*family*"_", J===Int32 ? :lp64 : :ilp64)
        @test backing!==nothing
        @test realpath(backing.libname)==realpath(Colmat.backend_library())
        direct_triangular_blas!(
            family,
            a,
            x,
            J;
            n=4,
            k=1,
            u=('L'),
            tr=('C'),
            d=('U'),
            inc=1,
            library=BLAS.libblastrampoline,
        )
        actual=if endswith(family, "sv")
            T[sum(conj(dense[j, i])*x[j + 1] for j in 1:4) for i in 1:4]
        else
            x[2:5]
        end
        expected=if endswith(family, "sv")
            before[2:5]
        else
            T[sum(conj(dense[j, i])*before[j + 1] for j in 1:4) for i in 1:4]
        end
        @test actual≈expected
        @test x[1]==x[end]==777
    end
end
