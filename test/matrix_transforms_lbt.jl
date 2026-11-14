using Test, LinearAlgebra, Libdl, Colmat
# This file also runs independently; the helpers' focused tests then run first.
for (binding, file) in (
    (:direct_storage_conversion!, "storage_conversions"),
    (:direct_matrix_copy!, "matrix_copy"),
    (:direct_laswp!, "matrix_permutation"),
    (:direct_lascl!, "matrix_scaling"),
    (:direct_precision!, "precision_conversion"),
    (:direct_lacgv!, "complex_auxiliary"),
    (:direct_matcopy!, "matrix_extensions"),
    (:direct_positive_equil!, "positive_equilibration"),
    (:direct_general_equil!, "general_equilibration"),
    (:direct_symmetric_equil!, "symmetric_equilibration"),
)
    isdefined(@__MODULE__, binding)||include(file*".jl")
end

@testset "All 112 matrix transform symbols numerically through LBT" begin
    tested=Set{Tuple{String,DataType}}()
    library=BLAS.libblastrampoline
    function check(prefix, family, J)
        name=prefix*family*"_"
        backing=BLAS.lbt_find_backing_library(name, J===Int32 ? :lp64 : :ilp64)
        @test backing!==nothing
        @test realpath(backing.libname)==realpath(Colmat.backend_library())
        push!(tested, (name, J))
    end
    for (T, p) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
        J in (Int32, Int64)

        R=typeof(real(zero(T)))
        for family in ("trttp", "tpttr", "trttf", "tfttr", "tpttf", "tfttp")
            check(p, family, J)
            fixture=conversion_fixture(T, 4, 'U', 'N', false)
            source=copy(getproperty(fixture, Symbol(family[1:2])))
            target=conversion_destination(T, family[4:5], 4)
            expected=copy(target)
            @test direct_storage_conversion!(
                family, source, target, J; n=4, uplo=('U'), library
            )==0
            @test direct_storage_conversion!(
                family,
                source,
                expected,
                Int64;
                n=4,
                uplo=('U'),
                library=conversion_reference(),
            )==0
            @test isequal(target, expected)
        end
        for family in ("lacpy", "laset")
            check(p, family, J)
            a=fill(T(2), 2, 2)
            b=fill(T(777), 2, 2)
            direct_matrix_copy!(
                family, a, b, J; uplo=('A'), m=2, n=2, alpha=T(2), beta=T(3), library
            )
            expected=family=="lacpy" ? a : T[3 2; 2 3]
            @test isequal(b, expected)
        end
        for family in ("laswp", "lapmr", "lapmt")
            check(p, family, J)
            a=reshape(T.(1:9), 3, 3)
            before=copy(a)
            if family=="laswp"
                direct_laswp!(a, J[3, 3]; n=3, k1=1, k2=2, incx=1, library)
                expected=before[[3, 1, 2], :]
            else
                direct_lapm!(family, a, J[3, 1, 2]; m=3, n=3, forward=J(1), library)
                expected=family=="lapmr" ? before[[3, 1, 2], :] : before[:, [3, 1, 2]]
            end
            @test isequal(a, expected)
        end
        check(p, "lascl", J)
        a=fill(T(2), 2, 3)
        @test direct_lascl!(a, J; kind=('G'), m=2, n=3, from=R(2), to=R(3), library)==0
        @test a==fill(T(3), 2, 3)
        for family in ("omatcopy", "imatcopy", "geadd")
            check(p, family, J)
            if family=="geadd"
                a=fill(T(2), 2, 2)
                b=fill(T(3), 2, 2)
                direct_geadd!(a, b, J; m=2, n=2, alpha=T(2), beta=T(3), library)
                @test b==fill(T(13), 2, 2)
            else
                a=fill(T(777), 8)
                a[2:5].=T(2)
                b=family=="imatcopy" ? copy(a) : fill(T(777), 8)
                direct_matcopy!(
                    family,
                    a,
                    b,
                    J;
                    order=('C'),
                    tr=('T'),
                    m=2,
                    n=2,
                    alpha=T(2),
                    la=2,
                    lb=2,
                    library,
                )
                expected=fill(T(777), 8)
                expected[2:5].=T(4)
                @test b==expected
            end
        end
        for family in ("poequ", "poequb", "pbequ", "ppequ")
            check(p, family, J)
            a=if family=="ppequ"
                T[777, 4, 0, 16, 777]
            elseif family=="pbequ"
                T[0 0; 4 16]
            else
                T[4 0; 0 16]
            end
            result=direct_positive_equil!(family, a, J; n=2, u=('U'), kd=1, library)
            @test result.info==0
            @test result.s==R[0.5, 0.25]
            @test result.cond==R(0.5)
            @test result.amax==R(16)
        end
        for family in ("geequ", "geequb", "gbequ", "gbequb")
            check(p, family, J)
            a=startswith(family, "gb") ? T[4 16; 0 0] : T[4 0; 0 16]
            result=direct_general_equil!(family, a, J; m=2, n=2, kl=0, ku=0, library)
            @test result.info==0
            @test result.r==R[0.25, 0.0625]
            @test result.c==R[1, 1]
            @test (result.rc, result.cc, result.amax)==(R(0.25), R(1), R(16))
        end
        for family in (T <: Complex ? ("syequb", "heequb") : ("syequb",))
            check(p, family, J)
            result=direct_symmetric_equil!(family, T[4 0; 0 16], J; n=2, u=('U'), library)
            @test result.info==0
            @test all(x->x>0&&isinteger(log2(x)), result.s)
            values=R[4, 16] .* result.s .^ 2
            @test maximum(values)/minimum(values)<=16
            @test result.amax==16
        end
        if T <: Complex
            check(p, "lacgv", J)
            x=fill(T(2, 1), 4)
            direct_lacgv!(x, J; n=2, inc=1, library)
            @test x==T[T(2, 1), T(2, -1), T(2, -1), T(2, 1)]
            check(p, "lacp2", J)
            b=fill(T(777), 2, 2)
            direct_lacp2!(fill(R(2), 2, 2), b, J; m=2, n=2, u=('A'), library)
            @test b==fill(T(2), 2, 2)
            for family in ("lacrm", "larcm")
                check(p, family, J)
                a=family=="lacrm" ? fill(T(2, 1), 2, 2) : fill(R(3), 2, 2)
                b=family=="lacrm" ? fill(R(3), 2, 2) : fill(T(2, 1), 2, 2)
                c=fill(T(NaN), 2, 2)
                direct_mixed!(T, J, family, a, b, c; m=2, n=2, library)
                @test c==fill(T(12, 6), 2, 2)
            end
        end
    end
    for (family, S, D) in (
            ("slag2d", Float32, Float64),
            ("dlag2s", Float64, Float32),
            ("clag2z", ComplexF32, ComplexF64),
            ("zlag2c", ComplexF64, ComplexF32),
            ("dlat2s", Float64, Float32),
            ("zlat2c", ComplexF64, ComplexF32),
        ),
        J in (Int32, Int64)

        check(family[1:1], family[2:end], J)
        a=fill(S(3), 2, 2)
        b=fill(D(777), 2, 2)
        expected=copy(b)
        for j in 1:2, i in 1:2
            if !occursin("lat", family)||i<=j
                expected[i, j]=D(3)
            end
        end
        @test direct_precision!(
            Val(Symbol(family)), a, b, J; m=2, n=2, uplo=('U'), library
        )==0
        @test b==expected
    end
    @test length(tested)==224
    @test length(Set(first(x) for x in tested))==112
end
