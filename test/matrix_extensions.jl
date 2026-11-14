using Test, LinearAlgebra, Libdl, Colmat

for (T, p) in ((Float32, "s"), (Float64, "d"), (ComplexF32, "c"), (ComplexF64, "z")),
    J in (Int32, Int64)

    suffix=J===Int64 ? "64_" : ""
    @eval function direct_matcopy!(
        family,
        a::Vector{$T},
        b::Vector{$T},
        ::Type{$J};
        order,
        tr,
        m,
        n,
        alpha,
        la,
        lb,
        library=Colmat.backend_library(),
    )
        f=Libdl.dlsym(Libdl.dlopen(library), Symbol($p*family*"_"*$suffix))
        if family=="omatcopy"
            ccall(
                f,
                Cvoid,
                (
                    Ref{UInt8},
                    Ref{UInt8},
                    Ref{$J},
                    Ref{$J},
                    Ref{$T},
                    Ptr{$T},
                    Ref{$J},
                    Ptr{$T},
                    Ref{$J},
                ),
                order,
                tr,
                m,
                n,
                alpha,
                pointer(a, 2),
                la,
                pointer(b, 2),
                lb,
            )
        else
            ccall(
                f,
                Cvoid,
                (
                    Ref{UInt8},
                    Ref{UInt8},
                    Ref{$J},
                    Ref{$J},
                    Ref{$T},
                    Ptr{$T},
                    Ref{$J},
                    Ref{$J},
                ),
                order,
                tr,
                m,
                n,
                alpha,
                pointer(b, 2),
                la,
                lb,
            )
        end
    end
    @eval function direct_geadd!(
        a::Matrix{$T},
        b::Matrix{$T},
        ::Type{$J};
        m,
        n,
        alpha,
        beta,
        library=Colmat.backend_library(),
    )
        f=Libdl.dlsym(Libdl.dlopen(library), Symbol($p*"geadd_"*$suffix))
        ccall(
            f,
            Cvoid,
            (Ref{$J}, Ref{$J}, Ref{$T}, Ptr{$T}, Ref{$J}, Ref{$T}, Ptr{$T}, Ref{$J}),
            m,
            n,
            alpha,
            a,
            size(a, 1),
            beta,
            b,
            size(b, 1),
        )
    end
end
function extension_reference()
    return only(
        filter(
            l->l.interface==:ilp64 && occursin("openblas", lowercase(basename(l.libname))),
            BLAS.get_config().loaded_libs,
        ),
    ).libname
end
extension_offset(order, i, j, ld) = order=='C' ? i+(j-1)*ld : j+(i-1)*ld
extension_product(a::Real, b::Real) = a*b
function extension_product(a::Complex, b::Complex)
    return typeof(a)(real(a)*real(b)-imag(a)*imag(b), real(a)*imag(b)+imag(a)*real(b))
end

@testset "OMATCOPY/IMATCOPY layouts transpose conjugation and overlap" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        m in 0:5,
        n in 0:5,
        order in ('C', 'R'),
        tr in ('N', 'T', 'R', 'C'),
        family in ("omatcopy", "imatcopy"),
        layout in 1:3,
        scenario in 1:2

        transpose=tr in ('T', 'C')
        outm, outn=transpose ? (n, m) : (m, n)
        minor, major=order=='C' ? (m, n) : (n, m)
        outminor, outmajor=order=='C' ? (outm, outn) : (outn, outm)
        la=if layout==1
            max(m, n)+2
        elseif layout==2
            minor+2
        else
            max(1, minor)
        end
        lb=if layout==1
            la
        elseif layout==2
            outminor+3
        else
            max(1, outminor)
        end
        count=max(la*major, lb*outmajor)+3
        guard=T(777)
        a=fill(guard, count)
        R=typeof(real(zero(T)))
        values=R[0, -0.0, Inf, -Inf, NaN, 1, -1]
        for j in 1:n, i in 1:m
            value=if scenario==1
                (T <: Complex ? T(i+2j, i-j) : T(i+2j))
            else
                (
                    if T <: Complex
                        T(values[mod1(i+j, 7)], values[mod1(i+3j, 7)])
                    else
                        T(values[mod1(i+j, 7)])
                    end
                )
            end
            a[1 + extension_offset(order, i, j, la)]=value
        end
        before=copy(a)
        for alpha in (T(0), T(1), T(-2), T <: Complex ? T(0.5, -1) : T(0.5))
            b=family=="imatcopy" ? copy(before) : fill(guard, count)
            expected=copy(b)
            for j in 1:n, i in 1:m
                value=before[1 + extension_offset(order, i, j, la)]
                if T <: Complex && tr in ('R', 'C')
                    value=conj(value)
                end
                scaled=if T <: Real && alpha==0
                    zero(T)
                elseif T <: Real && alpha==1
                    value
                else
                    extension_product(alpha, value)
                end
                oi, oj=transpose ? (j, i) : (i, j)
                expected[1 + extension_offset(order, oi, oj, lb)]=scaled
            end
            direct_matcopy!(family, a, b, J; order, tr, m, n, alpha, la, lb)
            @test isequal(b, expected)
            @test isequal(a, before)
            if J===Int64 && scenario==1
                external=family=="imatcopy" ? copy(before) : fill(guard, count)
                direct_matcopy!(
                    family,
                    a,
                    external,
                    J;
                    order,
                    tr,
                    m,
                    n,
                    alpha,
                    la,
                    lb,
                    library=extension_reference(),
                )
                @test isequal(b, external)
            end
        end
    end
end

@testset "GEADD coefficients and unused operands" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        J in (Int32, Int64),
        m in 0:5,
        n in 0:5,
        alpha in (T(0), T(1), T(-2), T <: Complex ? T(0.5, -1) : T(0.5)),
        beta in (T(0), T(1), T(-0.5), T <: Complex ? T(-0.25, 0.5) : T(-0.25))

        a=fill(T(NaN), m+2, n+1)
        b=fill(T(777), m+3, n+1)
        for j in 1:n, i in 1:m
            a[i, j]=if alpha==0
                T(NaN)
            elseif T <: Complex
                T(i+j, i-j)
            else
                T(i+j)
            end
            b[i, j]=if beta==0
                T(NaN)
            elseif T <: Complex
                T(i-j, i+j)
            else
                T(i-j)
            end
        end
        before=copy(a)
        original=copy(b)
        expected=copy(b)
        for j in 1:n, i in 1:m
            expected[i, j]=if alpha==0
                (
                    if beta==0
                        zero(T)
                    elseif beta==1
                        b[i, j]
                    else
                        extension_product(beta, b[i, j])
                    end
                )
            elseif beta==0
                extension_product(alpha, a[i, j])
            else
                extension_product(alpha, a[i, j])+extension_product(beta, b[i, j])
            end
        end
        direct_geadd!(a, b, J; m, n, alpha, beta)
        @test isequal(b, expected)
        @test isequal(a, before)
        if J===Int64 && alpha!=0 && beta!=0
            direct_geadd!(a, original, J; m, n, alpha, beta, library=extension_reference())
            @test isequal(b, original)
        end
    end
end
