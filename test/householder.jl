using Colmat, Test, LinearAlgebra, SparseArrays, Random

@testset "Native elementary Householder construction and ownership" begin
    rng=MersenneTwister(561)
    for T in (Float32, Float64, ComplexF32, ComplexF64), n in (0, 1, 2, 7, 31)
        for mode in (:random, :zero, :zero_tail)
            x=mode===:zero ? zeros(T, n) : randn(rng, T, n)
            mode===:zero_tail && n>1 && fill!(view(x, 2:n), zero(T))
            source=FaerVector(x)
            h=householder(source)
            m=Matrix(h)
            basis=Vector(h.v)
            @test h isa FaerHouseholder{T}
            @test size(h)==(n, n)
            @test size(h, 3)==1
            @test isopen(h)
            @test ishermitian(h)
            @test eltype(h.v)===T
            @test h.tau isa real(T)
            @test h.input_norm ≈ norm(x)
            @test abs(h.beta) ≈ norm(x)
            @test m ≈ m'
            @test m*m ≈ Matrix{T}(I, n, n)
            @test Matrix(h') ≈ m'
            @test Matrix(transpose(h)) ≈ transpose(m)
            @test Matrix(conj(h)) ≈ conj(m)
            @test Matrix(inv(h)) ≈ m'
            @test Matrix(h*h) ≈ Matrix{T}(I, n, n)
            @test all(isfinite, m)
            target=zeros(T, n)
            n>0 && (target[1]=h.beta)
            @test h*x ≈ target atol=100eps(real(T))*max(norm(x), one(real(T)))
            @test h\x ≈ h*x
            @test h*source isa FaerVector
            @test Vector(h*source) ≈ target atol=100eps(real(T))*max(norm(x), one(real(T)))
            if n>0
                @test basis[1]==one(T)
                @test m[1, 1]≈h[1, 1]
                @test m[n, n]≈h[n, n]
                formula=Matrix{T}(I, n, n)-basis*basis'/h.tau
                @test m ≈ formula
                if !isinf(h.tau)
                    @test (x[1]-h.beta)*h.head_with_beta_inv ≈ one(T)
                end
                source[1]+=1
                exported=h.v
                exported[1]+=1
                @test Matrix(h)==m
            else
                @test isempty(basis)
            end
            cloned=copy(h)
            deep=deepcopy(h)
            adjointcopy=h'
            close(source)
            close(h)
            @test !isopen(h)
            @test Matrix(cloned)==m && Matrix(deep)==m && Matrix(adjointcopy)==m
            @test_throws ArgumentError h.v
            @test_throws ArgumentError h.tau
            @test_throws ArgumentError Matrix(h)
            @test_throws ArgumentError h*x
        end
    end
end
@testset "Householder left/right native and strided application" begin
    rng=MersenneTwister(278)
    for T in (Float32, Float64, ComplexF32, ComplexF64), n in (0, 1, 6, 17)
        h=householder(randn(rng, T, n))
        m=Matrix(h)
        a=randn(rng, T, n, 4)
        b=randn(rng, T, 5, n)
        for src in (a, FaerMatrix(a))
            result=h*src
            @test result isa FaerMatrix{T}
            @test Matrix(result) ≈ m*a
            @test Matrix(src)==a
        end
        for src in (b, FaerMatrix(b))
            @test Matrix(src*h) ≈ b*m
        end
        native=FaerMatrix(a)
        @test lmul!(h, native)===native
        @test Matrix(native) ≈ m*a
        @test lmul!(h, native)===native
        @test Matrix(native) ≈ a
        native=FaerMatrix(b)
        @test rmul!(native, h)===native
        @test Matrix(native) ≈ b*m
        @test rmul!(native, h)===native
        @test Matrix(native) ≈ b
        dest=copy(a)
        @test lmul!(h, dest)===dest
        @test dest ≈ m*a
        dest=copy(b)
        @test rmul!(dest, h)===dest
        @test dest ≈ b*m
        v=randn(rng, T, n)
        dest=copy(v)
        @test lmul!(h, dest)===dest
        @test dest ≈ m*v
        nativevector=FaerVector(v)
        @test lmul!(h, nativevector)===nativevector
        @test Vector(nativevector) ≈ m*v
        row=FaerRow(v)
        @test row*h isa FaerRow{T}
        @test Matrix(row*h) ≈ reshape(v, 1, :)*m
        @test rmul!(row, h)===row
        @test Matrix(row) ≈ reshape(v, 1, :)*m
    end
end
@testset "Householder promotion, views and failure preservation" begin
    rng=MersenneTwister(63)
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        S in (Float32, Float64, ComplexF32, ComplexF64)

        x=randn(rng, T, 4)
        h=householder(x)
        a=randn(rng, S, 4, 5)
        P=promote_type(T, S)
        basis=P.(Vector(h.v))
        reference=Matrix{P}(I, 4, 4)-basis*basis'/h.tau
        @test eltype(h*FaerMatrix(a))===P
        @test Matrix(h*FaerMatrix(a)) ≈ reference*P.(a)
        @test eltype(h*a[:, 1])===P
        @test h*a[:, 1] ≈ reference*P.(a[:, 1])
        @test Matrix(FaerMatrix(a')*h) ≈ P.(a')*reference
    end
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        parent=FaerMatrix(randn(rng, T, 8, 10))
        v=view(parent, 8:-2:2, 10:-2:2)
        h=householder(randn(rng, T, 4))
        expected=Matrix(h)*Matrix(v)
        @test lmul!(h, v)===v
        @test Matrix(v) ≈ expected
        right=householder(randn(rng, T, 5))
        expected=Matrix(v)*Matrix(right)
        @test rmul!(v, right)===v
        @test Matrix(v) ≈ expected
        vector=FaerVector(randn(rng, T, 8))
        selected=view(vector, 8:-2:2)
        reflected=householder(selected)
        expected=Matrix(reflected)*Vector(selected)
        @test lmul!(reflected, selected)===selected
        @test Vector(selected) ≈ expected
        resize!(vector, 9)
        @test_throws InvalidStateException lmul!(reflected, selected)
        @test_throws InvalidStateException householder(selected)
        before=Matrix(parent)
        @test_throws DimensionMismatch lmul!(h, parent)
        @test Matrix(parent)==before
        close(v)
        @test_throws ArgumentError h*v
        @test_throws BoundsError h[0, 1]
        @test_throws ArgumentError householder(T[Inf, 1])
        @test_throws ArgumentError householder(T[NaN, 1])
    end
    h=householder(ComplexF64[1+im, 2, 3])
    dest=FaerMatrix(ones(3, 2))
    before=Matrix(dest)
    @test_throws InexactError lmul!(h, dest)
    @test Matrix(dest)==before
    integers=ones(Int, 3, 2)
    @test_throws ArgumentError lmul!(householder([1.0, 2, 3]), integers)
    @test integers==ones(Int, 3, 2)
    @test isempty(
        detect_ambiguities(Colmat, Base, LinearAlgebra, SparseArrays; recursive = true),
    )
end
@testset "Scaled Householder generation and linear storage application" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        R=real(T)
        for scale in (nextfloat(zero(R)), floatmin(R), R(1), floatmax(R)/R(100))
            x=T[scale, scale, -scale, scale]
            h=householder(x)
            m=Matrix(h)
            @test all(isfinite, m)
            @test m*m ≈ Matrix{T}(I, 4, 4)
            normalized=T[1, 1, -1, 1]
            y=h*normalized
            @test norm(y[2:end])<100eps(R)
            @test abs(y[1])≈2
        end
    end
    n=100_000
    h=householder(ones(n))
    result=h*FaerVector(ones(n))
    @test result isa FaerVector
    @test abs(result[1])≈sqrt(n)
    @test norm(Vector(result)[2:end])<1e-10
    @test length(h.v)==n
end

@testset "Householder mul! staging and alpha beta" begin
    rng=MersenneTwister(86)
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        h=householder(randn(rng, T, 4))
        m=Matrix(h)
        a=randn(rng, T, 4, 4)
        d=randn(rng, T, 4, 4)
        for native in (true, false), (alpha, beta) in ((1, 0), (2, 3), (0, 2), (0, 0))
            source=native ? FaerMatrix(a) : copy(a)
            dest=native ? FaerMatrix(d) : copy(d)
            @test mul!(dest, h, source, alpha, beta)===dest
            @test Matrix(dest) ≈ alpha*m*a+beta*d
            dest=native ? FaerMatrix(d) : copy(d)
            @test mul!(dest, source, h, alpha, beta)===dest
            @test Matrix(dest) ≈ alpha*a*m+beta*d
            v=native ? FaerVector(a[:, 1]) : copy(a[:, 1])
            target=native ? FaerVector(d[:, 1]) : copy(d[:, 1])
            @test mul!(target, h, v, alpha, beta)===target
            @test Vector(target) ≈ alpha*m*a[:, 1]+beta*d[:, 1]
        end
        source=FaerMatrix(a)
        @test mul!(source, h, source)===source
        @test Matrix(source) ≈ m*a
        source=FaerMatrix(a)
        @test mul!(source, source, h)===source
        @test Matrix(source) ≈ a*m
        v=FaerVector(a[:, 1])
        @test mul!(v, h, v)===v
        @test Vector(v) ≈ m*a[:, 1]
        @test_throws DimensionMismatch mul!(FaerMatrix(zeros(T, 4, 3)), h, FaerMatrix(a))
        @test_throws DimensionMismatch h*householder(ones(T, 3))
    end
    h=householder(ComplexF64[1+im, 2, 3])
    dest=ones(3, 2)
    @test_throws InexactError mul!(dest, h, ones(3, 2))
    @test dest==ones(3, 2)
    @test isempty(
        detect_ambiguities(Colmat, Base, LinearAlgebra, SparseArrays; recursive = true),
    )
end
