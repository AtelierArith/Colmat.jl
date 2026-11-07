using Test, LinearAlgebra, Random, SparseArrays, Colmat

const _DenseBroadcastTypes=(Float32, Float64, ComplexF32, ComplexF64)

# Julia reference for a native operand.
_ref(a::FaerMatrix) = Matrix(a)
_ref(a::FaerVector) = Vector(a)
_ref(a::FaerRow) = reshape(Vector(a), 1, :)
_ref(a::Colmat.FaerView) = Array(a)
_ref(a) = a

function _same_native(x, ref)
    expected=ref isa Vector ? FaerVector : FaerMatrix
    return x isa expected && eltype(x)===eltype(ref) && size(x)==size(ref) && Array(x)≈ref
end

@testset "Native dense broadcast returns native owners" begin
    rng=MersenneTwister(9898)
    for T in _DenseBroadcastTypes, (m, n) in ((4, 3), (0, 3), (3, 0), (1, 1))
        A=FaerMatrix{T}(randn(rng, T, m, n))
        B=FaerMatrix{T}(randn(rng, T, m, n))
        v=FaerVector{T}(randn(rng, T, m))
        r=FaerRow{T}(randn(rng, T, n))
        a, b, x, y=_ref(A), _ref(B), _ref(v), _ref(r)
        α=T<:Complex ? T(1.5-0.5im) : T(1.5)
        cases=(
            (A .+ B, a .+ b),
            (A .- B, a .- b),
            (α .* A, α .* a),
            (A .* B, a .* b),
            (abs.(A), abs.(a)),
            (conj.(A) .+ 2 .* B ./ 3, conj.(a) .+ 2 .* b ./ 3),
            (A .+ v, a .+ x),
            (A .* r, a .* y),
            (v .+ r, x .+ y),
            (A .* Ref(α), a .* α),
            (A .+ fill(α), a .+ α),
            (A .* A, a .* a),
            (A' .+ 1, a' .+ 1),
            (transpose(B) .* α, transpose(b) .* α),
            (v .* 2, x .* 2),
            (map(sin, A), map(sin, a)),
            (map(+, A, B), map(+, a, b)),
            (map(*, A, b), map(*, a, b)),
        )
        for (got, ref) in cases
            @test _same_native(got, ref)
        end
        # Rows stay rows when every native operand is a row.
        @test (r .+ 1) isa FaerRow && Array(r .+ 1)≈y .+ 1
        @test (r .* r) isa FaerRow
        # Non-native eltypes and other dimensionalities follow Julia's result.
        @test (real.(A) .> 0) isa BitMatrix && (real.(A) .> 0)==(real.(a) .> 0)
        @test round.(Int, real.(A)) isa Matrix{Int}
        @test (A .+ ones(m, n, 2)) isa Array{promote_type(T, Float64),3}
        @test (A .+ ones(m, n, 2))≈(a .+ ones(m, n, 2))
        m>0 && @test (v .+ (1, 2, 3, 4)[1:m]) isa FaerVector{promote_type(T, Int)}
        m>1 && @test_throws DimensionMismatch A .+ FaerMatrix{T}(randn(rng, T, m+1, n))
        m==0 && @test (v .+ ()) isa Vector{Union{}}
    end
    # Integer and mixed promotions follow promote_type.
    A=FaerMatrix{Float32}(rand(Float32, 2, 2))
    @test eltype(A .+ 1) === Float32 && eltype(A .+ 1.0) === Float64
    @test eltype(A .* im) === ComplexF32
    @test eltype(A .+ FaerMatrix{ComplexF64}(zeros(ComplexF64, 2, 2))) === ComplexF64
end

@testset "Native dense broadcast over views and borrowed arrays" begin
    for T in _DenseBroadcastTypes
        data=T[i+10j for i = 1:6, j = 1:5]
        A=FaerMatrix{T}(data)
        V=view(A, 2:2:6, 5:-2:1)
        @test _same_native(V .+ 1, data[2:2:6, 5:-2:1] .+ 1)
        row=view(A, 3, :)
        @test _same_native(row .* 2, data[3, :] .* 2)
        col=view(A, :, 2)
        @test _same_native(col .+ A, data[:, 2] .+ data)
        J=randn(T, 6, 5)
        W=faerview(J)
        @test _same_native(A .+ W, data .+ J)
        # Borrowed-only broadcasts stay Julia arrays.
        @test (W .+ 1) isa Matrix{T}
        # Julia destinations and borrowed destinations, including aliasing.
        M=zeros(T, 6, 5)
        M .= A .+ 1
        @test M == data .+ 1
        J2=copy(J)
        W2=faerview(J2)
        W2 .= A .+ W2
        @test J2≈data .+ J
        J3=collect(T, 1:6)
        W3=faerview(J3)
        view(W3, 2:6) .= view(W3, 1:5) .+ FaerVector{T}(ones(T, 5))
        @test J3==T[1, 2, 3, 4, 5, 6]
    end
end

@testset "In-place native dense broadcast" begin
    rng=MersenneTwister(7)
    for T in _DenseBroadcastTypes
        a=randn(rng, T, 5, 4)
        b=randn(rng, T, 5, 4)
        A=FaerMatrix{T}(a)
        B=FaerMatrix{T}(b)
        A .+= B
        @test Matrix(A)≈a .+ b
        A .= a
        @test Matrix(A)==a
        A .= A .+ A .* 2
        @test Matrix(A)≈3a
        A .= 0
        @test iszero(Matrix(A))
        A .= B
        @test Matrix(A)==b
        A .= view(B, :, 1)
        @test Matrix(A)==repeat(b[:, 1], 1, 4)
        # Overlapping views of the same owner read the old values.
        A .= a
        view(A, 2:5, :) .= view(A, 1:4, :) .+ 0
        expected=copy(a)
        expected[2:5, :]=a[1:4, :]
        @test Matrix(A)==expected
        v=FaerVector{T}(T[1, 2, 3])
        v .= v .* v
        @test Vector(v)==T[1, 4, 9]
        r=FaerRow{T}(T[1, 2])
        r .+= FaerRow{T}(T[3, 4])
        @test Vector(r)==T[4, 6]
        map!(x -> 2x, A, B)
        @test Matrix(A)==2b
        map!(+, A, A, B)
        @test Matrix(A)≈3b
        @test_throws DimensionMismatch A .= FaerMatrix{T}(zeros(T, 2, 2))
        if !(T<:Complex)
            # A failed conversion leaves the destination unchanged.
            before=Matrix(A)
            @test_throws InexactError A .= B .+ im
            @test Matrix(A)==before
            @test_throws InexactError map!(x -> x+im, A, B)
            @test Matrix(A)==before
        end
    end
end

@testset "Closed native operands" begin
    A=FaerMatrix(rand(2, 2))
    V=view(A, :, 1)
    B=copy(A)
    close(A)
    @test_throws ArgumentError A .+ 1
    @test_throws ArgumentError V .+ 1
    @test_throws ArgumentError B .= A
    @test_throws ArgumentError hcat(A, B)
    @test isopen(B)
end

@testset "Native dense concatenation" begin
    for T in _DenseBroadcastTypes
        a=randn(T, 3, 2)
        b=randn(T, 3, 2)
        A=FaerMatrix{T}(a)
        B=FaerMatrix{T}(b)
        v=FaerVector{T}(T[1, 2, 3])
        r=FaerRow{T}(T[4, 5])
        @test _same_native(hcat(A, B), hcat(a, b))
        @test _same_native(vcat(A, B), vcat(a, b))
        @test _same_native([A B; B A], [a b; b a])
        @test _same_native([A b], [a b])
        @test _same_native(hcat(A, v), hcat(a, Vector(v)))
        @test _same_native(vcat(A, r), vcat(a, Vector(r)'))
        @test _same_native(vcat(v, v), vcat(Vector(v), Vector(v)))
        @test _same_native(hcat(v, v), hcat(Vector(v), Vector(v)))
        @test _same_native(vcat(view(A, :, 1), v), vcat(a[:, 1], Vector(v)))
        @test hcat(r, r) isa FaerRow{T} && Vector(hcat(r, r))==T[4, 5, 4, 5]
        @test _same_native(vcat(r, r), T[4 5; 4 5])
        @test _same_native(cat(A, B; dims = 2), cat(a, b; dims = 2))
        @test cat(A, B; dims = 3) isa Array{T,3}
        @test cat(A, B; dims = 3)==cat(a, b; dims = 3)
        @test _same_native(hcat(A, zeros(Float64, 3, 0)), hcat(a, zeros(Float64, 3, 0)))
        @test_throws DimensionMismatch hcat(A, FaerMatrix{T}(zeros(T, 2, 2)))
    end
end

@testset "Mixed native sparse and Julia styles" begin
    A=FaerMatrix(rand(3, 3))
    S=FaerSparseMatrix(sparse([1.0 0 2; 0 0 0; 0 3 0]))
    @test (S .+ A) isa FaerSparseMatrix
    @test SparseMatrixCSC(S .+ A)≈SparseMatrixCSC(S) .+ Matrix(A)
    @test (S .* FaerVector(ones(3))) isa FaerSparseMatrix
    js=sparse([1.0 0 0; 0 2 0; 0 0 0])
    @test (js .* A) isa SparseMatrixCSC
    @test (js .* A)≈js .* Matrix(A)
    @test _same_native(A .+ Diagonal([1.0, 2, 3]), Matrix(A) .+ Diagonal([1.0, 2, 3]))
    @test _same_native(A .+ UpperTriangular(ones(3, 3)), Matrix(A) .+ UpperTriangular(ones(3, 3)))
end

@testset "Native dense broadcast allocations" begin
    n=200
    A=FaerMatrix(rand(n, n))
    B=FaerMatrix(rand(n, n))
    v=FaerVector(rand(n))
    bytes=n*n*sizeof(Float64)
    A .+ B
    A .+= B
    hcat(A, B)
    map(sin, A)
    sin.(A)
    A .+ v
    # The element-wise path allocated about 100 MB for each of these.
    @test @allocated(A .+ B) < 3bytes + 65536
    @test @allocated(sin.(A)) < 2bytes + 65536
    @test @allocated(A .+ v) < 2bytes + 65536
    @test @allocated(A .+= B) < 3bytes + 65536
    @test @allocated(map(sin, A)) < 2bytes + 65536
    @test @allocated(hcat(A, B)) < 4bytes + 65536
end

@testset "Native whole-matrix kernels match Julia's scalar rules" begin
    for T in _DenseBroadcastTypes
        R=real(T)
        a=T<:Complex ? T[Inf+1im 2; NaN 3-Inf*im] : T[Inf 2; NaN -0.0]
        b=T<:Complex ? T[1 2im; 3 4] : T[1 2; 3 4]
        # Constructors reject non-finite input; owners hold it after copyto!.
        A=copyto!(FaerMatrix{T}(undef, 2, 2), a)
        B=FaerMatrix{T}(b)
        for (got, ref) in (
            (A .+ B, a .+ b),
            (A .- B, a .- b),
            (2 .* A, 2 .* a),
            (A .* R(1.5), a .* R(1.5)),
            (false .* A, false .* a),
            (A .* (T<:Complex ? T(0, 1) : 2.0), a .* (T<:Complex ? T(0, 1) : 2.0)),
            ((T<:Complex ? 2.0 : 1.0im) .* A, (T<:Complex ? 2.0 : 1.0im) .* a),
            (big(2) .* A, big(2) .* a),
        )
            @test typeof(Array(got))===typeof(ref)
            @test isequal(Array(got), ref)
        end
        @test (big(2) .* A) isa Matrix
        C=copyto!(FaerMatrix{T}(undef, 2, 2), a)
        C .+= B
        @test isequal(Matrix(C), a .+ b)
        C .*= 2
        @test isequal(Matrix(C), 2 .* (a .+ b))
        # Non-finite broadcast results stay native.
        Z=FaerMatrix{T}(zeros(T, 2, 2))
        @test isequal(Array(B ./ Z), b ./ zeros(T, 2, 2)) && (B ./ Z) isa FaerMatrix{T}
        @test isequal(Array(FaerVector{T}(T[0, 1]) ./ 0), T[0, 1] ./ 0)
        @test isequal(Vector(FaerRow{T}(T[0, 1]) ./ 0), T[0, 1] ./ 0)
        @test isequal(Array(hcat(A, B)), hcat(a, b))
    end
    n=200
    A=FaerMatrix(rand(n, n))
    B=FaerMatrix(rand(n, n))
    A .+ B
    2 .* A
    # These reuse the native kernels and allocate no Julia snapshots.
    @test @allocated(A .+ B) < 16384
    @test @allocated(2 .* A) < 16384
    A .+= B
    @test @allocated(A .+= B) < 16384
end

@testset "In-place broadcast keeps non-finite values" begin
    for T in _DenseBroadcastTypes
        a=T[NaN 1; Inf 2]
        A=copyto!(FaerMatrix{T}(undef, 2, 2), a)
        v=FaerVector{T}(undef, 2)
        v[1]=NaN
        r=FaerRow{T}(undef, 2)
        r[2]=Inf
        for dest in (FaerMatrix{T}(undef, 2, 2), FaerVector{T}(undef, 2), FaerRow{T}(undef, 2))
            dest .= NaN
            @test all(isnan, Array(dest))
        end
        M=FaerMatrix{T}(undef, 2, 2)
        M .= A
        @test isequal(Matrix(M), a)
        W=FaerMatrix{ComplexF64}(undef, 2, 2)
        W .= A
        @test isequal(Matrix(W), ComplexF64.(a))
        w=FaerVector{T}(undef, 2)
        w .= v
        @test isequal(Vector(w), Vector(v))
        q=FaerRow{T}(undef, 2)
        q .= r
        @test isequal(Vector(q), Vector(r))
        c=FaerVector{T}(undef, 2)
        c .= view(A, :, 1)
        @test isequal(Vector(c), a[:, 1])
        V=view(FaerMatrix{T}(undef, 2, 2), :, :)
        V .= A .+ A
        @test isequal(Array(V), a .+ a)
        @test isequal(Array(view(A, :, 1) .+ 0), a[:, 1] .+ 0)
        @test isequal(Array(view(A, 1:2, 1:2) .* 1), a .* 1)
    end
end
