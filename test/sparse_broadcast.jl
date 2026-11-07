using Test, LinearAlgebra, Random, SparseArrays, Colmat

const _BroadcastTypes=(Float32, Float64, ComplexF32, ComplexF64)

function _broadcast_source(rng, ::Type{T}, m, n) where {T}
    s=sprand(rng, T, m, n, 0.4)
    # A stored zero: SparseArrays drops every zero that f produces.
    if m>0&&n>0
        s[1, 1]=zero(T)
        s[m, n]=one(T)
    end
    return s
end

function _same_sparse(a::FaerSparseMatrix, ref::SparseMatrixCSC, format)
    return eltype(a)===eltype(ref) &&
           a.format===format &&
           size(a)==size(ref) &&
           nnz(a)==nnz(ref) &&
           isequal(SparseMatrixCSC(a), ref)
end

@testset "Native single-sparse broadcast and map" begin
    rng=MersenneTwister(5050)
    for T in _BroadcastTypes, fmt in (:csc, :csr), (m, n) in ((6, 5), (0, 3), (1, 1))
        ref=_broadcast_source(rng, T, m, n)
        a=FaerSparseMatrix(ref; format = fmt)
        α=T<:Complex ? T(1.5-0.5im) : T(1.5)
        cases=(
            (abs.(a), abs.(ref)),
            (conj.(a), conj.(ref)),
            (real.(a), real.(ref)),
            (a .* α, ref .* α),
            (2 .* a, 2 .* ref),
            (a ./ 4, ref ./ 4),
            (a .* 0, ref .* 0),
            (a .^ 2, ref .^ 2),
            (abs2.(a), abs2.(ref)),
            (2 .* abs.(a) .+ 0, 2 .* abs.(ref) .+ 0),
            (a .* Ref(α), ref .* Ref(α)),
            (a .* a, ref .* ref),
            (Float32.(abs.(a)), Float32.(abs.(ref))),
            (a .* (1+2im), ref .* (1+2im)),
            (map(abs, a), map(abs, ref)),
            (map(x -> 3x, a), map(x -> 3x, ref)),
            (broadcast(-, a), broadcast(-, ref)),
        )
        for (got, want) in cases
            @test got isa FaerSparseMatrix
            @test _same_sparse(got, want, fmt)
        end
        # The source is never modified and its live value view stays valid.
        @test isequal(SparseMatrixCSC(a), ref)
    end
end

@testset "Sparse broadcast fallbacks follow SparseArrays" begin
    rng=MersenneTwister(5051)
    for T in _BroadcastTypes, fmt in (:csc, :csr)
        ref=_broadcast_source(rng, T, 5, 4)
        ref2=_broadcast_source(rng, T, 5, 4)
        a=FaerSparseMatrix(ref; format = fmt)
        b=FaerSparseMatrix(ref2; format = :csc)
        row=rand(rng, T, 1, 4)
        dense=rand(rng, T, 5, 4)
        cases=(
            (a .+ 1, ref .+ 1),
            (cos.(a), cos.(ref)),
            (a .* T(Inf), ref .* T(Inf)),
            (a .+ b, ref .+ ref2),
            (a .* b, ref .* ref2),
            (a .* row, ref .* row),
            (a .+ dense, ref .+ dense),
            (a .* ref2, ref .* ref2),
            (ref2 .- a, ref2 .- ref),
            (a .* FaerMatrix(dense), ref .* dense),
            (map(+, a, b), map(+, ref, ref2)),
        )
        for (got, want) in cases
            @test got isa FaerSparseMatrix
            @test _same_sparse(got, want, fmt)
        end
        # Non-native result scalars stay Julia sparse matrices.
        mask=a .!= 0
        @test mask isa SparseMatrixCSC{Bool,Int}
        @test mask==(ref .!= 0)
        wide=a .* big(2)
        @test wide isa SparseMatrixCSC
        @test eltype(wide)==eltype(ref .* big(2))
        @test wide==ref .* big(2)
        @test_throws DimensionMismatch map(+, a, FaerSparseMatrix(sprand(rng, T, 4, 4, 0.5)))
    end
end

@testset "In-place sparse broadcast and map!" begin
    rng=MersenneTwister(5052)
    for T in _BroadcastTypes, fmt in (:csc, :csr)
        ref=_broadcast_source(rng, T, 6, 5)
        dropzeros!(ref)
        α=T<:Complex ? T(0.5+0.25im) : T(0.5)

        # Unchanged structure: values are written in bulk, live views remain valid.
        a=FaerSparseMatrix(ref; format = fmt)
        vals=nonzeros(a)
        r=copy(ref)
        @test (a .*= α)===a
        r .*= α
        @test _same_sparse(a, r, fmt)
        @test isopen(vals)
        @test Vector(vals)==Vector(nonzeros(a))
        a .= abs.(a) .* 2
        r .= abs.(r) .* 2
        @test _same_sparse(a, r, fmt)
        @test isopen(vals)
        @test map!(x -> x/2, a, a)===a
        map!(x -> x/2, r, r)
        @test _same_sparse(a, r, fmt)
        @test isopen(vals)

        # Produced zeros are dropped and invalidate earlier value views.
        a .*= 0
        @test nnz(a)==0 && !isopen(vals)
        @test _same_sparse(a, ref .* 0, fmt)

        # Destination differs from the sparse source.
        a=FaerSparseMatrix(ref; format = fmt)
        d=FaerSparseMatrix(ref; format = fmt)
        vals=nonzeros(d)
        d .= conj.(a) .* 3
        @test _same_sparse(d, conj.(ref) .* 3, fmt)
        @test isopen(vals)
        e=FaerSparseMatrix(spzeros(T, 6, 5); format = fmt)
        map!(abs, e, a)
        @test _same_sparse(e, SparseMatrixCSC{T,Int}(abs.(ref)), fmt)

        # f(0) != 0 fills the pattern like SparseArrays.
        f=FaerSparseMatrix(ref; format = fmt)
        f .+= 1
        @test _same_sparse(f, ref .+ 1, fmt)
        # Scalar `.=` is Base's fill!: zero keeps the pattern like SparseArrays,
        # nonzero would create implicit entries and is rejected.
        g=FaerSparseMatrix(ref; format = fmt)
        r=copy(ref)
        g .= 0
        r .= 0
        @test _same_sparse(g, r, fmt)
        @test_throws ArgumentError (g .= 1)
        h=FaerSparseMatrix(ref; format = fmt)
        h .= h .+ FaerSparseMatrix(ref; format = :csc)
        @test _same_sparse(h, ref .+ ref, fmt)

        # Failed conversions leave the destination unchanged.
        if T<:Real
            k=FaerSparseMatrix(ref; format = fmt)
            vals=nonzeros(k)
            @test_throws InexactError (k .*= im)
            @test _same_sparse(k, ref, fmt)
            @test isopen(vals)
        end

        # Dense destinations accept sparse broadcasts.
        D=zeros(T, 6, 5)
        D .+= FaerSparseMatrix(ref; format = fmt) .* 2
        @test D≈Matrix(ref) .* 2
    end
end

@testset "Closed sparse broadcast operands" begin
    a=FaerSparseMatrix(sparse([1, 2], [1, 2], [1.0, -2.0], 2, 2))
    b=copy(a)
    close(a)
    @test_throws ArgumentError abs.(a)
    @test_throws ArgumentError a .+ 1
    @test_throws ArgumentError map(abs, a)
    @test_throws ArgumentError (a .*= 2)
    @test_throws ArgumentError (b .= abs.(a))
    @test_throws ArgumentError (a .= abs.(b))
    @test Matrix(b)==[1.0 0; 0 -2.0]
end

@testset "Sparse broadcast allocations and ambiguities" begin
    rng=MersenneTwister(5053)
    a=FaerSparseMatrix(sprand(rng, 400, 400, 0.02))
    k=nnz(a)
    abs.(a); a .*= 1.0; map(abs, a)
    # One nnz-length value buffer plus wrapper overhead; the previous
    # dense/element-wise path allocated hundreds of megabytes.
    @test @allocated(abs.(a)) < 2k*sizeof(Float64) + 4096
    @test @allocated(a .*= 1.0) < 2k*sizeof(Float64) + 4096
    @test @allocated(map(abs, a)) < 2k*sizeof(Float64) + 4096
    @test isempty(detect_ambiguities(Colmat, Base, LinearAlgebra, SparseArrays; recursive = true))
end
