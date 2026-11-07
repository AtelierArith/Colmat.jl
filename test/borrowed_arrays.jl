using Test, LinearAlgebra, Statistics, SparseArrays, Colmat

_borrowed_bytes(f, args...) = @allocated f(args...)

@testset "Borrowed Julia arrays" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        data=reshape(T.(1:30), 5, 6)
        T<:Complex && (data .+= im .* reverse(data; dims = 1))
        layouts=(
            identity,
            x->view(x, :, 1:2:6),
            x->view(x, 5:-1:1, :),
            x->view(x, 4:-2:1, 6:-3:1),
            x->view(x, 1:0, 1:6),
        )
        for layout in layouts
            parent=copy(data)
            julia=layout(parent)
            b=faerview(julia)
            @test b isa FaerBorrowedMatrix{T}
            @test size(b)==size(julia)
            @test strides(b)==strides(julia)
            @test Base.parent(b)===julia
            @test b==julia
            # Native snapshots read the Julia memory directly.
            @test Matrix(copy(b))==julia
            @test copy(b) isa FaerMatrix{T}
            @test Matrix(b+b)==julia+julia
            @test Matrix(b-FaerMatrix(julia))==zero(julia)
            if size(julia, 1)>0
                @test Matrix(b'*b)≈julia'julia
                @test Matrix(transpose(b)*FaerMatrix(julia))≈transpose(julia)*julia
                @test Matrix(FaerMatrix(julia)*b')≈julia*julia'
            end
            # Zero-copy: later Julia writes are visible, borrowed writes reach Julia.
            if !isempty(julia)
                julia[1, 1]=T(-7)
                @test Matrix(copy(b))[1, 1]==T(-7)
                b[end, end]=T(9)
                @test julia[end, end]==T(9)
                @test mean(b; dims = 1)≈mean(julia; dims = 1)
                @test var(b)≈var(julia)
            end
        end
        # Nested views remain borrowed Julia memory.
        parent=copy(data)
        b=faerview(parent)
        w=view(b, 5:-2:1, 2:3)
        @test w isa FaerBorrowedMatrix{T}
        @test Matrix(copy(w))==data[5:-2:1, 2:3]
        c=view(b, :, 2)
        @test c isa FaerBorrowedVector{T}
        @test Vector(copy(c))==data[:, 2]
        @test copy(c) isa FaerVector{T}
        # Matrix-vector products with borrowed vectors.
        v=faerview(T.(1:6))
        @test Vector(FaerMatrix(data)*v)≈data*T.(1:6)
        @test Vector(b*v)≈data*T.(1:6)
        # Mixed scalar types promote like native arrays.
        if T!==ComplexF64
            @test Matrix(b*faerview(ComplexF64.(data')))≈data*ComplexF64.(data')
        end

        # Assignment sources borrow Julia memory without a staging copy.
        a=FaerMatrix(zero(data))
        src=copy(data)
        copyto!(view(a, :, :), src)
        @test Matrix(a)==data
        copyto!(view(a, 5:-1:1, :), view(src, :, :))
        @test Matrix(a)==data[5:-1:1, :]
        copyto!(view(a, :, 2), faerview(src[:, 3]))
        @test Matrix(a)[:, 2]==data[:, 3]
        copyto!(view(a, 2, :), src[4, :])
        @test Matrix(a)[2, :]==data[4, :]
        big=FaerMatrix{T}(undef, 200, 200)
        buffer=rand(T, 200, 200)
        dest=view(big, :, :)
        copyto!(dest, buffer)
        @test _borrowed_bytes(copyto!, dest, buffer)<sizeof(buffer)÷8
        @test Matrix(big)==buffer
        bb=faerview(buffer)
        copy(bb)
        @test _borrowed_bytes(copy, bb)<2*sizeof(buffer)
        # A failed assignment leaves the destination intact.
        a=FaerMatrix(data)
        @test_throws DimensionMismatch copyto!(view(a, :, :), ones(T, 6, 5))
        @test Matrix(a)==data

        # Borrowed arrays survive collection of every other Julia reference.
        b=faerview(copy(data))
        GC.gc()
        @test Matrix(copy(b))==data
    end

    # Only native scalars with non-overlapping layouts are borrowed.
    @test_throws ArgumentError faerview(ones(Int, 2, 2))
    @test_throws ArgumentError faerview(ones(BigFloat, 2, 2))
    @test_throws ArgumentError faerview(view(ones(4, 4), StepRangeLen(1, 0, 3), 1:2))
    @test_throws ArgumentError faerview(zeros(2, 2, 2))
    # Views keep the faerview invariant; rejected layouts stay Julia views.
    @test !(view(faerview(ones(4, 4)), StepRangeLen(1, 0, 3), 1:2) isa FaerBorrowedArray)
    # Misaligned reinterpreted memory is rejected and staged when copied.
    bytes=zeros(UInt8, 17)
    misaligned=reinterpret(Float64, view(bytes, 2:17))
    misaligned[1], misaligned[2]=1.5, -2.5
    @test_throws ArgumentError faerview(misaligned)
    f=FaerVector(zeros(2))
    copyto!(view(f, :), misaligned)
    @test Vector(f)==[1.5, -2.5]
    @test faerview(faerview(ones(2, 2))) isa FaerBorrowedMatrix{Float64,Matrix{Float64}}
    # Vector parents that are resized are re-read at every native call.
    x=collect(1.0:3.0)
    bx=faerview(x)
    append!(x, 4.0:1000.0)
    @test length(bx)==1000
    @test Vector(copy(bx))==1.0:1000.0
    resize!(x, 2)
    @test Vector(copy(bx))==[1.0, 2.0]
    # Statistics has internal ambiguities of its own; check only Colmat's.
    ambiguities=Test.detect_ambiguities(
        Colmat,
        Base,
        LinearAlgebra,
        SparseArrays,
        Statistics;
        recursive = true,
    )
    @test isempty(filter(a->a[1].module===Colmat||a[2].module===Colmat, ambiguities))
end
