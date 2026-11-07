using Test, Colmat, LinearAlgebra, SparseArrays, Statistics

# An independent NPY fixture encoder: writes IEEE component bits explicitly.
function _npy_bytes(
    A::Matrix{T};
    shape = size(A),
    fortran = false,
    big = false,
    version = 1,
    descr = nothing,
    extra = 0,
) where {T}
    R=typeof(real(zero(T)))
    U=R===Float32 ? UInt32 : UInt64
    kind=T<:Complex ? "c" : "f"
    descr=isnothing(descr) ? string(big ? '>' : '<', kind, sizeof(T)) : descr
    shaped=isempty(shape) ? "()" :
           length(shape)==1 ? "($(shape[1]),)" : "("*join(shape, ", ")*")"
    header="{'descr': '$descr', 'fortran_order': $(fortran ? "True" : "False"), 'shape': $shaped, }"
    prefix=version==1 ? 10 : 12
    header*=repeat(" ", mod(-prefix-ncodeunits(header)-1, 64)+extra)*"\n"
    bytes=UInt8[0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, version, 0]
    append!(
        bytes,
        [UInt8((UInt32(ncodeunits(header))>>(8j))&0xff) for j = 0:(version==1 ? 1 : 3)],
    )
    append!(bytes, codeunits(header))
    values=fortran ? vec(A) : vec(permutedims(A))
    for x in values, part in (T<:Complex ? (real(x), imag(x)) : (x,))
        word=reinterpret(U, part)
        for j in (big ? reverse(0:(sizeof(R)-1)) : (0:(sizeof(R)-1)))
            push!(bytes, UInt8((word>>(8j))&0xff))
        end
    end
    return bytes
end
_bits(A) = collect(reinterpret(UInt8, vec(A)))
@testset "Native NPY import and owned views" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        for dims in ((1, 1), (0, 0), (0, 3), (3, 0), (3, 4)),
            fortran in (false, true),
            big in (false, true),
            version in (1, 2, 3)

            m, n=dims
            A=reshape(T.(1:(m*n)), m, n)
            T<:Complex && (A .+= T(0.25im))
            bytes=_npy_bytes(A; fortran, big, version)
            loaded=loadnpy(bytes)
            @test loaded isa FaerMatrix{T}
            @test Matrix(loaded)==A
            V=npyview(bytes)
            @test V isa FaerNpyMatrix{T}
            @test size(V)==dims
            @test _bits(Matrix(V))==_bits(A)
            info=npyinfo(V)
            @test info.dtype===T
            @test info.shape==dims
            @test info.fortran_order==fortran
            @test info.version==(version, 0)
            @test info.byteorder==(big ? :big : :little)
            @test info.aligned
            @test info.normalized_byteorder==(Base.ENDIAN_BOM==0x04030201 ? :little : :big)
            if !isempty(A)
                @test V[1, 1]==A[1, 1]
                @test V[end, end]==A[end, end]
                @test V[1]==A[1]
                @test_throws ArgumentError V[1, 1]=zero(T)
            end
            @test_throws BoundsError V[0, 1]
            copied=copy(V)
            @test copied isa FaerMatrix{T}
            @test Matrix(copied)==A
            @test Matrix(FaerMatrix{T}(V))==A
            # Input mutation cannot invalidate the owned normalized byte view.
            fill!(bytes, 0x00)
            GC.gc(false)
            @test Matrix(V)==A
            close(V)
            @test !isopen(V)
            @test_throws ArgumentError Matrix(V)
            @test_throws ArgumentError copy(V)
            @test_throws ArgumentError npyinfo(V)
            @test Matrix(copied)==A
        end
        # faer's to_mat reads unaligned and foreign-endian sources in place
        # (tensor4all/faer#210/#211 fixed by tensor4all/faer#215).
        A=reshape(T.(1:6), 2, 3)
        T<:Complex && (A .-= T(0.5im))
        for fortran in (false, true), big in (false, true), version in (1, 2, 3)
            bytes=_npy_bytes(A; fortran, big, version)
            for offset = 1:9
                padded=vcat(zeros(UInt8, offset), bytes, UInt8[0x7f])
                source=view(padded, (offset+1):(offset+length(bytes)))
                loaded=loadnpy(source)
                @test loaded isa FaerMatrix{T}
                @test _bits(Matrix(loaded))==_bits(A)
                fill!(padded, 0x00)
                @test _bits(Matrix(loaded))==_bits(A)
            end
            strided=view(repeat(bytes; inner = 2), 1:2:(2*length(bytes)))
            @test _bits(Matrix(loadnpy(strided)))==_bits(A)
        end
        for shape in ((), (0,), (3,))
            A=reshape(T.(1:(isempty(shape) ? 1 : only(shape))), :, 1)
            bytes=_npy_bytes(A; shape, big = true, version = 3)
            @test Matrix(loadnpy(bytes))==A
            @test npyinfo(bytes).shape==shape
        end
        # NaN payloads, infinities and signed zero survive endian normalization.
        R=typeof(real(zero(T)))
        payload_nan=reinterpret(
            R,
            R===Float32 ? UInt32(0x7fc00001) : UInt64(0x7ff8000000000001),
        )
        values=T<:Complex ?
               T[
            complex(payload_nan, -zero(R)),
            complex(-zero(R), payload_nan),
            complex(R(Inf), R(-Inf)),
            complex(R(1.25), R(-2.5)),
        ] : R[payload_nan, Inf, -Inf, -zero(R)]
        A=reshape(values, 2, 2)
        for big in (false, true), fortran in (false, true)
            bytes=_npy_bytes(A; big, fortran)
            @test _bits(Matrix(loadnpy(bytes)))==_bits(A)
            @test _bits(Matrix(npyview(bytes)))==_bits(A)
        end
    end
    A=[1.0 2 3; 4 5 6]
    @test Matrix(loadnpy(joinpath(@__DIR__, "fixtures", "npy_c_f64.npy")))==A
    @test fetch(@async Matrix(loadnpy(joinpath(@__DIR__, "fixtures", "npy_c_f64.npy"))))==A
    bytes=_npy_bytes(A)
    io=IOBuffer(bytes)
    @test Matrix(loadnpy(io))==A
    @test eof(io)
    padded=vcat(UInt8[0], bytes, UInt8[0])
    V=npyview(view(padded, 2:(length(padded)-1)))
    @test Matrix(V)==A
    copiedview=deepcopy(V)
    @test typeof(copiedview)===typeof(V)
    @test copiedview.handle.ptr!=V.handle.ptr
    close(V)
    @test Matrix(copiedview)==A
    mktemp() do path, io
        write(io, bytes)
        close(io)
        @test Matrix(loadnpy(path))==A
        @test Matrix(loadnpy(path; view = true))==A
        @test npyinfo(path).shape==size(A)
    end
    for bad in (
        UInt8[],
        UInt8[1, 2, 3],
        bytes[1:9],
        bytes[1:(end-1)],
        vcat(bytes, UInt8[0]),
        _npy_bytes(A; descr = "<i8"),
        _npy_bytes(A; shape = (2, 3, 1)),
        _npy_bytes(A; shape = (-1, 6)),
        _npy_bytes(A; shape = (typemax(UInt64), 0)),
        _npy_bytes(A; shape = (2^62, 4)),
    )
        @test_throws ArgumentError loadnpy(bad)
    end
    bad=copy(bytes)
    bad[7]=4
    @test_throws ArgumentError loadnpy(bad)
    bad=copy(bytes)
    bad[11]=0x00
    @test_throws ArgumentError loadnpy(bad)
    bad=UInt8[0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 2, 0, 0xff, 0xff, 0xff, 0xff]
    @test_throws ArgumentError loadnpy(bad)
    bad=copy(bytes)
    bad[8]=1
    @test_throws ArgumentError loadnpy(bad)
    # A stale wrapper must never export using reloaded dtype/shape metadata.
    for replacement in
        (_npy_bytes(reshape(collect(1.0:6), 3, 2)), _npy_bytes(zeros(Float32, 2, 3)))
        V=npyview(bytes)
        GC.@preserve replacement Colmat._native_call(
            Colmat.RustCrate.npy_load,
            V,
            pointer(replacement),
            UInt(length(replacement)),
        )
        @test_throws Colmat.RustCall.RustError Matrix(V)
        @test_throws Colmat.RustCall.RustError V[1, 1]
        @test_throws Colmat.RustCall.RustError copy(V)
    end
    @test isempty(
        Test.detect_ambiguities(
            Colmat,
            Base,
            LinearAlgebra,
            SparseArrays;
            recursive = true,
        ),
    )
end

# Byte offset of element (i, j) inside an NPY payload of an m×n matrix.
_npy_offset(T, m, n, i, j, fortran) =
    (fortran ? (j-1)*m+(i-1) : (i-1)*n+(j-1))*sizeof(T)
_poke!(bytes, at, x) = copyto!(bytes, at+1, reinterpret(UInt8, [x]), 1, sizeof(x))
@testset "Zero-copy NPY borrows of mapped files and byte vectors" begin
    nativebig=Base.ENDIAN_BOM!=0x04030201
    for T in (Float32, Float64, ComplexF32, ComplexF64),
        fortran in (false, true),
        version in (1, 2, 3)

        m, n=3, 4
        A=reshape(T.(1:(m*n)), m, n)
        T<:Complex && (A .+= T(0.5im))
        bytes=_npy_bytes(A; fortran, version, big = nativebig)
        off=length(bytes)-sizeof(A)
        # In-memory bytes are aliased, not copied.
        B=npyborrow(bytes)
        @test B isa FaerBorrowedMatrix{T}
        @test size(B)==(m, n)
        @test B==A
        @test strides(B)==(fortran ? (1, m) : (n, 1))
        @test pointer(B)==Ptr{T}(pointer(bytes, off+1))
        _poke!(bytes, off+_npy_offset(T, m, n, 2, 3, fortran), T(42))
        A[2, 3]=T(42)
        @test B[2, 3]==T(42)
        # Native consumers read the aliased payload in place.
        C=copy(B)
        @test C isa FaerMatrix{T}
        @test Matrix(C)==A
        @test Matrix(B*faerview(Matrix(A')))≈A*A'
        @test Matrix(transpose(B)*C)≈transpose(A)*A
        @test Matrix(B+C)==2A
        @test sum(B)≈sum(A)
        @test mean(B)≈mean(A)
        @test Matrix(B*FaerVector(T.(1:n)))≈A*T.(1:n)
        # Read-only: every mutation throws instead of writing the source.
        @test_throws ArgumentError B[1, 1]=zero(T)
        @test_throws ArgumentError B.=zero(T)
        @test_throws ArgumentError copyto!(B, zeros(T, m, n))
        @test_throws ArgumentError fill!(B, one(T))
        @test_throws ArgumentError mul!(B, A, Matrix{T}(I, n, n))
        U=UpperTriangular(FaerMatrix(Matrix{T}(I, m, m)+triu(ones(T, m, m))))
        @test_throws ArgumentError ldiv!(U, B)
        @test_throws ArgumentError ldiv!(B, U, FaerMatrix(A))
        S=UpperTriangular(FaerSparseMatrix(sparse(Matrix{T}(2I, m, m))))
        @test_throws ArgumentError ldiv!(S, B)
        @test B==A
        # Mapped files alias the file pages.
        mktemp() do path, io
            write(io, bytes)
            flush(io)
            M=npyborrow(path)
            @test M isa FaerBorrowedMatrix{T}
            @test M==A
            @test strides(M)==strides(B)
            seek(io, off+_npy_offset(T, m, n, 1, 2, fortran))
            write(io, T(7))
            flush(io)
            @test M[1, 2]==T(7)
            A=copy(A)
            A[1, 2]=T(7)
            @test Matrix(copy(M))==A
            @test sum(M)≈sum(A)
            @test_throws ArgumentError M[1, 1]=zero(T)
            @test_throws ArgumentError copyto!(M, zeros(T, m, n))
            close(io)
            # The mapping outlives the closed handle and the unlinked path.
            rm(path)
            GC.gc()
            GC.gc()
            @test M==A
            @test Matrix(transpose(M)*copy(M))≈transpose(A)*A
        end
        # Foreign endian and misaligned payloads fall back to the copying loadnpy.
        big=_npy_bytes(A; fortran, version, big = !nativebig)
        @test npyborrow(big) isa FaerMatrix{T}
        @test Matrix(npyborrow(big))==A
        @test_throws ArgumentError npyborrow(big; fallback = false)
        odd=_npy_bytes(A; fortran, version, big = nativebig, extra = 1)
        @test (length(odd)-sizeof(A))%Base.datatype_alignment(T)!=0
        @test npyborrow(odd) isa FaerMatrix{T}
        @test Matrix(npyborrow(odd))==A
        @test_throws ArgumentError npyborrow(odd; fallback = false)
        mktemp() do path, io
            write(io, odd)
            close(io)
            @test Matrix(npyborrow(path))==A
            @test_throws ArgumentError npyborrow(path; fallback = false)
        end
        # Non-Vector byte sources cannot be aliased.
        @test Matrix(npyborrow(view(bytes, 1:length(bytes))))==B
        @test_throws ArgumentError npyborrow(view(bytes, 1:length(bytes)); fallback = false)
    end
    # Scalars, vectors and empty payloads.
    for (shape, dims) in (((), (1, 1)), ((3,), (3, 1)), ((0, 3), (0, 3)), ((3, 0), (3, 0)))
        A=reshape(collect(1.0:prod(dims)), dims)
        B=npyborrow(_npy_bytes(A; shape, big = nativebig))
        @test B isa FaerBorrowedMatrix{Float64}
        @test size(B)==dims
        @test B==A
        @test Matrix(copy(B))==A
    end
    # A shrunk source vector is detected before any read.
    bytes=_npy_bytes([1.0 2; 3 4]; big = nativebig)
    B=npyborrow(bytes)
    resize!(bytes, 16)
    @test_throws ArgumentError B[1, 1]
    @test_throws ArgumentError copy(B)
    @test_throws ArgumentError sum(B)
    # The borrow keeps the source alive.
    B=npyborrow(_npy_bytes([1.0 2; 3 4]; big = nativebig))
    GC.gc()
    GC.gc()
    @test B==[1.0 2; 3 4]
    for bad in (UInt8[], UInt8[1, 2, 3], _npy_bytes([1.0 2]; descr = "<i8"))
        @test_throws ArgumentError npyborrow(bad)
    end
    mktemp() do path, io
        close(io)
        @test_throws ArgumentError npyborrow(path)
    end
    # Mapping and aliasing keep Julia-side allocations independent of payload size.
    small=_npy_bytes(rand(8, 8); big = nativebig)
    large=_npy_bytes(rand(512, 512); big = nativebig)
    npyborrow(small)
    npyborrow(large)
    @test (@allocated npyborrow(large))<4096
    @test (@allocated npyborrow(large))<=(@allocated npyborrow(small))+256
end
