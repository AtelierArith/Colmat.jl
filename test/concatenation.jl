using Test, LinearAlgebra, Random, SparseArrays, Colmat

const _ConcatTypes = (Float32, Float64, ComplexF32, ComplexF64)

function _same_native(x, ref)
    expected = ref isa Vector ? FaerVector : FaerMatrix
    return x isa expected &&
           eltype(x) === eltype(ref) &&
           size(x) == size(ref) &&
           Array(x) == ref
end
function _same_sparse(x, ref)
    return x isa FaerSparseMatrix &&
           eltype(x) === eltype(ref) &&
           size(x) == size(ref) &&
           SparseMatrixCSC(x) == ref
end

@testset "Concatenation with a later native argument" begin
    rng = MersenneTwister(98)
    for T in _ConcatTypes
        a = randn(rng, T, 3, 2)
        j = randn(rng, T, 3, 2)
        x = randn(rng, T, 3)
        A = FaerMatrix{T}(a)
        v = FaerVector{T}(x)
        r = FaerRow{T}(T[4, 5])
        @test _same_native(hcat(j, A), hcat(j, a))
        @test _same_native(vcat(j, A), vcat(j, a))
        @test _same_native([j A; A j], [j a; a j])
        @test _same_native([j j A], [j j a])
        @test _same_native(hcat(j, j, j, j, j, j, j, A), hcat(j, j, j, j, j, j, j, a))
        @test _same_native(cat(j, A; dims=2), cat(j, a; dims=2))
        @test cat(j, A; dims=3) isa Array{T,3} && cat(j, A; dims=3) == cat(j, a; dims=3)
        @test _same_native(vcat(one(T), v), vcat(one(T), x))
        @test _same_native(vcat(x, v), vcat(x, x))
        @test _same_native(hcat(x, v), hcat(x, x))
        @test _same_native(vcat(j[1:1, :], r), vcat(j[1:1, :], T[4 5]))
        @test hcat(one(T), r) isa FaerRow{T} && Vector(hcat(one(T), r)) == T[1, 4, 5]
        @test _same_native(vcat(j', A'), vcat(collect(j'), collect(a')))
        @test _same_native(hcat(view(j, :, 1), view(A, :, 2)), hcat(j[:, 1], a[:, 2]))
        @test_throws DimensionMismatch hcat(j, FaerMatrix{T}(zeros(T, 2, 2)))
    end
end

@testset "Typed and n-dimensional concatenation" begin
    for T in _ConcatTypes
        a = T[1 2; 3 4]
        j = T[5 6; 7 8]
        A = FaerMatrix{T}(a)
        @test _same_native(T[j A], T[j a])
        @test _same_native(T[j; A], T[j; a])
        @test _same_native(T[A j; j A], T[a j; j a])
        @test _same_native(T[j A; A j], T[j a; a j])
        @test _same_native(ComplexF64[A j], ComplexF64[a j])
        r = FaerRow{T}(T[1, 2])
        @test T[r r] isa FaerRow{T} && Vector(T[r r]) == T[1, 2, 1, 2]
        @test _same_native(T[A' j], T[collect(a') j])
        @test _same_native([A;; j], [a;; j])
        @test _same_native([j;; A], [j;; a])
        @test _same_native(T[j;; A], T[j;; a])
        r3 = [A;;; j]
        @test r3 isa Array{T,3} && r3 == [a;;; j]
        @test [j;;; A] == [j;;; a]
        @test [A j;;; j A] == [a j;;; j a]
    end
    A = FaerMatrix([1.0 2; 3 4])
    j = [5.0 6; 7 8]
    # Non-native result scalars stay Julia arrays.
    @test Int[A [1 2; 3 4]] isa Matrix{Int}
    @test Int[[1 2; 3 4] A] == [1 2 1 2; 3 4 3 4]
    @test BigFloat[j A] isa Matrix{BigFloat}
    @test BigFloat[j A] == BigFloat[j Matrix(A)]
end

@testset "Concatenation result convention" begin
    a = [1.0 2; 3 4]
    A = FaerMatrix(a)
    B = faerview(copy(a))
    @test _same_native(hcat(B, A), hcat(a, a))
    @test _same_native(hcat(a, B, A), hcat(a, a, a))
    @test hcat(B, a) isa Matrix{Float64}
    @test hcat(a, B) isa Matrix{Float64}
    @test hcat(a, a) isa Matrix{Float64}
    @test hcat(1.0, 2.0) isa Matrix{Float64}
    @test vcat([1.0], [2.0]) isa Vector{Float64}
end

@testset "Native sparse concatenation" begin
    for T in _ConcatTypes
        s = sparse(T[1 0; 0 2])
        t = sparse(T[0 3; 4 0])
        S = FaerSparseMatrix(s)
        U = FaerSparseMatrix(t)
        d = T[5 6; 7 8]
        A = FaerMatrix{T}(d)
        @test _same_sparse(hcat(S, U), hcat(s, t))
        @test _same_sparse(vcat(S, U), vcat(s, t))
        @test _same_sparse([S U; U S], [s t; t s])
        @test _same_sparse(hcat(S, t), hcat(s, t))
        @test _same_sparse(hcat(t, S), hcat(t, s))
        @test _same_sparse(hcat(d, S), hcat(d, s))
        @test _same_sparse(hcat(S, A), hcat(s, d))
        @test _same_sparse(hcat(A, S), hcat(d, s))
        @test _same_sparse(vcat(d, S, A), vcat(d, s, d))
        # Typed literals are dense in SparseArrays too.
        @test _same_native(T[d S], T[d s])
        @test _same_sparse(cat(S, U; dims=(1, 2)), cat(s, t; dims=(1, 2)))
        # Julia sparse with native dense follows SparseArrays and stays Julia.
        @test hcat(t, A) isa SparseMatrixCSC{T}
        @test hcat(t, A) == hcat(t, d)
    end
    # Non-finite stored values and stored zeros survive the round trip.
    s = SparseMatrixCSC(2, 2, [1, 3, 4], [1, 2, 2], [Inf, 0.0, NaN])
    S = FaerSparseMatrix(s)
    r = hcat(S, S)
    @test r isa FaerSparseMatrix
    @test isequal(SparseMatrixCSC(r), hcat(s, s))
    @test nnz(r) == 6
    C = FaerSparseMatrix(sparse([1.0 0; 0 2]); format=:csr)
    @test _same_sparse(vcat(C, C), vcat(sparse([1.0 0; 0 2]), sparse([1.0 0; 0 2])))
end

@testset "Reductions and foreach over native arrays" begin
    for T in _ConcatTypes
        a = T[1 2; 3 4; 5 6]
        A = FaerMatrix{T}(a)
        v = FaerVector{T}(T[1, 2, 3])
        @test _same_native(reduce(hcat, [A, A]), hcat(a, a))
        @test _same_native(reduce(vcat, [A, A]), vcat(a, a))
        @test _same_native(reduce(hcat, [v, v]), hcat(Vector(v), Vector(v)))
        @test _same_native(reduce(vcat, [v, v]), vcat(Vector(v), Vector(v)))
        @test _same_native(reduce(hcat, Any[a, A]), hcat(a, a))
        seen = T[]
        foreach(x -> push!(seen, x), A)
        @test seen == vec(a)
        empty!(seen)
        foreach(x -> push!(seen, x), A')
        @test seen == vec(collect(a'))
        empty!(seen)
        foreach((x, y) -> push!(seen, x + y), v, T[10, 20, 30])
        @test seen == T[11, 22, 33]
        @test mapreduce(abs2, +, A) ≈ mapreduce(abs2, +, a)
        @test sum(x -> 2x, A) ≈ sum(x -> 2x, a)
        @test maximum(abs, A) == maximum(abs, a)
        @test mapreduce(abs2, +, A; init=zero(real(T))) ≈ sum(abs2, a)
        @test isequal(Array(mapreduce(abs2, +, A; dims=1)), mapreduce(abs2, +, a; dims=1))
        @test mapreduce(identity, +, v) == 6
    end
end

@testset "Map with a later native argument" begin
    for T in _ConcatTypes
        a = T[1 2; 3 4]
        j = T[5 6; 7 8]
        A = FaerMatrix{T}(a)
        @test _same_native(map(+, j, A), j .+ a)
        @test _same_native(map(+, j, j, A), j .+ j .+ a)
        @test _same_native(
            map(*, view(j, :, 1), FaerVector{T}(T[1, 2])), j[:, 1] .* T[1, 2]
        )
        @test map(==, j, A) isa Array{Bool}
        dest = zeros(T, 2, 2)
        map!(+, dest, j, A)
        @test dest == j .+ a
        map!(abs, dest, A)
        @test dest == abs.(a)
    end
end

@testset "Native diagonal broadcast and concatenation" begin
    for T in _ConcatTypes
        d = T[1, 2, 3]
        D = FaerDiagonal{T}(d)
        J = Diagonal(d)
        m = randn(T, 3, 3)
        A = FaerMatrix{T}(m)
        r = D .* 2
        @test r isa FaerDiagonal{T} && Matrix(r) == J .* 2
        r = abs.(D)
        @test r isa FaerDiagonal && Matrix(r) == abs.(J)
        @test _same_native(D .+ 1, J .+ 1)
        @test _same_native(D .+ A, J .+ m)
        @test _same_native(A .* D, m .* J)
        r = D .* D
        @test r isa FaerDiagonal{T} && Matrix(r) == J .* J
        @test (D .== 0) isa AbstractMatrix{Bool}
        @test (D .== 0) == (J .== 0)
        E = FaerDiagonal{T}(T[7, 8, 9])
        E .= D .* 3
        @test Matrix(E) == Matrix(J .* 3)
        E .*= 2
        @test Matrix(E) == Matrix(J .* 6)
        # A failed write (a nonzero off-diagonal) leaves the destination unchanged.
        before = Matrix(E)
        @test_throws ArgumentError E .= D .+ 1
        @test Matrix(E) == before
        @test _same_native(hcat(D, A), hcat(Matrix(J), m))
        @test _same_native(hcat(m, D), hcat(m, Matrix(J)))
    end
    n = 200
    D = FaerDiagonal(rand(n))
    A = FaerMatrix(rand(n, n))
    f1() = D .* 2
    f2() = D .+ A
    f1()
    f2()
    @test @allocated(f1()) < 65536
    @test @allocated(f2()) < 4 * n * n * sizeof(Float64) + 65536
end

@testset "Closed native operands in mixed calls" begin
    J = [1.0 2; 3 4]
    A = FaerMatrix(J)
    D = FaerDiagonal([1.0, 2.0])
    S = FaerSparseMatrix(sparse(J))
    close(A)
    close(D)
    close(S)
    @test_throws ArgumentError hcat(J, A)
    @test_throws ArgumentError [J A; A J]
    @test_throws ArgumentError map(+, J, A)
    @test_throws ArgumentError foreach(identity, A)
    @test_throws ArgumentError sum(abs2, A)
    @test_throws ArgumentError D .* 2
    @test_throws ArgumentError hcat(J, S)
    E = FaerDiagonal([5.0, 6.0])
    @test_throws ArgumentError E .= D .* 2
    @test Vector(E) == [5.0, 6.0]
    dest = zeros(2, 2)
    @test_throws ArgumentError map!(+, dest, J, A)
    @test iszero(dest)
end

# Julia reference of a concatenation argument.
_concat_ref(x::Union{FaerMatrix,FaerVector,FaerRow,Colmat.FaerView}) = Array(x)
_concat_ref(x::Adjoint{<:Any,<:FaerMatrix}) = collect(x)
_concat_ref(x::FaerSparseMatrix) = SparseMatrixCSC(x)
_concat_ref(x::FaerDiagonal) = Diagonal(Vector(x))
_concat_ref(x::Colmat.FaerBorrowedArray) = parent(x)
_concat_ref(x) = x

# Calls every argument combination; returns the calls that hit something other
# than a shape error or disagree with the same call on Julia arguments. (Kept in
# a function: Julia 1.13 crashes inferring this loop as top-level code.)
function _mixed_concat_failures(kinds)
    failures = Any[]
    calls = (hcat, vcat, (xs...) -> hvcat((2,), xs...))
    for p in kinds, q in kinds, (k, f) in enumerate(calls), args in ((p, q), (p, q, p))
        k == 3 && length(args) == 3 && continue
        got = try
            f(args...)
        catch e
            e isa DimensionMismatch || push!(failures, (k, typeof.(args), e))
            continue
        end
        ref = f(map(_concat_ref, args)...)
        isequal(collect(got), collect(ref)) || push!(failures, (k, typeof.(args), :value))
    end
    for p in kinds[2:end], q in kinds[2:end]
        got = try
            map(+, p, q)
        catch e
            e isa DimensionMismatch || push!(failures, (:map, typeof.((p, q)), e))
            continue
        end
        isequal(collect(got), collect(map(+, _concat_ref(p), _concat_ref(q)))) ||
            push!(failures, (:map, typeof.((p, q)), :value))
    end
    return failures
end

@testset "No method ambiguities in mixed concatenation" begin
    a = [1.0 2; 3 4]
    x = [1.0, 2.0]
    kinds = Any[
        1.0,
        a,
        x,
        x',
        sparse(a),
        faerview(copy(a)),
        view(a, :, :),
        FaerMatrix(a),
        FaerVector(x),
        FaerRow(x),
        view(FaerMatrix(a), :, :),
        FaerMatrix(a)',
        FaerSparseMatrix(sparse(a)),
        FaerDiagonal(x),
        Diagonal(x),
    ]
    @test isempty(_mixed_concat_failures(kinds))
    @test isempty(
        filter(Test.detect_ambiguities(Colmat; recursive=true)) do (m1, m2)
            m1.module === Colmat || m2.module === Colmat
        end,
    )
end

@testset "Mixed concatenation allocations" begin
    n = 200
    A = FaerMatrix(rand(n, n))
    J = rand(n, n)
    S = FaerSparseMatrix(sprand(n, n, 0.05))
    bytes = n * n * sizeof(Float64)
    f1() = hcat(J, A)
    f2() = [J A; A J]
    f3() = Float64[A J; J A]
    f4() = [A;; J]
    f5() = map(+, J, A)
    f6() = hcat(S, A)
    f7() = (c=Ref(0.0); foreach(x -> (c[] += x), A); c[])
    f8() = sum(abs2, A)
    f9() = Float64[A J]
    for f in (f1, f2, f3, f4, f5, f6, f7, f8, f9)
        f()
    end
    # Each of these allocated 90-570 MB element by element before.
    @test @allocated(f1()) < 4bytes + 65536
    @test @allocated(f2()) < 12bytes + 65536
    @test @allocated(f3()) < 12bytes + 65536
    @test @allocated(f4()) < 4bytes + 65536
    @test @allocated(f5()) < 4bytes + 65536
    # The dense operand becomes sparse storage (about 2.5 bytes per entry here).
    @test @allocated(f6()) < 12bytes + 65536
    @test @allocated(f7()) < 2bytes + 65536
    @test @allocated(f8()) < 2bytes + 65536
    # A native first argument concatenates straight into a native owner.
    @test @allocated(f9()) < 65536
end
