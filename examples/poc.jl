using Colmat, LinearAlgebra

a = [4.0 1 2; 1 5 1; 2 1 6]
b = [1.0, 2, 3]

@show faer_mul(a, a)
x = faer_solve(a, b)
@show x norm(a * x - b)
f = faer_svd(a)
@show f.S norm(f.U * Diagonal(f.S) * f.Vt - a)

# Keep matrices resident in Rust across a sequence of operations.
A, B = FaerMatrix(a), FaerMatrix(Matrix{Float64}(I, 3, 3))
C = 2A + B - A / 2
@show Matrix(C) Matrix(A * B) A \ b
@show svd(A).S svdvals(A)
mul!(C, A, B)
@assert Matrix(C) ≈ a
close(A)
close(B)
close(C)
