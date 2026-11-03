using Colmat, LinearAlgebra

a = [4.0 1 2; 1 5 1; 2 1 6]
b = [1.0, 2, 3]

@show faer_mul(a, a)
x = faer_solve(a, b)
@show x norm(a * x - b)
f = faer_svd(a)
@show f.S norm(f.U * Diagonal(f.S) * f.Vt - a)
