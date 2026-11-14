using Colmat, LinearAlgebra

A = [4.0 1; 1 3]
B = [2.0 0; 0 2]
@assert A * B ≈ [8.0 2; 2 6]
C = similar(A)
mul!(C, A, B)
@assert C ≈ [8.0 2; 2 6]
x = A \ [1.0, 2.0] # Colmat LAPACK drivers handle the decomposition.
@assert A * x ≈ [1.0, 2.0]
println(BLAS.get_config())
