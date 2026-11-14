"""
    Colmat

BLAS backend powered by Rust's colmat. Loading the package installs LP64 and
ILP64 forwards in libblastrampoline. Use ordinary Julia arrays and LinearAlgebra;
unprovided BLAS and LAPACK routines retain their existing backend.
"""
module Colmat

using Libdl: Libdl
using LinearAlgebra: BLAS

public load_backend, backend_library, get_num_threads, set_num_threads

include("backend.jl")

function __init__()
    return load_backend()
end

end
