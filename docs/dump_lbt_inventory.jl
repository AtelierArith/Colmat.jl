# The same canonical symbol table is shared by LBT's LP64 and ILP64 interfaces.
using Colmat, LinearAlgebra
cfg = BLAS.get_config()
native = Dict(interface => Set{String}() for interface in (:lp64, :ilp64))
for lib in cfg.loaded_libs
    if realpath(lib.libname) == realpath(Colmat.backend_library())
        union!(native[lib.interface], BLAS.lbt_forwarded_funcs(cfg, lib))
    end
end
for symbol in cfg.exported_symbols
    println(symbol, '\t', symbol in native[:lp64], '\t', symbol in native[:ilp64])
end
