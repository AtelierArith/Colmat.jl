# Trace project-owned foreign calls during the ordinary full suite. Colmat must
# load inside suite.jl, after the original fallback pointers have been captured.
include("abi_execution_evidence.jl")
ABIEvidence.include_traced(Main, joinpath(@__DIR__, "suite.jl"))
ABIEvidence.finalize()
