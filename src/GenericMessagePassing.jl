module GenericMessagePassing

# imported packages
using Graphs, LinearAlgebra
using ProblemReductions, GenericTensorNetworks, OMEinsum
using TensorInference

using Statistics, Suppressor

# imported functions
using OMEinsum.OMEinsumContractionOrders: IncidenceList

# exported types
export FactorGraph
export BPConfig, TNBPConfig

# exported functions
export random_k_sat
export tn_model

export bp, marginal_bp, entropy_bp
export marginal_tnbp

include("types.jl")
include("generic.jl")
include("utils.jl")

include("mp.jl")
include("tnmp.jl")

end
