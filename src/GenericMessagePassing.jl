module GenericMessagePassing

# imported packages
using Graphs
using ProblemReductions, GenericTensorNetworks, OMEinsum
using TensorInference

using Statistics, Suppressor

# imported functions
using OMEinsum.OMEinsumContractionOrders: IncidenceList

# exported types
export BPConfig, TNBPConfig

# exported functions
export bp, message2marginals
export tnbp

include("types.jl")
include("utils.jl")

include("mp.jl")
include("tnmp.jl")

end
