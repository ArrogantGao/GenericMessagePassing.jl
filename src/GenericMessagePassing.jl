module GenericMessagePassing

# imported packages
using Graphs
using ProblemReductions, GenericTensorNetworks, OMEinsum
using TensorInference

# imported functions
using OMEinsum.OMEinsumContractionOrders: IncidenceList

# exported types
export BPConfig, TNBPConfig

# exported functions
export bp, message2marginals

include("types.jl")
include("utils.jl")

include("mp.jl")
include("tnmp.jl")

end
