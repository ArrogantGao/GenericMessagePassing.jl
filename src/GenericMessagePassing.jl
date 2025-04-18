module GenericMessagePassing

using Graphs
using ProblemReductions, GenericTensorNetworks
using OMEinsum, OMEinsumContractionOrders

export BPConfig
export bp, message2marginals

include("types.jl")
include("utils.jl")

include("bp.jl")
include("tnmp.jl")
include("ftnmp.jl")

end
