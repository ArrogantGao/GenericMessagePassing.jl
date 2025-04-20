module GenericMessagePassing

using Graphs
using ProblemReductions, GenericTensorNetworks, OMEinsum
using TensorInference

export BPConfig
export bp, message2marginals

include("types.jl")
include("utils.jl")

include("mp.jl")
include("tnmp.jl")

end
