using GenericMessagePassing
using Test

@testset "basic" begin
    include("types.jl")
    include("generic.jl")
    include("utils.jl")
end

@testset "message passing" begin
   include("mp.jl")
end

@testset "tensor network message passing" begin
   include("tnmp.jl")
end
