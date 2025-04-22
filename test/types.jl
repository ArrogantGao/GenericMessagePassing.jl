using GenericMessagePassing
using Graphs, OMEinsum

using Test
using Random
Random.seed!(1234)

using GenericMessagePassing: FactorGraph

@testset "FactorGraph" begin
    code = EinCode([[1, 2], [2, 3], [1, 3, 4]], Int[])
    fg = FactorGraph(code)
    @test fg.num_vars == 4
    @test nv(fg.g) == 7

    @test Set(neighbors(fg, 1)) == Set([1, 3] .+ 4)
end
