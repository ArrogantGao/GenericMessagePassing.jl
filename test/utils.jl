using GenericMessagePassing
using Graphs, OMEinsum

using Test
using Random
Random.seed!(1234)

using GenericMessagePassing: FactorGraph, open_neighbors, open_boundaries, isolate_vertex!, isolate_vertices!

@testset "operations on FactorGraph" begin
    code = EinCode([[1, 2], [2, 3], [1, 3, 4]], Int[])
    fg = FactorGraph(code)

    @test Set(open_neighbors(fg, [1, 5])) == Set([2, 7])
    @test Set(open_boundaries(fg, [1, 2, 5, 6, 7])) == Set([6, 7])

    fgt = copy(fg)
    isolate_vertex!(fgt, 1)
    @test isempty(neighbors(fgt, 1))

    fgt = copy(fg)
    isolate_vertices!(fgt, [1, 2])
    @test isempty(neighbors(fgt, 1))
    @test isempty(neighbors(fgt, 2))
end