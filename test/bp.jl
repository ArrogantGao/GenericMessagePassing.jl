using GenericMessagePassing
using Graphs
using ProblemReductions, GenericTensorNetworks
using OMEinsum, OMEinsumContractionOrders
using TensorInference

using Test
using Random
Random.seed!(1234)

function graphic_marginals(g::SimpleGraph, tensors)
    ixs = [[i] for i in 1:nv(g)] ∪ [[e.src, e.dst] for e in edges(g)]
    iy = Int[]
    code = EinCode(ixs, iy)
    tensors = vcat([[1.0, 1.0] for i in 1:nv(g)], tensors)
    tn = TensorNetworkModel(1:nv(g), code, tensors)
    return marginals(tn)
end

@testset "marginal tree" begin
    tree_g = GenericMessagePassing.random_tree(30)
    tensors = [rand(2, 2) for _ in 1:ne(tree_g)]
    ti_sol = graphic_marginals(tree_g, tensors)

    code = EinCode([[e.src, e.dst] for e in edges(tree_g)], Int[])
    size_dict = uniformsize(code, 2)

    for random_order in [true, false]
        bp_config = BPConfig(random_order = random_order, verbose = true)
        bp_sol = message2marginals(bp(code, size_dict, tensors, bp_config)[1])
        for i in keys(bp_sol)
            @test isapprox(ti_sol[[i]][1], bp_sol[i][1], atol = 1e-4)
            @test isapprox(ti_sol[[i]][2], bp_sol[i][2], atol = 1e-4)
        end
    end
end

@testset "marginal uai2014" begin
    problem = problem_from_artifact("uai2014", "MAR", "Promedus", 14)
    optimizer = TreeSA(ntrials = 1, niters = 5, βs = 0.1:0.1:100)
    evidence = Dict{Int, Int}()
    model = read_model(problem)

    tn = TensorNetworkModel(model; optimizer, evidence)
    ti_sol = marginals(tn)

    code = tn.code.eins
    tensors = tn.tensors
    size_dict = Dict(i => model.cards[i] for i in 1:model.nvars)

    for random_order in [true, false]
        bp_config = BPConfig(random_order = random_order, verbose = true)
        bp_sol = message2marginals(bp(code, size_dict, tensors, bp_config))
        for i in keys(bp_sol)
            @test isapprox(ti_sol[[i]][1], bp_sol[i][1], atol = 1e-4)
            @test isapprox(ti_sol[[i]][2], bp_sol[i][2], atol = 1e-4)
        end
    end
end
