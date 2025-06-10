using GenericMessagePassing
using Graphs
using ProblemReductions, GenericTensorNetworks, OMEinsum
using TensorInference

using Test
using Random
Random.seed!(1234)

@testset "marginal tree" begin
    tree_g = GenericMessagePassing.random_tree(30)
    tn = tn_model(tree_g, [rand(2, 2) for _ in 1:ne(tree_g)])
    ti_sol = marginals(tn)

    tensors = tn.tensors
    code = tn.code

    for random_order in [true, false]
        bp_config = BPConfig(random_order = random_order, verbose = true)
        bp_sol = marginal_bp(code, tensors, bp_config)
        for i in keys(bp_sol)
            @test isapprox(ti_sol[[i]][1], bp_sol[i][1], atol = 1e-4)
            @test isapprox(ti_sol[[i]][2], bp_sol[i][2], atol = 1e-4)
        end
    end
end

@testset "marginal spinglass tree" begin
    # tree
    g = GenericMessagePassing.random_tree(30)
    h = ones(nv(g))
    J = ones(ne(g))
    β = 1.0
    tn, code, tensors = GenericMessagePassing.ising_model(g, h, J, β, verbose = true)
    ti_sol = marginals(tn)
    bp_sol = marginal_bp(code, tensors, BPConfig(verbose = true))
    for i in keys(bp_sol)
        @test isapprox(ti_sol[[i]][1], bp_sol[i][1], atol = 1e-4)
        @test isapprox(ti_sol[[i]][2], bp_sol[i][2], atol = 1e-4)
    end
end

@testset "marginal spinglass cycle" begin
    # cycle
    g = SimpleGraph(30)
    for i in 1:29
        add_edge!(g, i, i+1)
    end
    add_edge!(g, 30, 1)
    h = ones(nv(g))
    J = ones(ne(g))
    β = 1.0
    tn, code, tensors = GenericMessagePassing.ising_model(g, h, J, β, verbose = true)
    ti_sol = marginals(tn)
    bp_sol = marginal_bp(code, tensors, BPConfig(verbose = true))
    for i in keys(bp_sol)
        @test isapprox(ti_sol[[i]][1], bp_sol[i][1], atol = 1e-4)
        @test isapprox(ti_sol[[i]][2], bp_sol[i][2], atol = 1e-4)
    end
end

@testset "marginal spinglass rr3" begin
    # rr3
    g = random_regular_graph(30, 3)
    h = ones(nv(g))
    J = -1 .* ones(ne(g))
    β = 1.0
    tn, code, tensors = GenericMessagePassing.ising_model(g, h, J, β, verbose = true)
    ti_sol = marginals(tn)
    bp_sol = marginal_bp(code, tensors, BPConfig(verbose = true))
    for i in keys(bp_sol)
        @test isapprox(ti_sol[[i]][1], bp_sol[i][1], atol = 1e-4)
        @test isapprox(ti_sol[[i]][2], bp_sol[i][2], atol = 1e-4)
    end
end

@testset "marginal uai2014" begin
    for problem in [problem_from_artifact("uai2014", "MAR", "Promedus", 14), problem_from_artifact("uai2014", "MAR", "ObjectDetection", 42)]
        optimizer = TreeSA(ntrials = 1, niters = 5, βs = 0.1:0.1:100)
        evidence = Dict{Int, Int}()
        model = read_model(problem)

        tn = TensorNetworkModel(model; optimizer, evidence)
        ti_sol = marginals(tn)

        code = tn.code.eins
        tensors = tn.tensors
        size_dict = Dict(i => d for (i, d) in enumerate(model.cards))

        for random_order in [true, false]
            bp_config = BPConfig(random_order = random_order, verbose = true, error = 1e-12)
            bp_sol = marginal_bp(code, tensors, bp_config)
            for i in keys(bp_sol)
                @test isapprox(ti_sol[[i]][1], bp_sol[i][1], atol = 1e-2)
                @test isapprox(ti_sol[[i]][2], bp_sol[i][2], atol = 1e-2)
            end
        end
    end
end
