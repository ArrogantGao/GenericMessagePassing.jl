using GenericMessagePassing
using Graphs
using ProblemReductions, GenericTensorNetworks, OMEinsum
using TensorInference

using Test
using Random
Random.seed!(1234)

@testset "bp precompute plans" begin
    g = random_regular_graph(12, 3)
    h = ones(nv(g))
    J = ones(ne(g))
    β = 1.0
    _, code, tensors = GenericMessagePassing.ising_model(g, h, J, β, verbose = false)

    icode, _ = GenericMessagePassing.intcode(code)
    ixs = getixsv(icode)
    iy = getiyv(icode)
    ids = uniquelabels(icode)
    size_dict = OMEinsum.get_size_dict(ixs, tensors)
    hyper_graph = GenericMessagePassing.IncidenceList(Dict([i => ix for (i, ix) in enumerate(ixs)]), openedges = iy)

    e2v_plans, v2e_plans = GenericMessagePassing.bp_precompute_plans(hyper_graph, ixs, ids, size_dict)
    @test length(e2v_plans) == sum(length.(values(hyper_graph.e2v)))
    @test length(v2e_plans) == sum(length.(values(hyper_graph.e2v)))
    for e in ids, v in hyper_graph.e2v[e]
        @test haskey(e2v_plans, (e, v))
        @test haskey(v2e_plans, (v, e))
    end
end

@testset "marginal tree" begin
    tree_g = GenericMessagePassing.random_tree(30)
    tn = tn_model(tree_g, [rand(2, 2) for _ in 1:ne(tree_g)])
    ti_sol = marginals(tn)

    for random_order in [true, false]
        bp_config = BPConfig(random_order = random_order, verbose = true)
        bp_sol = marginal_bp(tn, bp_config)
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
    bp_sol = marginal_bp(tn, BPConfig(verbose = true))
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
    bp_sol = marginal_bp(tn, BPConfig(verbose = true))
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
        tn = tn_model(problem)
        ti_sol = marginals(tn)

        for random_order in [true, false]
            bp_config = BPConfig(random_order = random_order, verbose = true, error = 1e-12)
            bp_sol = marginal_bp(tn, bp_config)
            for i in keys(bp_sol)
                for j in 1:length(bp_sol[i])
                    @test isapprox(ti_sol[[i]][j], bp_sol[i][j], atol = 1e-2)
                end
            end
        end
    end
end
