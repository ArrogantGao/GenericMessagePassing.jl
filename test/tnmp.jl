using GenericMessagePassing
using Graphs
using ProblemReductions, GenericTensorNetworks, OMEinsum
using TensorInference

using Test
using Random
Random.seed!(1234)

@testset "neighbor generation, precomputing" begin
    for g in [GenericMessagePassing.random_tree(30), random_regular_graph(100, 3), SimpleGraph(GenericTensorNetworks.random_square_lattice_graph(30, 30, 0.8))]
        for r in [2, 3]
            tn, code, tensors = GenericMessagePassing.ising_model(g, ones(nv(g)), ones(ne(g)), 1.0, verbose = true)
            factor_graph = FactorGraph(code)

            neibs, boundaries = GenericMessagePassing.generate_neighborhoods(factor_graph, r)
            
            @testset "neighbor generation" begin
                for v in keys(neibs)
                    @test v ∈ neibs[v]
                    @test issubset(boundaries[v], neibs[v])
                    @test Set(GenericMessagePassing.open_boundaries(factor_graph, neibs[v])) == Set(boundaries[v])
                    fgt = copy(factor_graph)
                    GenericMessagePassing.isolate_vertices!(fgt, setdiff(neibs[v], boundaries[v]))
                    for i in 1:length(boundaries[v]) - 1
                        for j in i+1:length(boundaries[v])
                            s, d = boundaries[v][i], boundaries[v][j]
                            path = a_star(fgt, s, d)
                            @test (length(path) == 0) || (length(path) - 1 > r)
                        end
                    end
                end
            end

            # messages, eins, ptensors, mars_eins, mars_tensors = GenericMessagePassing.tnbp_precompute(factor_graph, code, tensors, neibs, boundaries, GreedyMethod())

            # @testset "precompute" begin
                
            # end
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
    bp_sol = marginal_tnbp(code, tensors, TNBPConfig(verbose = true))
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
    bp_sol = marginal_tnbp(code, tensors, TNBPConfig(verbose = true))
    for i in keys(bp_sol)
        @test isapprox(ti_sol[[i]][1], bp_sol[i][1], atol = 1e-4)
        @test isapprox(ti_sol[[i]][2], bp_sol[i][2], atol = 1e-4)
    end
end

@testset "marginal spinglass rr3" begin
    # rr3
    g = random_regular_graph(100, 3)
    h = ones(nv(g))
    J = -1 .* ones(ne(g))
    β = 1.0
    tn, code, tensors = GenericMessagePassing.ising_model(g, h, J, β, verbose = true)
    ti_sol = marginals(tn)
    bp_sol = marginal_tnbp(code, tensors, TNBPConfig(verbose = true))
    for i in keys(bp_sol)
        @test isapprox(ti_sol[[i]][1], bp_sol[i][1], atol = 1e-4)
        @test isapprox(ti_sol[[i]][2], bp_sol[i][2], atol = 1e-4)
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

    for random_order in [false]
        tnbp_config = TNBPConfig(random_order = random_order, verbose = true, error = 1e-12, r = 2)
        bp_sol = marginal_tnbp(code, tensors, tnbp_config)
        for i in keys(bp_sol)
            @test isapprox(ti_sol[[i]][1], bp_sol[i][1], atol = 1e-3)
            @test isapprox(ti_sol[[i]][2], bp_sol[i][2], atol = 1e-3)
        end
    end

    # problem_from_artifact("uai2014", "MAR", "ObjectDetection", 42)
end
