using GenericMessagePassing
using Graphs
using ProblemReductions, GenericTensorNetworks, OMEinsum
using TensorInference

using Test
using Random
Random.seed!(1234)

@testset "neighbor generation, precomputing" begin
    codes = []
    for g in [GenericMessagePassing.random_tree(30), random_regular_graph(100, 3), SimpleGraph(GenericTensorNetworks.random_square_lattice_graph(30, 30, 0.8))]
        tn, code, tensors = GenericMessagePassing.ising_model(g, ones(nv(g)), ones(ne(g)), 1.0, verbose = true)
        push!(codes, code)
    end

    for problem in [problem_from_artifact("uai2014", "MAR", "Promedus", 14)]
        model = read_model(problem)
        tn = TensorNetworkModel(model; optimizer = GreedyMethod())
        push!(codes, tn.code)
    end

    for (ic, code) in enumerate(codes)
        for r in [2, 3, 4, 5]
            (ic == 3 && r == 5) && continue
            factor_graph = FactorGraph(code)
            neibs, boundaries = GenericMessagePassing.generate_neighborhoods(factor_graph, r)

            for v in keys(neibs)
                @test v ∈ neibs[v]
                @test issubset(boundaries[v], neibs[v])
                @test Set(GenericMessagePassing.open_boundaries(factor_graph, neibs[v])) == Set(boundaries[v])
                fgt = copy(factor_graph)
                GenericMessagePassing.isolate_vertices!(fgt, setdiff(neibs[v], boundaries[v]))
                GenericMessagePassing.isolate_pairs!(fgt, boundaries[v])
                for i in 1:length(boundaries[v]) - 1
                    for j in i+1:length(boundaries[v])
                        s, d = boundaries[v][i], boundaries[v][j]
                        path = a_star(fgt, s, d)
                        @test (length(path) == 0) || (length(path) - 1 > r)
                    end
                end

                for w in neibs[v]
                    for u in neighbors(factor_graph, w)
                        if length(neighbors(factor_graph, u)) == 1
                            @test u ∈ neibs[v]
                            @test u ∉ boundaries[v]
                        end
                    end
                end
            end
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
    bp_sol = marginal_tnbp(tn, TNBPConfig(verbose = true))
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
    bp_sol = marginal_tnbp(tn, TNBPConfig(verbose = true))
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
    bp_sol = marginal_tnbp(tn, TNBPConfig(verbose = true))
    for i in keys(bp_sol)
        @test isapprox(ti_sol[[i]][1], bp_sol[i][1], atol = 1e-4)
        @test isapprox(ti_sol[[i]][2], bp_sol[i][2], atol = 1e-4)
    end
end

@testset "marginal spinglass square-lattice" begin
    g = SimpleGraph(random_square_lattice_graph(20, 20, 0.8))
    h = ones(nv(g))
    J = -1 .* ones(ne(g))
    β = 1.0
    tn, code, tensors = GenericMessagePassing.ising_model(g, h, J, β, verbose = true)
    ti_sol = marginals(tn)
    bp_sol = marginal_tnbp(tn, TNBPConfig(verbose = true, r = 4, optimizer = TreeSA(sc_target = 20, ntrials = 1, niters = 5, βs = 0.1:0.1:100)))
    for i in keys(bp_sol)
        @test isapprox(ti_sol[[i]][1], bp_sol[i][1], atol = 1e-4)
        @test isapprox(ti_sol[[i]][2], bp_sol[i][2], atol = 1e-4)
    end
end

@testset "marginal uai2014 Promedus" begin
    # this problem is very baised, I am not sure is it a good test case
    problem = problem_from_artifact("uai2014", "MAR", "Promedus", 14)
    for rr in [2, 3, 4]
        tn = tn_model(problem)
        ti_sol = marginals(tn)

        for random_order in [false, true]
            tnbp_config = TNBPConfig(random_order = random_order, verbose = true, error = 1e-6, r = rr, optimizer = TreeSA(sc_target = 20, ntrials = 1, niters = 5, βs = 0.1:0.1:100))
            bp_sol = marginal_tnbp(tn, tnbp_config)
            for i in keys(bp_sol)
                @test isapprox(ti_sol[[i]][1], bp_sol[i][1], atol = 1e-3)
                @test isapprox(ti_sol[[i]][2], bp_sol[i][2], atol = 1e-3)
            end
        end
    end
end

@testset "marginal uai2014 ObjectDetection" begin
    for problem in [problem_from_artifact("uai2014", "MAR", "ObjectDetection", 42), problem_from_artifact("uai2014", "MAR", "ObjectDetection", 28)]
        for rr in [2, 3]
            tn = tn_model(problem)
            ti_sol = marginals(tn)

            random_order = false
            tnbp_config = TNBPConfig(random_order = random_order, verbose = true, error = 1e-12, r = rr, optimizer = TreeSA(sc_target = 20, ntrials = 1, niters = 5, βs = 0.1:0.1:100))
            bp_sol = marginal_tnbp(tn, tnbp_config)
            max_err = 0.0
            for i in keys(bp_sol)
                for j in 1:length(bp_sol[i])
                    @test isapprox(ti_sol[[i]][j], bp_sol[i][j], atol = 1e-2)
                    max_err = max(max_err, abs(ti_sol[[i]][j] - bp_sol[i][j]))
                end
            end
        end
    end
end