@kwdef struct BPConfig
    error::Float64 = 1e-6
    max_iter::Int = 10000
    random_order::Bool = true
    damping::Float64 = 0.2
    verbose::Bool = false
end

# factor graph, the first num_vars are the variables (the tensors), the rest are the factors (the indices)
struct FactorGraph{T}
    g::SimpleGraph{T}
    num_vars::Int
    function FactorGraph(g::SimpleGraph{T}, num_vars::Int) where T
        for e in edges(g)
            s, d = src(e), dst(e)
            # neighbor of a variable is a factor, and vice versa
            @assert ((s ≤ num_vars) && (d > num_vars)) || ((s > num_vars) && (d ≤ num_vars))
        end
        new{T}(g, num_vars)
    end
    function FactorGraph(hyper_graph::IncidenceList)
        num_vars = length(keys(hyper_graph.v2e))
        num_factors = length(keys(hyper_graph.e2v))
        g = SimpleGraph(num_vars + num_factors)
        for e in keys(hyper_graph.e2v)
            for v in hyper_graph.e2v[e]
                add_edge!(g, num_vars + e, v)
            end
        end
        FactorGraph(g, num_vars)
    end
end

Base.show(io::IO, fg::FactorGraph) = print(io, "FactorGraph{variables: $(fg.num_vars), factors: $(nv(fg.g) - fg.num_vars)}")
Base.copy(fg::FactorGraph) = FactorGraph(copy(fg.g), fg.num_vars)

Graphs.edges(fg::FactorGraph) = edges(fg.g)
Graphs.vertices(fg::FactorGraph) = vertices(fg.g)
Graphs.nv(fg::FactorGraph) = nv(fg.g)
Graphs.ne(fg::FactorGraph) = ne(fg.g)
Graphs.has_edge(fg::FactorGraph, s, d) = has_edge(fg.g, s, d)
Graphs.has_vertex(fg::FactorGraph, v) = has_vertex(fg.g, v)
Graphs.rem_edge!(fg::FactorGraph, s, d) = rem_edge!(fg.g, s, d)
Graphs.rem_vertex!(fg::FactorGraph, v) = rem_vertex!(fg.g, v)
Graphs.rem_vertices!(fg::FactorGraph, vs) = rem_vertices!(fg.g, vs)
Graphs.add_edge!(fg::FactorGraph, s, d) = add_edge!(fg.g, s, d)
Graphs.add_vertex!(fg::FactorGraph) = add_vertex!(fg.g)
Graphs.add_vertices!(fg::FactorGraph, n) = add_vertices!(fg.g, n)
Graphs.neighbors(fg::FactorGraph, v) = neighbors(fg.g, v)
is_factor(fg::FactorGraph, v) = v > fg.num_vars
is_variable(fg::FactorGraph, v) = v ≤ fg.num_vars

@kwdef struct TNBPConfig
    error::Float64 = 1e-6
    max_iter::Int = 10000
    random_order::Bool = true
    damping::Float64 = 0.2
    verbose::Bool = false
    r::Int = 5
end