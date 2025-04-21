function message2marginals(messages_v2e::Dict{Tuple{Int, TL}, Vector{TT}}) where {TL, TT<:Number}

    marginals = Dict{TL, Vector{TT}}()
    for (v, e) in collect(keys(messages_v2e))
        haskey(marginals, e) ? marginals[e] .*= messages_v2e[(v, e)] : marginals[e] = messages_v2e[(v, e)]
    end

    for e in keys(marginals)
        marginals[e] ./= sum(marginals[e])
    end

    return marginals
end

function random_tree(n::Int)
    tree_g = SimpleGraph(n)
    for i in 2:n
        add_edge!(tree_g, i, rand(1:i - 1))
    end
    return tree_g
end

function marginal_ising(g::SimpleGraph, h::Vector{T}, J::Vector{T}, β::Float64; verbose::Bool = false) where T<:Number
    p = SpinGlass(g, J, h)
    tn = TensorNetworkModel(p, β, optimizer = TreeSA(sc_target = 20, ntrials = 1, niters = 5, βs = 0.1:0.1:100))
    ti_sol = marginals(tn)

    code = EinCode(getixsv(tn.code)[nv(g)+1:end], Int[])
    tensors = tn.tensors[nv(g)+1:end]

    bp_config = BPConfig(random_order = true, verbose = verbose, max_iter = 5000, error = 1e-8)
    bp_sol = message2marginals(bp(code, tensors, bp_config)[1])
    return bp_sol, ti_sol
end

function marginal_ising_bp(g::SimpleGraph, h::Vector{T}, J::Vector{T}, β::Float64; verbose::Bool = false) where T<:Number
    p = SpinGlass(g, J, h)
    tn = TensorNetworkModel(p, β)

    code = EinCode(getixsv(tn.code)[nv(g)+1:end], Int[])
    tensors = tn.tensors[nv(g)+1:end]

    bp_config = BPConfig(random_order = true, verbose = verbose, max_iter = 5000, error = 1e-8)
    bp_sol = message2marginals(bp(code, tensors, bp_config)[1])
    return bp_sol
end

function intcode(code::TC) where TC <: AbstractEinsum
    ids = uniquelabels(code)
    dict = Dict(zip(ids, 1:length(ids)))
    idict = Dict(zip(1:length(ids), ids))
    ixs = getixsv(code)
    iy = getiyv(code)
    icode = EinCode([[dict[ixi] for ixi in ix] for ix in ixs], [dict[iyi] for iyi in iy])
    return icode, idict
end

# in this function, we assume that the vertices of the factor graph are represented by integers from 1 to n
function linegraph(fg::IncidenceList{Int, Int})
    lg = SimpleGraph(length(keys(fg.e2v)))
    for e in keys(fg.e2v)
        for v in fg.e2v[e]
            for w in fg.v2e[v]
                (e != w) && add_edge!(lg, e, w)
            end
        end
    end
    return lg
end

function open_neighbors(g::Union{SimpleGraph, FactorGraph}, vs::Vector{Int})
    neis = Vector{Int}()
    for v in vs
        for w in neighbors(g, v)
            push!(neis, w)
        end
    end
    return setdiff(unique!(neis), vs)
end

function open_boundaries(g::Union{SimpleGraph, FactorGraph}, vs::Vector{Int})
    boundaries = Vector{Int}()
    for v in vs
        if !isempty(setdiff(neighbors(g, v), vs))
            push!(boundaries, v)
        end
    end
    return boundaries
end

function isolate_vertex!(fg::FactorGraph, v)
    ws = copy(neighbors(fg, v))
    for w in ws
        rem_edge!(fg, v, w)
    end
    return true
end
function isolate_vertices!(fg::FactorGraph, vs)
    for v in vs
        isolate_vertex!(fg, v)
    end
    return true
end
function isolate_pairs!(fg::FactorGraph, vs)
    for i in 1:length(vs) - 1, j in i + 1:length(vs)
        if has_edge(fg, vs[i], vs[j])
            rem_edge!(fg, vs[i], vs[j])
        end
    end
    return true
end

Graphs.a_star(fg::FactorGraph, s, d) = a_star(fg.g, s, d)