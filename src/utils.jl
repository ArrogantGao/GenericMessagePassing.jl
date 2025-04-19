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
    tn = TensorNetworkModel(p, β, optimizer = TreeSA(ntrials = 1, niters = 5, βs = 0.1:0.1:100))
    ti_sol = marginals(tn)

    code = EinCode(getixsv(tn.code)[nv(g)+1:end], Int[])
    tensors = tn.tensors[nv(g)+1:end]
    size_dict = uniformsize(code, 2)

    bp_config = BPConfig(random_order = true, verbose = verbose, max_iter = 1000, error = 1e-12)
    bp_sol = message2marginals(bp(code, size_dict, tensors, bp_config)[1])
    return bp_sol, ti_sol
end