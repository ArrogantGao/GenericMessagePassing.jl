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