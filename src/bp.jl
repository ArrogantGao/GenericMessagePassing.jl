# given a tensor network, solve the message via bp
# notice that this is pure bp, did not use tn for update

using OMEinsumContractionOrders: IncidenceList

function bp(code::AbstractEinsum, size_dict::Dict{TL, Int}, tensors::Vector{TA}, bp_config::BPConfig) where {TL, TA<:AbstractArray}
    TT = eltype(TA)

    ixs = getixsv(code)
    iy = getiyv(code)
    ids = uniquelabels(code)
    # factor graph, e are the indices, v are the tensors
    factor_graph = IncidenceList(Dict([i=>ix for (i, ix) in enumerate(ixs)]), openedges = iy)

    # initialize messages
    # 1. from indices to tensors: messages_e2v
    
    messages_e2v = Dict{Tuple{TL, Int}, Vector{TT}}()
    # random initialize
    for e in ids
        for v in factor_graph.e2v[e]
            messages_e2v[(e, v)] = rand(TT, size_dict[e])
            messages_e2v[(e, v)] ./= sum(messages_e2v[(e, v)])
        end
    end

    # bp
    for i in 1:bp_config.max_iter
        error_max_e2v = bp_update!(factor_graph, tensors, ixs, size_dict, messages_e2v, bp_config)
        bp_config.verbose && println("iter $i: error_max_e2v = $error_max_e2v")
        error_max_e2v < bp_config.error && break
    end

    # 1. from tensors to indices: messages_v2e
    messages_v2e = Dict{Tuple{Int, TL}, Vector{TT}}()
    # update messages_v2e according to the updated messages_e2v
    for (e, v) in keys(messages_e2v)
        local_tensors = Vector{AbstractArray{TT}}()
        local_ixs = Vector{Vector{TL}}()
        local_iy = [e]
        
        push!(local_tensors, tensors[v])
        push!(local_ixs, ixs[v])

        for n in factor_graph.v2e[v]
            if n != e
                push!(local_tensors, messages_e2v[(n, v)])
                push!(local_ixs, [n])
            end
        end

        # contract the local tensors
        rawcode = EinCode(local_ixs, local_iy)
        nested_code = optimize_code(rawcode, size_dict, GreedyMethod())
        t = nested_code(local_tensors...)

        # normalize
        messages_v2e[(v, e)] = t ./ sum(t)
    end

    return messages_v2e, messages_e2v
end

function bp_update!(factor_graph::IncidenceList, tensors::Vector{TA}, ixs::Vector{Vector{TL}}, size_dict::Dict{TL, Int}, messages_e2v::Dict{Tuple{TL, Int}, Vector{TT}}, bp_config::BPConfig) where {TL, TT<:Number, TA<:AbstractArray}
    t = collect(keys(messages_e2v))
    order = bp_config.random_order ? t[sortperm(rand(length(t)))] : t

    error_max_v2e = 0.0
    error_max_e2v = 0.0

    for (e, v) in order
        # update messages_e2v
        local_tensors = Vector{AbstractArray{TT}}()
        local_ixs = Vector{Vector{TL}}()
        local_iy = [e]
        
        for vp in factor_graph.e2v[e]
            if vp != v
                # inlcude the other tensor connected to e
                push!(local_tensors, tensors[vp])
                push!(local_ixs, ixs[vp])

                # include the message from vp to e
                for ep in factor_graph.v2e[vp]
                    if ep != e
                        push!(local_ixs, [ep])
                        push!(local_tensors, messages_e2v[(ep, vp)])
                    end
                end
            end
        end

        # contract the local tensors
        rawcode = EinCode(local_ixs, local_iy)
        nested_code = optimize_code(rawcode, size_dict, GreedyMethod())
        t = nested_code(local_tensors...)

        # normalize
        t ./= sum(t)

        # update error
        # error = maximum(abs.(messages_e2v[(e, v)] .- t) ./ abs.(messages_e2v[(e, v)]))
        error = maximum(abs.(t .- messages_e2v[(e, v)]))
        error_max_e2v = max(error_max_e2v, error)

        messages_e2v[(e, v)] = t
    end

    return error_max_e2v
end