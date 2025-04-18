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
    # 1. from tensors to indices: messages_v2e
    messages_v2e = Dict{Tuple{Int, TL}, Vector{TT}}()
    # random initialize
    for e in ids
        for v in factor_graph.e2v[e]
            messages_v2e[(v, e)] = rand(TT, size_dict[e])
        end
    end

    # 2. from indices to tensors: messages_e2v, initialize with the message_v2e value to avoid repeated calculation
    messages_e2v = Dict{Tuple{TL, Int}, Vector{TT}}()
    for e in ids
        for v in factor_graph.e2v[e]
            t = ones(TT, size_dict[e])
            for n in factor_graph.e2v[e]
                n != v && (t .*= messages_v2e[(n, e)])
            end
            # normalize
            messages_e2v[(e, v)] = t ./ sum(t)
        end
    end

    # bp
    for i in 1:bp_config.max_iter
        error_max_v2e, error_max_e2v = bp_update!(factor_graph, tensors, ixs, size_dict, messages_e2v, messages_v2e, bp_config)
        bp_config.verbose && println("iter $i: error_max_v2e = $error_max_v2e, error_max_e2v = $error_max_e2v")
        (error_max_e2v < bp_config.error && error_max_v2e < bp_config.error) && break
    end

    # normalize the messages_v2e
    for (v, e) in keys(messages_v2e)
        messages_v2e[(v, e)] ./= sum(messages_v2e[(v, e)])
    end

    return messages_v2e, messages_e2v
end

function bp_update!(factor_graph::IncidenceList, tensors::Vector{TA}, ixs::Vector{Vector{TL}}, size_dict::Dict{TL, Int}, messages_e2v::Dict{Tuple{TL, Int}, Vector{TT}}, messages_v2e::Dict{Tuple{Int, TL}, Vector{TT}}, bp_config::BPConfig) where {TL, TT<:Number, TA<:AbstractArray}
    t = collect(keys(messages_v2e))
    order = bp_config.random_order ? t[sortperm(rand(length(t)))] : t

    error_max_v2e = 0.0
    error_max_e2v = 0.0

    for (v, e) in order
        # update messages_v2e according to messages_e2v
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
        nested_code = optimize_code(rawcode, size_dict, TreeSA(; ntrials=1, niters=10)).eins
        t = nested_code(local_tensors...)

        # update error
        error = maximum(abs.(messages_v2e[(v, e)] .- t) ./ abs.(messages_v2e[(v, e)]))
        error_max_v2e = max(error_max_v2e, error)

        # normalize
        messages_v2e[(v, e)] = t
    end

    # update messages_e2v according to the updated messages_v2e
    for e in unique(vcat(ixs...))
        for n in factor_graph.e2v[e]
            t = ones(TT, size_dict[e])
            for m in factor_graph.e2v[e]
                m != n && (t .*= messages_v2e[(m, e)])
            end
            t ./= sum(t)
            error = maximum(abs.(messages_e2v[(e, n)] .- t) ./ abs.(messages_e2v[(e, n)]))
            error_max_e2v = max(error_max_e2v, error)
            messages_e2v[(e, n)] = t
        end
    end

    return error_max_v2e, error_max_e2v
end