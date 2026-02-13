# given a tensor network, solve the message via bp
# notice that this is pure bp, did not use tn for update

function bp_precompute_plans(
    hyper_graph::IncidenceList,
    ixs::Vector{Vector{Int}},
    ids::Vector{Int},
    size_dict::Dict{Int, Int};
    optimizer::CodeOptimizer = GreedyMethod(),
)
    e2v_plans = Dict{Tuple{Int, Int}, NamedTuple{(:eins, :arg_keys), Tuple{AbstractEinsum, Vector{Tuple{Int, Int}}}}}()
    v2e_plans = Dict{Tuple{Int, Int}, NamedTuple{(:eins, :tensor_v, :msg_keys), Tuple{AbstractEinsum, Int, Vector{Tuple{Int, Int}}}}}()

    for e in ids, v in hyper_graph.e2v[e]
        local_ixs = Vector{Vector{Int}}()
        arg_keys = Tuple{Int, Int}[]

        for vp in hyper_graph.e2v[e]
            if vp != v
                push!(local_ixs, ixs[vp])
                push!(arg_keys, (vp, 0))
                for ep in hyper_graph.v2e[vp]
                    if ep != e
                        push!(local_ixs, [ep])
                        push!(arg_keys, (ep, vp))
                    end
                end
            end
        end

        rawcode = EinCode(local_ixs, [e])
        e2v_plans[(e, v)] = (eins = optimize_code(rawcode, size_dict, optimizer), arg_keys = arg_keys)
    end

    for e in ids, v in hyper_graph.e2v[e]
        local_ixs = Vector{Vector{Int}}()
        msg_keys = Tuple{Int, Int}[]

        push!(local_ixs, ixs[v])
        for n in hyper_graph.v2e[v]
            if n != e
                push!(local_ixs, [n])
                push!(msg_keys, (n, v))
            end
        end

        rawcode = EinCode(local_ixs, [e])
        v2e_plans[(v, e)] = (eins = optimize_code(rawcode, size_dict, optimizer), tensor_v = v, msg_keys = msg_keys)
    end

    return e2v_plans, v2e_plans
end

# return the normalized messages
function bp(hyper_graph::IncidenceList, ixs::Vector{Vector{Int}}, iy::Vector{Int}, ids::Vector{Int}, size_dict::Dict{Int, Int}, tensors::Vector{Array{TT}}, bp_config::BPConfig) where {TT<:Number}
    # initialize messages
    # 1. from indices to tensors: messages_e2v
    messages_e2v = Dict{Tuple{Int, Int}, Vector{TT}}()
    # random initialize
    for e in ids
        for v in hyper_graph.e2v[e]
            # (length(ixs[v]) == 1) && continue # message to a 1d tensor is not needed
            messages_e2v[(e, v)] = uniform(TT, size_dict[e]) # init with uniform vector
            normalize!(messages_e2v[(e, v)])
        end
    end
    e2v_plans, v2e_plans = bp_precompute_plans(hyper_graph, ixs, ids, size_dict)

    # bp
    for i in 1:bp_config.max_iter
        error_max_e2v = bp_update!(tensors, messages_e2v, e2v_plans, bp_config)
        bp_config.verbose && println("iter $i: error_max_e2v = $error_max_e2v")
        error_max_e2v < bp_config.error && break
    end

    # 1. from tensors to indices: messages_v2e
    messages_v2e = Dict{Tuple{Int, Int}, Vector{TT}}()
    # update messages_v2e according to the updated messages_e2v
    for key in keys(v2e_plans)
        plan = v2e_plans[key]
        local_tensors = Vector{AbstractArray{TT}}(undef, length(plan.msg_keys) + 1)
        local_tensors[1] = tensors[plan.tensor_v]
        for (i, msg_key) in enumerate(plan.msg_keys)
            local_tensors[i + 1] = messages_e2v[msg_key]
        end
        messages_v2e[key] = normalize!(plan.eins(local_tensors...))
    end

    return messages_e2v, messages_v2e
end

function bp_update!(tensors::Vector{Array{TT}}, messages_e2v::Dict{Tuple{Int, Int}, Vector{TT}}, e2v_plans::Dict{Tuple{Int, Int}, NamedTuple{(:eins, :arg_keys), Tuple{AbstractEinsum, Vector{Tuple{Int, Int}}}}}, bp_config::BPConfig) where {TT<:Number}
    t = collect(keys(messages_e2v))
    order = bp_config.random_order ? t[sortperm(rand(length(t)))] : t

    error_max_e2v = 0.0

    for (e, v) in order
        plan = e2v_plans[(e, v)]
        local_tensors = Vector{AbstractArray{TT}}(undef, length(plan.arg_keys))
        for (i, key) in enumerate(plan.arg_keys)
            if key[2] == 0
                local_tensors[i] = tensors[key[1]]
            else
                local_tensors[i] = messages_e2v[key]
            end
        end

        t = plan.eins(local_tensors...)
        normalize!(t)

        # update error
        error_t = abs_error(messages_e2v[(e, v)], t)
        error_max_e2v = max(error_max_e2v, error_t)

        # damping
        damp!(messages_e2v[(e, v)], t, bp_config.damping)

        messages_e2v[(e, v)] = t
    end

    return error_max_e2v
end

function marginal_bp(tn::TensorNetworkModel, bp_config::BPConfig)
    code = tn.code isa SlicedEinsum ? tn.code.eins : tn.code
    tensors = tn.tensors
    return marginal_bp(code, tensors, bp_config)
end

# use the bp message to compute the marginal distribution
function marginal_bp(code::AbstractEinsum, tensors::Vector{Array{TT}}, bp_config::BPConfig) where {TT<:Number}
    icode, idict = intcode(code)

    ixs = getixsv(icode)
    iy = getiyv(icode)
    ids = uniquelabels(icode)
    size_dict = OMEinsum.get_size_dict(ixs, tensors)

    # hyper graph, e are the indices, v are the tensors
    hyper_graph = IncidenceList(Dict([i=>ix for (i, ix) in enumerate(ixs)]), openedges = iy)

    messages_e2v, messages_v2e = bp(hyper_graph, ixs, iy, ids, size_dict, tensors, bp_config)

    marginals = Dict{Int, Vector{TT}}()
    new_marginals = Dict()

    for (v, e) in collect(keys(messages_v2e))
        haskey(marginals, e) ? marginals[e] .*= messages_v2e[(v, e)] : marginals[e] = messages_v2e[(v, e)]
    end

    for e in keys(marginals)
        normalize!(marginals[e])
        new_marginals[idict[e]] = marginals[e]
    end

    return new_marginals
end

# implemented according to "The Nature of Computation", page 777, Eq (14.71) and (14.72)
function entropy_bp(code::AbstractEinsum, tensors::Vector{Array{TT}}, bp_config::BPConfig) where {TT<:Number}
    icode, idict = intcode(code)
    ixs = getixsv(icode)
    iy = getiyv(icode)
    ids = uniquelabels(icode)
    size_dict = OMEinsum.get_size_dict(ixs, tensors)

    # hyper graph, e are the indices, v are the tensors
    hyper_graph = IncidenceList(Dict([i=>ix for (i, ix) in enumerate(ixs)]), openedges = iy)

    messages_e2v, messages_v2e = bp(hyper_graph, ixs, iy, ids, size_dict, tensors, bp_config)

    z_v = Dict{Int, TT}()
    z_e = Dict{Int, TT}()
    z_ve = Dict{Tuple{Int, Int}, TT}()

    # entropy calcuation, no need to normalize
    for (v, e) in collect(keys(messages_v2e))
        z_ve[(v, e)] = log(dot(messages_e2v[(e, v)], (messages_v2e[(v, e)])))
    end

    for e in ids
        t = ones(TT, size_dict[e])
        for v in hyper_graph.e2v[e]
            t .*= messages_v2e[(v, e)]
        end
        z_e[e] = sum(t)
    end

    for v in 1:length(tensors)
        t_ixs = [ixs[v]]
        for ei in hyper_graph.v2e[v]
            push!(t_ixs, [ei])
        end

        t_code = EinCode(t_ixs, Int[])
        z_v[v] = t_code(tensors[v], [messages_e2v[(ei, v)] for ei in hyper_graph.v2e[v]]...)[]
    end

    S = zero(TT) # entropy
    for v in collect(keys(z_v))
        S += log(z_v[v])
    end

    for e in collect(keys(z_e))
        S += log(z_e[e])
    end

    for (v, e) in collect(keys(z_ve))
        S -= z_ve[(v, e)]
    end

    return S
end
