# implementation of tensor network message passing, based on the following papers
# "Tensor Network Message Passing" 10.1103/PhysRevLett.132.117401
# "Counting the number of solutions in satisfiability problems with tensor-network message passing" 10.1103/PhysRevE.110.034126

# generate neighbor hoods of the vertices of a factor graph
# r is the shortest path between the boundary vertices of neighborhood after removing the vertices of the neighborhood in the factor graph
function generate_neighborhoods(fg::FactorGraph{T}, r::Int) where T
    neibs = Dict{Int, Vector{Int}}()
    boundaries = Dict{Int, Vector{Int}}()
    
    # for each vertex v in the factor graph, generate its neighborhood so that the shortest path between the boundary vertices of the neighborhood is r
    for v in vertices(fg)
        # if v is a rank-1 tensor, skip
        (is_factor(fg, v) && length(neighbors(fg, v)) ≤ 1) && continue

        fgt = copy(fg)
        neib = vcat([v], neighbors(fg, v))
        boundary = open_boundaries(fg, neib)
        isolate_vertex!(fgt, v)
        isolate_pairs!(fgt, boundary)

        for rr in 1:r
            safe_vertices = Set{Int}()
            safe_pairs = Set{Tuple{Int, Int}}()
            while true
                boundary = open_boundaries(fg, neib)
                for v in boundary
                    v ∈ safe_vertices && continue
                    for w in neighbors(fg, v)
                        if length(neighbors(fg, w)) == 1
                            push!(neib, w)
                        end
                    end
                    push!(safe_vertices, v)
                end

                unique!(neib)
                boundary = open_boundaries(fg, neib)
                isolate_vertices!(fgt, setdiff(neib, boundary))
                isolate_pairs!(fgt, boundary)

                length(boundary) ≤ 1 && break
                flag = true
                for i in 1:length(boundary) - 1
                    for j in i + 1:length(boundary)
                        (s, d) = minmax(boundary[i], boundary[j])
                        ((s, d) ∈ safe_pairs) && continue

                        path = a_star(fgt, s, d)
                        if !isempty(path) && length(path) - 1 ≤ rr
                            for p in path
                                push!(neib, p.dst)
                            end
                            unique!(neib)
                            # if meet condition not satisfied, set flag to false and break
                            flag = false
                            break
                        end
                        push!(safe_pairs, (s, d))
                    end
                    # neib is modified, break
                    !(flag) && break
                end
                # if flag is true (all conditions satisfied), break
                flag && break
            end
        end

        neibs[v] = neib
        boundaries[v] = open_boundaries(fg, neib)
    end
    return neibs, boundaries
end

function tnbp_precompute(fg::FactorGraph, icode::TE, tensors::Vector{TA}, neibs::Dict{Int, Vector{Int}}, boundaries::Dict{Int, Vector{Int}}, optimizer::CodeOptimizer) where {TA <: AbstractArray, TE <: AbstractEinsum}

    TT = eltype(TA)

    # the messages are defined from vertices on the factor graph to the neigborhoods
    messages = Dict{Tuple{Int, Int}, AbstractArray}()
    local_iys = Dict{Tuple{Int, Int}, Vector{Int}}()
    eins = Dict{Tuple{Int, Int}, AbstractEinsum}()
    ptensors = Dict{Tuple{Int, Int}, Vector{AbstractArray}}()

    ixs = getixsv(icode)
    size_dict = OMEinsum.get_size_dict(ixs, tensors)

    for v in keys(neibs)
        neib_v = neibs[v]
        boundary_v = boundaries[v]

        # check the eins of the messages, generate the message tensors
        for w in boundary_v
            # message from w to neighborhood of v
            if is_factor(fg, w)
                local_iy = [u for u in neighbors(fg, w) if u ∈ neib_v]
            else
                local_iy = [w]
            end
            local_iys[(w, v)] = local_iy
            t = rand(TT, [size_dict[y] for y in local_iy]...)
            messages[(w, v)] = t ./ sum(t)
        end
    end

    for v in keys(neibs)
        neib_v = neibs[v]
        boundary_v = boundaries[v]

        for w in boundary_v
            # neib_w / neib_v
            core_v = setdiff(neib_v, boundary_v)
            neib_w = neibs[w]
            boundary_w = boundaries[w]
            local_region = setdiff(neib_w, core_v)

            local_tensors = Vector{AbstractArray}()
            local_ixs = Vector{Vector{Int}}()

            boundary_with_message = intersect(boundary_w, local_region)
            boundary_without_message = setdiff(open_boundaries(fg, local_region), boundary_with_message)

            for b in local_region
                if b == w
                    if is_factor(fg, b)
                        push!(local_ixs, ixs[b - fg.num_vars])
                        push!(local_tensors, tensors[b - fg.num_vars])
                    end
                elseif b ∈ boundary_with_message
                    push!(local_ixs, local_iys[(b, w)])
                    push!(local_tensors, messages[(b, w)])
                elseif b ∈ boundary_without_message
                    if is_factor(fg, b)
                        local_ix = [u for u in neighbors(fg, b) if u ∈ local_region]
                    else
                        local_ix = [b]
                    end
                    t = ones(TT, [size_dict[x] for x in local_ix]...)
                    push!(local_ixs, local_ix)
                    push!(local_tensors, t ./ sum(t))
                else
                    if is_factor(fg, b)
                        push!(local_ixs, ixs[b - fg.num_vars])
                        push!(local_tensors, tensors[b - fg.num_vars])
                    end
                end
            end

            eincode = EinCode(local_ixs, local_iys[(w, v)])
            optcode = @suppress optimize_code(eincode, size_dict, optimizer)
            eins[(w, v)] = optcode
            ptensors[(w, v)] = local_tensors
        end
    end

    # for marginal probability calculation
    mars_eins = Dict{Int, AbstractEinsum}()
    mars_tensors = Dict{Int, Vector{AbstractArray}}()
    for v in 1:fg.num_vars
        local_ixs = Vector{Vector{Int}}()
        local_iy = [v]
        local_tensors = Vector{AbstractArray}()
        for w in neibs[v]
            if w ∈ boundaries[v]
                push!(local_ixs, local_iys[(w, v)])
                push!(local_tensors, messages[(w, v)])
            else
                if is_factor(fg, w)
                    push!(local_ixs, ixs[w - fg.num_vars])
                    push!(local_tensors, tensors[w - fg.num_vars])
                end
            end
        end
        eincode = EinCode(local_ixs, local_iy)
        optcode = @suppress optimize_code(eincode, size_dict, optimizer)
        mars_eins[v] = optcode
        mars_tensors[v] = local_tensors
    end

    return messages, eins, ptensors, mars_eins, mars_tensors
end

function tnbp_update!(messages::Dict{Tuple{Int, Int}, TA}, eins::Dict{Tuple{Int, Int}, TE}, ptensors::Dict{Tuple{Int, Int}, Vector{TA}}, tnbp_config::TNBPConfig) where {TA <: AbstractArray, TE <: AbstractEinsum}

    t_order = collect(keys(messages))
    order = tnbp_config.random_order ? t_order[sortperm(rand(length(t_order)))] : t_order
        
    error_max = 0.0

    for (w, v) in order
        t = eins[(w, v)](ptensors[(w, v)]...)
        t = t ./ sum(t)
        error_t = maximum(abs.(t - messages[(w, v)]))
        error_max = max(error_max, error_t)

        # these inplace update will change ptensors at the same time
        messages[(w, v)] .*= tnbp_config.damping
        messages[(w, v)] .+= (1 - tnbp_config.damping) .* t
    end

    return error_max
end

function marginal_tnbp(code::AbstractEinsum, tensors::Vector{TA}, tnbp_config::TNBPConfig) where {TA <: AbstractArray}
    
    icode, idict = intcode(code)
    ixs = getixsv(icode)
    iy = getiyv(icode)

    hyper_graph = IncidenceList(Dict([i=>ix for (i, ix) in enumerate(ixs)]), openedges = iy)
    factor_graph = FactorGraph(hyper_graph)

    neibs, boundaries = generate_neighborhoods(factor_graph, tnbp_config.r)

    if tnbp_config.verbose
        println("--------------------------------")
        println("average size of neibs: $(mean(length.(values(neibs))))")
        println("maximum size of neibs: $(maximum(length.(values(neibs))))")
        println("--------------------------------")
    end

    messages, eins, ptensors, mars_eins, mars_tensors = tnbp_precompute(factor_graph, icode, tensors, neibs, boundaries, tnbp_config.optimizer)

    if tnbp_config.verbose
        size_dict = OMEinsum.get_size_dict(ixs, tensors)
        sc_max = 0
        tc_total = 0.0
        for (w, v) in keys(eins)
            cc = contraction_complexity(eins[(w, v)], size_dict)
            sc_max = max(sc_max, cc.sc)
            tc_total += 2^cc.tc
        end
        println("maximum size of sc: $sc_max")
        println("total contraction cost: $(log2(tc_total))")
        println("--------------------------------")
    end

    # tnbp update
    for i in 1:tnbp_config.max_iter
        error_max = tnbp_update!(messages, eins, ptensors, tnbp_config)
        tnbp_config.verbose && println("iter $i: error_max = $error_max")
        error_max < tnbp_config.error && break
    end

    # calculate the marginal probability of the variables
    mars = Dict()
    for v in 1:factor_graph.num_vars
        t = mars_eins[v](mars_tensors[v]...)
        t = t ./ sum(t)
        mars[idict[v]] = t
    end
            
    return mars
end
