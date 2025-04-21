# implementation of tensor network message passing, based on the following papers
# "Tensor Network Message Passing" 10.1103/PhysRevLett.132.117401
# "Counting the number of solutions in satisfiability problems with tensor-network message passing" 10.1103/PhysRevE.110.034126

# generate neighbor hoods of the vertices of a factor graph
# r is the shortest path between the boundary vertices of neighborhood after removing the vertices of the neighborhood in the factor graph
function generate_neighborhoods(fg::FactorGraph{T}, r::Int) where T
    neibs = Vector{Vector{Int}}()
    boundaries = Vector{Vector{Int}}()
    
    # for each vertex v in the factor graph, generate its neighborhood so that the shortest path between the boundary vertices of the neighborhood is r
    for v in vertices(fg)
        fgt = copy(fg)
        neib = vcat([v], neighbors(fg, v))
        boundary = open_boundaries(fg, neib)
        isolate_vertex!(fgt, v)
        isolate_pairs!(fgt, boundary)

        for rr in 1:r
            while true
                length(boundary) ≤ 1 && break
                flag = true
                for i in 1:length(boundary) - 1
                    for j in i + 1:length(boundary)
                        s = boundary[i]
                        d = boundary[j]

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
                    end
                    # neib is modified, break
                    !(flag) && break
                end

                boundary = open_boundaries(fg, neib)
                isolate_vertices!(fgt, setdiff(neib, boundary))
                isolate_pairs!(fgt, boundary)

                # if flag is true (all conditions satisfied), break
                flag && break
            end
        end
        push!(neibs, neib)
        push!(boundaries, boundary)
    end
    return neibs, boundaries
end

function precompute_eins(fg::FactorGraph, icode::TE, tensors::Vector{TA}, neibs::Vector{Vector{Int}}, boundaries::Vector{Vector{Int}}; optimizer::CodeOptimizer = GreedyMethod()) where {TA <: AbstractArray, TE <: AbstractEinsum}

    TT = eltype(TA)

    # the messages are defined from vertices on the factor graph to the neigborhoods
    messages = Dict{Tuple{Int, Int}, AbstractArray}()
    local_iys = Dict{Tuple{Int, Int}, Vector{Int}}()
    eins = Dict{Tuple{Int, Int}, AbstractEinsum}()
    ptensors = Dict{Tuple{Int, Int}, Vector{AbstractArray}}()

    ixs = getixsv(icode)
    size_dict = OMEinsum.get_size_dict(ixs, tensors)

    for v in vertices(fg)
        neib_v = neibs[v]
        boundary_v = boundaries[v]

        # check the eins of the messages, generate the message tensors
        for w in boundary_v
            # message from w to neighborhood of v
            if is_factor(fg, w)
                local_iy = [w - fg.num_vars]
            else
                local_iy = [u - fg.num_vars for u in neighbors(fg, w) if u ∈ neib_v]
            end
            local_iys[(w, v)] = local_iy
            t = rand(TT, [size_dict[y] for y in local_iy]...)
            messages[(w, v)] = t ./ sum(t)
        end
    end

    for v in vertices(fg)
        neib_v = neibs[v]
        boundary_v = boundaries[v]

        for w in boundary_v
            # neib_w / neib_v
            core_v = setdiff(neib_v, boundary_v)
            neib_w = neibs[w]
            boundary_w = boundaries[w]
            local_region = setdiff(neib_w, core_v)

            @show v, w
            @show neib_w, neib_v
            @show boundary_w, boundary_v
            @show core_v
            @show local_region

            local_tensors = Vector{AbstractArray}()
            local_ixs = Vector{Vector{Int}}()

            boundary_with_message = intersect(boundary_w, local_region)
            boundary_without_message = setdiff(open_boundaries(fg, local_region), boundary_with_message)

            @show boundary_with_message
            @show boundary_without_message
            
            for b in local_region
                @show b
                if b == w
                    if is_variable(fg, b)
                        push!(local_ixs, ixs[b])
                        @show ixs[b]
                        push!(local_tensors, tensors[b])
                    end
                elseif b ∈ boundary_with_message
                    push!(local_ixs, local_iys[(b, w)])
                    push!(local_tensors, messages[(b, w)])
                elseif b ∈ boundary_without_message
                    if is_factor(fg, b)
                        local_ix = [b - fg.num_vars]
                    else
                        local_ix = [u - fg.num_vars for u in neighbors(fg, b) if u ∈ local_region]
                    end
                    t = ones(TT, [size_dict[x] for x in local_ix]...)
                    push!(local_ixs, local_ix)
                    push!(local_tensors, t ./ sum(t))
                else
                    if is_variable(fg, b)
                        push!(local_ixs, ixs[b])
                        push!(local_tensors, tensors[b])
                    end
                end
            end

            eincode = EinCode(local_ixs, local_iys[(w, v)])

            @show eincode

            optcode = optimize_code(eincode, size_dict, optimizer)
            eins[(w, v)] = optcode
            ptensors[(w, v)] = local_tensors
        end
    end


    return messages, eins, ptensors
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