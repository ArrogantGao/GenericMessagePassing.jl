function random_tree(n::Int)
    tree_g = SimpleGraph(n)
    for i in 2:n
        add_edge!(tree_g, i, rand(1:i - 1))
    end
    return tree_g
end

function ising_model(g::SimpleGraph, h::Vector{T}, J::Vector{T}, β::Float64; verbose::Bool = false) where T<:Number
    p = SpinGlass(g, J, h)
    tn = TensorNetworkModel(p, β, optimizer = TreeSA(sc_target = 20, ntrials = 1, niters = 5, βs = 0.1:0.1:100))

    code = EinCode(getixsv(tn.code)[nv(g)+1:end], Int[])
    tensors = tn.tensors[nv(g)+1:end]

    return tn, code, tensors
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