using Random
using Statistics
using Graphs
using GenericMessagePassing
using TensorInference

function err_metrics(sol, ref)
    max_abs = 0.0
    mae = 0.0
    tv = 0.0
    nvars = 0
    nstates = 0
    for i in sort(collect(keys(sol)))
        p = sol[i]
        q = ref[[i]]
        d = abs.(p .- q)
        max_abs = max(max_abs, maximum(d))
        mae += sum(d)
        tv += 0.5 * sum(d)
        nvars += 1
        nstates += length(p)
    end
    return (max_abs=max_abs, mean_abs=mae/nstates, mean_tv=tv/nvars)
end

n = 150
degree = 3
β = 1.0
seed = 1234

Random.seed!(seed)
g = random_regular_graph(n, degree)
h = ones(nv(g))
J = -1 .* ones(ne(g))
tn, code, tensors = GenericMessagePassing.ising_model(g, h, J, β, verbose = false)

refs = marginals(tn)

bp_cfg = BPConfig(verbose = false, random_order = false)
# Warmup (compile + first run effects)
bp_mars = marginal_bp(code, tensors, bp_cfg)
err_metrics(bp_mars, refs)

# (max_abs = 6.1269634136554545e-6, mean_abs = 1.4323184810885062e-7, mean_tv = 1.4323184810885062e-7)

r = 7
tnbp_cfg = TNBPConfig(verbose = true, random_order = false, r = r)
tnbp_mars = marginal_tnbp(code, tensors, tnbp_cfg)
err_metrics(tnbp_mars, refs)

# r = 7
# (max_abs = 4.802298558814755e-10, mean_abs = 1.4760213425200204e-10, mean_tv = 1.4760213425200204e-10)

# r = 9
# (max_abs = 1.1473122452088091e-10, mean_abs = 3.0145538705077306e-11, mean_tv = 3.0145538705077306e-11)