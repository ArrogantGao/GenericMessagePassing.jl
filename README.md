# GenericMessagePassing

<!-- [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ArrogantGao.github.io/GenericMessagePassing.jl/stable/) -->
<!-- [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ArrogantGao.github.io/GenericMessagePassing.jl/dev/) -->
[![Build Status](https://github.com/ArrogantGao/GenericMessagePassing.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ArrogantGao/GenericMessagePassing.jl/actions/workflows/CI.yml?query=branch%3Amain)
<!-- [![Coverage](https://codecov.io/gh/ArrogantGao/GenericMessagePassing.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ArrogantGao/GenericMessagePassing.jl) -->

**This package is under development.**

Target of this package is to build up a generic framework for message passing algorithms based on tensor networks.

## Usage

Here is an example of solving a inference problem from the uai2014 dataset.
```julia
julia> using GenericMessagePassing

# TensorInference provides models and exact inference based on tensor networks contraction
julia> using TensorInference

# load a problem from the uai2014 dataset
julia> problem = problem_from_artifact("uai2014", "MAR", "ObjectDetection", 42);
       
# initialize the model as a tensor network
julia> tn = tn_model(problem);
       
# run the inference by belief propagation
julia> bp_sol = marginal_bp(tn, BPConfig(verbose = false));
       
# run the inference by tensor network message passing
julia> tnbp_sol = marginal_tnbp(tn, TNBPConfig(verbose = false, r = 3, optimizer = TreeSA(sc_target = 20, ntrials = 1, niters = 5, βs = 0.1:0.1:100)));
--------------------------------
average size of neibs: 145.73777777777778
maximum size of neibs: 218
--------------------------------
maximum size of sc: 24.21602133046108
total contraction cost: 31.816622083010046
--------------------------------

# for small problems, one can generate reference solution by exact tn contractions
julia> ti_sol = marginals(tn);

ulia> @show ti_sol[[1]];
ti_sol[[1]] = [0.0, 0.2654933645896844, 0.1419587997079056, 0.1495284508172771, 0.10910602300771183, 0.10705297002673668, 0.07556569719292902, 0.04701197229900054, 0.03515919421147721, 8.833252881681324e-5, 0.06903519561846086]

julia> @show bp_sol[1];
bp_sol[1] = [0.0, 0.26606070883455224, 0.14160702801058442, 0.1494109445995478, 0.1092919826423247, 0.1073224746724277, 0.07559511369377542, 0.04659493434397761, 0.034818155487839765, 8.708748726390353e-5, 0.06921157022770634]

julia> @show tnbp_sol[1];
tnbp_sol[1] = [0.0, 0.2654933629893131, 0.1419588020020375, 0.14952844824880088, 0.10910602391156107, 0.10705297000684023, 0.07556569603555476, 0.047011973489407576, 0.035159195195138784, 8.833252807940897e-5, 0.06903519559326682]

julia> max_err_bp = 0.0;

julia> max_err_tnbp = 0.0;

julia> for i in keys(bp_sol)
           for j in 1:length(bp_sol[i])
               max_err_bp = max(max_err_bp, abs(bp_sol[i][j] - ti_sol[[i]][j]))
               max_err_tnbp = max(max_err_tnbp, abs(tnbp_sol[i][j] - ti_sol[[i]][j]))
           end
       end

# maximum error of bp and tnbp compared to the exact solution, clearly tnbp is more accurate
julia> max_err_bp, max_err_tnbp
(0.0023346724101498094, 1.7785191580545895e-5)
```

## Current status

### Implemented algorithms

- tensor network based classical belief propagation
- SAT solver based on belief propagation
- tensor network message passing (tnmp) for marginal inference (yijia's work [url](https://doi.org/10.1103/PhysRevLett.132.117401))

### TODO

- tensor network based warning propagation (related to tropical algebra)
- tensor network based survey propagation (how?)

### Possible usage

- tnmp for gauging graph tensor network state