@kwdef struct BPConfig
    error::Float64 = 1e-6
    max_iter::Int = 10000
    random_order::Bool = true
    damping::Float64 = 0.2
    verbose::Bool = false
end

