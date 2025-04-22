# unified operations for generic element types
# one can extend this file to support more element types

# the following functions have to be defined
function uniform(::Type{T}, s) where {T}
    @error "uniform is not defined for $(T)"
end
function normalize!(x::T) where {T}
    @error "normalize! is not defined for $(T)"
end
function damp!(old::Array{T}, new::Array{T}, rate) where {T}
    @error "damp! is not defined for $(T)"
end
function abs_error(old::Array{T}, new::Array{T}) where {T}
    @error "error is not defined for $(T)"
end

# operations for float types
function uniform(::Type{T}, s::TI) where {T <: AbstractFloat, TI <: Integer}
    return ones(T, s) ./ s
end
function uniform(::Type{T}, s::Vector{TI}) where {T <: AbstractFloat, TI <: Integer}
    return ones(T, s...) ./ prod(s)
end
function normalize!(x::Array{T}) where {T <: AbstractFloat}
    return x ./= sum(x)
end
function damp!(old::Array{T}, t::Array{T}, rate) where {T <: AbstractFloat}
    return old .= old .* rate .+ t .* (1 - rate)
end
function abs_error(t1::Array{T}, t2::Array{T}) where {T <: AbstractFloat}
    return maximum(abs.(t1 .- t2))
end