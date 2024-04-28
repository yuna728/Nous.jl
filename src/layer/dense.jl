mutable struct Dense <: NLayer
    weights::AbstractMatrix
    bias::AbstractMatrix
end

function Dense(in_size::Int, out_size::Int)
    return Dense(randn(in_size, out_size), zeros(1, out_size))
end

function build(layer::Dense)
    return [layer.weights, layer.bias]
end

function forward(layer::Dense, x)
    return x * layer.weights .+ layer.bias
end
