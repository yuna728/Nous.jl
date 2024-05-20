mutable struct Dense <: Layer
    weight::M{Float32}
    bias::M{Float32}
    name::String
end

function Dense(in_dim::Int, out_dim::Int; activation=identity, name::String="dense")
    return Dense(glorot_uniform(out_dim, in_dim), zeros(Float32, out_dim, 1), name)
end

function build(layer::Dense)
    trainable_layer = []
    for field in fieldnames(typeof(layer))
        x = getfield(layer, field) 
        if x isa M{Float32}
            push!(trainable_layer, (layer.name * "." * field, x))
        end
    end
    return trainable_layer
end

function (layer::Dense)(x::A{T}; training=false) where T <: AbstractFloat
    weight = T.(layer.weight)
    bias = T.(layer.bias)
    return weight * x .+ bias
end

function gpu(layer::Dense)
    weight = CuArray(layer.weight)
    bias = CuArray(layer.bias)
    return Dense(weight, bias, layer.name)
end