mutable struct Dense{F} <: Layer
    weight::M{Float32}
    bias::M{Float32}
    activation::F
    name::String
end

function Dense(in_dim::Int, out_dim::Int; activation=identity, name::String="dense")
    return Dense(glorot_uniform(out_dim, in_dim), zeros(Float32, out_dim, 1), activation, name)
end

function build(layer::Dense)
    trainable_layer = []
    for field in fieldnames(typeof(layer))
        x = getfield(layer, field) 
        if x isa M{Float32}
            push!(trainable_layer, (layer.name * "." * string(field), x))
        end
    end
    return trainable_layer
end

function (layer::Dense)(x::A{T}; training=false) where T <: AbstractFloat
    weight = T.(layer.weight)
    bias = T.(layer.bias)

    x_flat = reshape(x, (size(x,1),:))
    y_flat = weight * x_flat .+ bias

    y = reshape(y_flat, size(y_flat,1), size(x)[2:end]...)
    return layer.activation(y)
end

function gpu(layer::Dense)
    weight = cu(layer.weight)
    bias = cu(layer.bias)
    return Dense(weight, bias, layer.activation, layer.name)
end