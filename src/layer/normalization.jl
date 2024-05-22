using Statistics

mutable struct LayerNormalization <: Layer
    beta::V{Float32}
    gamma::V{Float32}
    eps::Float32
    name::String
end

function LayerNormalization(in_dim::Int; epsilon=1f-3, name::String="layer_norm")
    beta = zeros(Float32, in_dim)
    gamma = ones(Float32, in_dim)
    return LayerNormalization(beta, gamma, Float32(epsilon), name)
end

function build(layer::LayerNormalization)
    trainable_layer = []
    for field in fieldnames(typeof(layer))
        x = getfield(layer, field) 
        if x isa V{Float32}
            push!(trainable_layer, (layer.name * "." * string(field), x))
        end
    end
    return trainable_layer
end

function (layer::LayerNormalization)(x::A{T}; training=false) where T <: AbstractFloat
    if T == Float16
        x = Float32.(x)
    end
    mean_x = mean(x, dims=1)
    var_x = var(x, dims=1)
    inv = 1.0f0 ./ sqrt.(var_x .+ fill(layer.eps, size(var_x)))
    inv = inv .* layer.gamma
    res = -mean_x .* inv
    res = res .+ layer.beta
    x_norm = x .* inv .+ res
    return T.(x_norm)
end

function gpu(layer::LayerNormalization)
    member_list = []
    for field in fieldnames(typeof(layer))
        x = getfield(layer, field) 
        push!(member_list, CuArray(x))
    end
    return LayerNormalization(member_list..., layer.name)
end

mutable struct BatchNormalization <: Layer
    beta::V{Float32}
    gamma::V{Float32}
    momentum::Float32
    eps::Float32
    moving_mean::V{Float32}
    moving_variance::V{Float32}
    name::String
end

function BatchNormalization(in_dim::Int; momentum=0.99f0, eps=1f-3, name::String="batch_norm")
    beta = zeros(Float32, in_dim)
    gamma = ones(Float32, in_dim)
    moving_mean = zeros(Float32, in_dim)
    moving_variance = ones(Float32, in_dim)
    return BatchNormalization(beta, gamma, Float32(momentum), Float32(eps), moving_mean, moving_variance, name)
end

function build(layer::BatchNormalization)
    return [(layer.name * ".beta", layer.beta), (layer.name * ".gamma", layer.gamma)]
end

function (layer::BatchNormalization)(x::A{T}; training=false) where T <: AbstractFloat
    if T == Float16
        x = Float32.(x)
    end
    if training
        reduction_axes = collect(2:ndims(x))
        mean_x = mean(x, dims=reduction_axes) # (size(x, 1), 1, 1,...)
        var_x = var(x, dims=reduction_axes)  # (size(x, 1), 1, 1,...)
        layer.moving_mean = layer.moving_mean .* layer.momentum .+ mean .* (1.0f0 - layer.momentum)
        layer.moving_variance = layer.moving_variance .* layer.momentum .+ variance .* (1.0f0 - layer.momentum)
    else
        mean_x = layer.moving_mean
        var_x = layer.moving_variance
    end
    inv = 1.0f0 ./ sqrt.(var_x .+ fill(layer.eps, size(var_x)))
    inv = inv .* layer.gamma
    res = -mean_x .* inv
    res = res .+ layer.beta
    x_norm = x .* inv .+ res
    return T.(x_norm)
end

function gpu(layer::BatchNormalization)
    member_list = []
    for field in fieldnames(typeof(layer))
        x = getfield(layer, field) 
        push!(member_list, CuArray(x))
    end
    return BatchNormalization(member_list..., layer.name)
end

