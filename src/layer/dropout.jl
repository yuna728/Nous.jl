using Random

mutable struct Dropout <: Layer
    dropout_rate::Float32
    name::String
end

function Dropout(dropout_rate::AbstractFloat; name::String="dropout")
    return Dropout(Float32(dropout_rate, name))

function build(layer::Dropout)
    return []
end

function (layer::Dropout)(x::A{T}; training=false) where T <: AbstractFloat
    if training
        keep_prob = T(1.0 - layer.dropout_rate)
        mask = rand(T, Bernoulli(keep_prob), size(x))
        return mask .* (x ./ keep_prob)
    else
        return x
    end
end

function gpu(layer::Dropout)
    return layer
end

