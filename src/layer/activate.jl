### Relu ###
struct ReLU <: Layer
    name::String
end

function ReLU(; name::String="relu")
    return ReLU(name)
end

function build(act_func::ReLU)
    return []
end

function (act_func::ReLU)(x::A{T}; training=false) where T <: AbstractFloat
    return ifelse.(x.<0, zeros(T, size(x)), x)
end

function gpu(layer::ReLU)
    return layer
end

### Softmax ###
struct Softmax <: Layer
    name::String
end

function Softmax(; name::String="softmax")
    return Softmax(name)
end

function build(act_func::Softmax)
    return []
end

function (act_func::Softmax)(x::A{T}; dim::Int=1, training::Bool=false) where T <: AbstractFloat
    return softmax(x, dim=dim)
end

function gpu(layer::Softmax)
    return layer
end

function softmax(x::A{T}; dim::Int=1) where T <: AbstractFloat
    if T == Float16
        x = Float32.(x)
    end
    c = maximum(x, dims=dim)
    exp_x = exp.(x .- c)
    sum_exp_x = sum(exp_x, dims=dim)
    y = exp_x ./ sum_exp_x
    return T.(y)
end