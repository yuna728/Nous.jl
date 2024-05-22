mutable struct Adam <: Optimizer
    lr::Float32
    beta1::Float32
    beta2::Float32
    epsilon::Float32
    iter::Int
    momentum::Dict{String, A{Float32}}
    velocity::Dict{String, A{Float32}}
end

function Adam(lr::AbstractFloat=0.001f0, beta1::AbstractFloat=0.9f0, beta2::AbstractFloat=0.999f0, epsilon::AbstractFloat=1f-7)
    return Adam(Float32(lr), Float32(beta1), Float32(beta2), Float32(epsilon), 0, Dict{String, A{Float32}}(), Dict{String, A{Float32}}())
end

function optimizer_step!(opt::Adam, trainable_list, grad_list)
    if isempty(opt.momentum)
        for (name, weight) in trainable_list
            opt.momentum[name] = zeros(Float32, size(weight))
            opt.velocity[name] = zeros(Float32, size(weight))
        end
    end
    opt.iter += 1
    for (name, weight) in trainable_list
        step = Float32(opt.iter)
        beta1_power = opt.beta1 ^ step
        beta2_power = opt.beta2 ^ step
        momentum = opt.momentum[name]
        velocity = opt.velocity[name]
        alpha = opt.lr * sqrt(1.0f0 - beta2_power) / (1.0f0 - beta1_power)
        opt.momentum[name] = momentum * opt.beta1 .+ grad_list[weight] * (1.0f0 - opt.beta1)
        opt.velocity[name] = velocity * opt.beta2 .+ grad_list[weight].^2 * (1.0f0 - opt.beta2)
        trainable .-= (opt.momentum[name] * alpha) ./ (sqrt.(opt.velocity[name]) .+ opt.epsilon)
    end
end