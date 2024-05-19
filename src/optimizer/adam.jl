mutable struct Adam <: Optimizer
    lr::Float32
    beta_1::Float32
    beta_2::Float32
    epsilon::Float32
    iter::Int
    momentum
    velocity
end

function Adam(lr::AbstractFloat=0.001f0, beta_1::AbstractFloat=0.9f0, beta_2::AbstractFloat=0.999f0, epsilon::AbstractFloat=1e-7f0)
    return Adam(Float32(lr), Float32(beta_1), Float32(beta_2), Float32(epsilon), 0, nothing, nothing)
end

function optimizer_step!(opt::Adam, trainable_list, grad_list)
    if isnothing(opt.momentum)
        for trainable in trainable_list:
            opt.momentum[trainable[1]] = zeros(Float32, size(trainable[2])) # 全ての要素が0
            opt.velocity[trainable[1]] = zeros(Float32, size(trainable[2])) 
    for trainable in trainable_list
        step = Float32(opt.iter + 1)
        beta_1_power = opt.beta_1 ^ step
        beta_2_power = opt.beta_2 ^ step
        momentum = opt.momentum[trainable[1]]
        velocity = opt.velocity[trainable[1]]
        alpha = opt.lr * sqrt(1.0f0 - beta_2_power) / (1.0f0 - beta_1_power)
        opt.momentum[trainable[1]] = momentum .* opt.beta1 .+ grad_list[trainable] .* (1.0f0 - opt.beta1)
        opt.velocity[trainable[1]] = velocity .* opt.beta2 .+ grad_list[trainable].^2 .* (1.0f0 - opt.beta2)
        trainable .-= (opt.momentum[trainable[1]] .* alpha) ./ (sqrt.(opt.velocity[trainable[1]]) .+ opt.epsilon)
    end
end