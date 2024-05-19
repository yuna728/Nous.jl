mutable struct SGD <: Optimizer
    lr::Float32
end

function SGD(lr::AbstractFloat=0.01f0)
    return SGD(Float32(lr))
end

function optimizer_step!(opt::SGD, trainable_list, grad_list)
    for trainable in trainable_list
        trainable .-= opt.lr .* grad_list[trainable]
    end
end