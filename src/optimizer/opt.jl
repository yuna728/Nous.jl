mutable struct SGD <: NOptimizer
    lr::AbstractFloat
end

function optimizer_step!(opt::SGD, trainable_list, grad_list)
    for trainable in trainable_list
        trainable .-= opt.lr .* grad_list[trainable]
    end
end