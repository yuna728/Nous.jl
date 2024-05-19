mutable struct DynamicLossScale <: Optimizer
    inner_optimizer::Optimizer
    scale::Float32
    dynamic_growth_steps::Int
    step_counter::Int
end

function DynamicLossScale(inner_optimizer::Optimizer, initial_scale::AbstractFloat=2.0f0^15, dynamic_growth_steps::Int=2000)
    return DynamicLossScale(inner_optimizer, Float32(initial_scale), dynamic_growth_steps, 0)
end

function get_scaled_loss(opt::DynamicLossScale, loss::Float32)
    return loss * opt.scale
end

function get_unscaled_gradients(opt::DynamicLossScale, scaled_grads)
    return map(grad -> grad / scale, scaled_grads)
end

function optimizer_step!(opt::DynamicLossScale, trainable_list, grad)
    finite = check_finite(grad)
    if finite
        unscaled_grad = get_unscaled_gradients(opt, grad)
        optimizer_step!(opt.inner_optimizer, trainable_list, unscaled_grad)
        if opt.step_counter == opt.dynamic_growth_steps-1
            opt.step_counter = 0
            opt.scale *= 2.0f0
        else
            opt.step_counter += 1
    else
        opt.step_counter = 0
        opt.dynamic_scale /= 2.0f0
    return finite
end

function check_finite(grads)
    for grad in grads
        if any(isnan, grad) || any(isinf, grad)
            return false
        end
    end
    return true
end
