struct CrossEntropy <: Loss
    epsilon::Float32
end

function CrossEntropy(;epsilon::AbstractFloat=1f-07)
    return CrossEntropy(Float32(epsilon))
end

function calc_loss(loss::CrossEntropy, y_true::A{Float32}, y_pred::A{Float32})
    return mean(-1.0f0 * sum(y_true .* log.(y_pred .+ loss.epsilon), dims=2))
end