struct CrossEntropy <: NLoss
    epsilon::AbstractFloat
end

function calc_loss(loss::CrossEntropy, y_true, y_pred)
    return mean(-1. .* sum(y_true .* log.(y_pred .+ loss.epsilon), dims=2))
end