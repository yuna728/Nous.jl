module Train

using LinearAlgebra, Statistics
using Zygote
using ..Nous.NType
using ..Nous.NModel
using ..Nous.NLoss
using ..Nous.NOptimizer
using ..Nous.NMetrics

export train_loop!

function train_loop!(model::Model, train_batch)
    (x, y) = train_batch
    y_pred = nothing
    loss = nothing
    grad = gradient(Params(model.trainable)) do
        y_pred = model(x)
        loss = calc_loss(model.loss, y, y_pred)
    end
    optimizer_step!(model.optimizer, model.trainable, grad)
    acc = get_accuracy(y, y_pred)
    return loss, acc
end

function train_loop!(model::SSNet, train_batch; amp::Bool=false)
    (x, y_label_da, y_label_ie) = train_batch
    y_pred_da = nothing
    y_pred_ie = nothing
    loss_da = nothing
    loss_ie = nothing
    if amp
        x = Float16.(x)
    end
    scaled_grads = gradient(Params(model.trainable)) do
        y_pred_da, y_pred_ie, _, _  = model(x)
        y_pred_da = Float32.(y_pred_da)
        y_pred_ie = Float32.(y_pred_ie)
        loss_da = calc_loss(model.loss_da, y_label_da, y_pred_da)
        loss_ie = calc_loss(model.loss_ie, y_label_ie, y_pred_ie)
        loss = loss_da + loss_ie
        scaled_loss = get_scaled_loss(model.optimizer, loss)
    end
    finite = optimizer_step!(model.optimizer, model.trainable, grad)
    #acc = get_accuracy(y, y_pred)
    return finite, loss, acc
end

end