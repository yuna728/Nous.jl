module Train

using LinearAlgebra, Statistics
using Zygote
using ..Nous.MyType
using ..Nous.Model
using ..Nous.Loss
using ..Nous.Optimizer
using ..Nous.Metrics

export train_loop!

function train_loop!(model::NModel, train_batch)
    (x, y) = train_batch
    y_pred = nothing
    loss = nothing
    grad = gradient(Params(model.trainable)) do
        y_pred = forward!(model,x)
        loss = calc_loss(model.loss, y, y_pred)
    end
    optimizer_step!(model.optimizer, model.trainable, grad)
    acc = get_accuracy(y, y_pred)
    return loss, acc
end

end