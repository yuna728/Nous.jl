mutable struct MNISTModel <: NModel
    layers::Vector{NLayer}
    loss::NLoss
    optimizer::NOptimizer
    trainable::Array
end

function MNISTModel()
    model = MNISTModel([Dense(28*28 ,32), Relu(), Dense(32, 10), Softmax()], CrossEntropy(1e-7), SGD(1.0), [])
    build!(model)
    return model
end

function build!(model::MNISTModel)
    for layer in model.layers
        trainable_layer = build(layer)
        for trainable in trainable_layer
            push!(model.trainable, trainable)
        end
    end
end

function forward!(model::MNISTModel, x)
    for layer in model.layers
        x = forward(layer, x)
    end
    return x
end
