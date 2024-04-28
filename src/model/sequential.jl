### Basic Model ###
mutable struct BasicModel <: NModel
    layers::Vector{NLayer}
    loss::NLoss
    optimizer::NOptimizer
    trainable::Array
end

function build!(model::BasicModel)
    for layer in model.layers
        trainable_layer = build(layer)
        for trainable in trainable_layer
            push!(model.trainable, trainable)
        end
    end
end

function forward!(model::BasicModel, x)
    for layer in model.layers
        x = forward(layer, x)
    end
    return x
end

### Sequential Model ###
mutable struct SequentialModel <: NModel
    layers::Vector{NLayer}
    loss::NLoss
    optimizer::NOptimizer
    trainable::Array
end

function SequentialModel()
    layers = []
    loss = CrossEntropy()
    optimizer = SGD(1.0)
    trainable = []
    return SequentialModel(layers, loss, optimizer, trainable)
end

function add!(model::SequentialModel, layer::NLayer)
    push!(model.layers, layer)
end

function set_loss!(model::SequentialModel, loss::NLoss)
    model.loss = loss
end

function set_optimizer!(model::SequentialModel, opt::NOptimizer)
    model.optimizer = opt
end

function build!(model::SequentialModel)
    for layer in model.layers
        trainable_layer = build(layer)
        for trainable in trainable_layer
            push!(model.trainable, trainable)
        end
    end
end

function forward!(model::SequentialModel, x)
    for layer in model.layers
        x = forward(layer, x)
    end
    return x
end
