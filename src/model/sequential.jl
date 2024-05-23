### Sequential Model ###
mutable struct SequentialModel <: Model
    layers::Vector{Layer}
    loss::Loss
    optimizer::Optimizer
    trainable::Array
end

function SequentialModel()
    layers = []
    loss = CrossEntropy()
    optimizer = SGD(1.0)
    trainable = []
    return SequentialModel(layers, loss, optimizer, trainable)
end

function add!(model::SequentialModel, layer::Layer)
    push!(model.layers, layer)
end

function set_loss!(model::SequentialModel, loss::Loss)
    model.loss = loss
end

function set_optimizer!(model::SequentialModel, opt::Optimizer)
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

function (model::SequentialModel)(x; training=false)
    for layer in model.layers
        x = layer(x, training=training)
    end
    return x
end

function gpu_model(model::SequentialModel)
    new_layer = []
    for layer in model.layers
        push!(new_layer, layer |> gpu)
    end
    return SequentialModel(new_layer, model.loss, model.ooptimizer, model.trainable)
end

