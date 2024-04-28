module Nous

using LinearAlgebra, Statistics
using Zygote

include("type.jl")
using .MyType
export NLayer, NLoss, NOptimizer, NModel

include("layer/Layer.jl")
using .Layer
export Relu, Softmax,
       Dense

include("loss/Loss.jl")
using .Loss
export CrossEntropy

include("metrics/Metrics.jl")
using .Metrics
export get_accuracy

include("optimizer/Optimizer.jl")
using .Optimizer
export SGD

include("model/Model.jl")
using .Model
export SequentialModel,
       MNISTModel

include("train.jl")
using .Train
export train_loop!

end