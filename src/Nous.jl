module Nous

using LinearAlgebra, Statistics
using Zygote

include("type.jl")
using .NType
export Layer, Loss, Optimizer, Model

include("layer/Layer.jl")
using .NLayer
export Relu, Softmax,
       Dense

include("loss/Loss.jl")
using .NLoss
export CrossEntropy

include("metrics/Metrics.jl")
using .NMetrics
export get_accuracy

include("optimizer/Optimizer.jl")
using .NOptimizer
export SGD

include("model/Model.jl")
using .NModel
export SequentialModel,
       MNISTModel

include("train.jl")
using .Train
export train_loop!

end