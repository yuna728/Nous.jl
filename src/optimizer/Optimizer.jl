module NOptimizer

using LinearAlgebra, Statistics
using ..Nous.NType

export  SGD,
        Adam,
        DynamicLossScale, get_scaled_loss,
        optimizer_step!

include("sgd.jl")
include("adam.jl")
include("loss_scale.jl")

end