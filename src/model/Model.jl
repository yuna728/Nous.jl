module NModel

using LinearAlgebra, Statistics
using ..Nous.NType
using ..Nous.NLayer
using ..Nous.NLoss
using ..Nous.NOptimizer

export SequentialModel,
       MNISTModel,
       SSNet,
       build!, add!, set_loss!, set_loss_da!, set_loss_ie!, set_optimizer!

include("sequential.jl")
include("my_mnist.jl")
include("ssnet.jl")

end