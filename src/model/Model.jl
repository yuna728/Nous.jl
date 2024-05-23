module NModel

using LinearAlgebra, Statistics
using CUDA
using ..Nous.NType
using ..Nous.NLayer
using ..Nous.NLoss
using ..Nous.NOptimizer

export SequentialModel,
       MNISTModel,
       SSNet, create_masks,
       build!, add!, set_loss!, set_loss_da!, set_loss_ie!, set_optimizer!, gpu_model

include("sequential.jl")
include("my_mnist.jl")
include("ssnet.jl")

end