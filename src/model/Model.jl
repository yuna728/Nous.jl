module Model

using LinearAlgebra, Statistics
using ..Nous.MyType
using ..Nous.Layer
using ..Nous.Layer: build, forward
using ..Nous.Loss
using ..Nous.Optimizer

export SequentialModel,
       MNISTModel,
       forward!, build!, add!, set_loss!, set_optimizer!

include("sequential.jl")
include("my_mnist.jl")

end