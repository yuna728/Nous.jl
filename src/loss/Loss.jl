module NLoss

using LinearAlgebra, Statistics
using ..Nous.NType

export CrossEntropy,
       calc_loss

include("loss_func.jl")

end