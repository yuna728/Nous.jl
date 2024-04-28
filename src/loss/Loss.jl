module Loss

using LinearAlgebra, Statistics
using ..Nous.MyType

export CrossEntropy,
       calc_loss

include("loss_func.jl")

end