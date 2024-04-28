module Optimizer

using LinearAlgebra, Statistics
using ..Nous.MyType

export SGD,
       optimizer_step!

include("opt.jl")

end