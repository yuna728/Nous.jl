module Layer

using LinearAlgebra, Statistics
using ..Nous.MyType

export Relu, Softmax,
       Dense,
       forward, build

include("activate.jl")
include("dense.jl")

end