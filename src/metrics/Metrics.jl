module Metrics

using LinearAlgebra, Statistics
using Flux: onecold

export get_accuracy

include("metrics_func.jl")

end