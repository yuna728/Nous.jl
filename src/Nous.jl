module Nous

using LinearAlgebra, Statistics
using Zygote
using CUDA
using Random

include("type.jl")
using .NType
export Layer, Loss, Optimizer, Model

include("initializer.jl")
using .NInitializer
export glorot_uniform

include("layer/Layer.jl")
using .NLayer
export Reshape,
       ReLU, Softmax, softmax,
       Dense, 
       Conv1D, 
       Dropout,
       EmbedBlock,
       LayerNormalization, BatchNormalization,
       MultiHeadAttention, FFN, Encoder,
       CompressAttention, Compress,
       build, gpu

include("loss/Loss.jl")
using .NLoss
export CrossEntropy

include("metrics/Metrics.jl")
using .NMetrics
export get_accuracy

include("optimizer/Optimizer.jl")
using .NOptimizer
export SGD, Adam,
       DynamicLossScale, get_scaled_loss

include("model/Model.jl")
using .NModel
export SequentialModel,
       MNISTModel,
       SSNet

include("train.jl")
using .Train
export train_loop!

end