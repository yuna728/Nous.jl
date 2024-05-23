module Nous

using LinearAlgebra, Statistics
using Zygote
using CUDA
using NNlib
using Random
using Distributions

include("type.jl")
using .NType
export Layer, Loss, Optimizer, Model, M, V, A, dev

include("initializer.jl")
using .NInitializer
export glorot_uniform

include("layer/Layer.jl")
using .NLayer
export Reshape,
       ReLU, Softmax, softmax,
       Dense, 
       Conv1D, ConvRBlock,
       Dropout,
       EmbedBlock,
       LayerNormalization, BatchNormalization,
       MultiHeadAttention, FFN, Encoder,
       CompressAttention, Compress,
       build, gpu

include("loss/Loss.jl")
using .NLoss
export CrossEntropy, calc_loss

include("metrics/Metrics.jl")
using .NMetrics
export get_accuracy

include("optimizer/Optimizer.jl")
using .NOptimizer
export SGD, Adam,
       DynamicLossScale, get_scaled_loss,
       optimizer_step!

include("model/Model.jl")
using .NModel
export SequentialModel,
       MNISTModel,
       SSNet, create_masks,
       set_loss_ie!, set_loss_da!, set_optimizer!, gpu_model

include("train.jl")
using .Train
export train_loop!

end