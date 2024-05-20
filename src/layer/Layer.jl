module NLayer

using LinearAlgebra, Statistics
using CUDA
using NNlib
using ..Nous.NType
using ..Nous.NInitializer

export Reshape,
       ReLU, Softmax, softmax,
       LayerNormalization, BatchNormalization,
       Dense, 
       Conv1D, ConvRBlock,
       Dropout,
       EmbedBlock,
       MultiHeadAttention, FFN, Encoder,
       CompressAttention, Compress,
       build, gpu

include("reshape.jl")
include("activate.jl")
include("normalization.jl")
include("dense.jl")
include("conv.jl")
include("dropout.jl")
include("embed.jl")
include("attention.jl")
include("compress_attention.jl")

end