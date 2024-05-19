module NLayer

using LinearAlgebra, Statistics
using NNlib
using ..Nous.NType
using ..Nous.Initializer

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

include("reshape.jl")
include("activate.jl")
include("dense.jl")
include("conv.jl")
include("dropout.jl")
include("embed.jl")
include("normalization.jl")
include("attention.jl")
include("compress_attention.jl")

end