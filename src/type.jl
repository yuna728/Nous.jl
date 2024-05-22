module NType

using CUDA

export Layer, Loss, Optimizer, Model,
        M, V, A, Mask

abstract type Layer end
abstract type Loss end
abstract type Optimizer end
abstract type Model end

M{T} = Union{Matrix{T}, CuMatrix{T}}
V{T} = Union{Vector{T}, CuVector{T}}
A{T, N} = Union{Array{T, N}, CuArray{T, N}}
Mask{N} = Union{Array{Bool, N}, CuArray{Bool, N}, Nothing}

end