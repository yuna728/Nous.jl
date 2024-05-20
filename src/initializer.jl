module NInitializer

export glorot_uniform

nfan() = 1, 1 # fan_in, fan_out
nfan(n) = 1, n # A vector is treated as a n×1 matrix
nfan(n_out, n_in) = n_in, n_out # In case of Dense kernels: arranged as matrices
nfan(dims::Tuple) = nfan(dims...)
nfan(dims...) = prod(dims[1:end-2]) .* (dims[end-1], dims[end]) # In case of convolution kernels

function glorot_uniform(dims::Int...)
    scale = sqrt(24.0f0 / sum(nfan(dims...)))
    return (rand(Float32, dims...) .- 0.5f0) .* scale
end

end