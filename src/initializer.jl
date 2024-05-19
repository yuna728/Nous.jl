module Initializer

export glorot_uniform

function glorot_uniform(dims::Int...)
    scale = sqrt(24.0f0 / sum(nfan(dims...)))
    return (rand(Float32, dims...) .- 0.5f0) .* scale
end

end