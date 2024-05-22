mutable struct Reshape <: Layer
  shape::Tuple{Vararg{Int}}
  name::String
end

function Reshape(shape::Int...; name::String="reshape")
  return Reshape(shape, name)
end

function build(layer::Reshape)
  return []
end

function (layer::Reshape)(x::A{T}; training=false) where T <: AbstractFloat
  return reshape(x, layer.shape..., size(x, ndims(x)))
end

function gpu(layer::Reshape)
  return layer
end