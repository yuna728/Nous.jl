using NNlib: conv

mutable struct Conv1D{F} <: Layer
    weight::M{Float32}
    bias::M{Float32}
    activation::F
    stride::Int
    pad::Tuple
    dilation::Int
    name::String
  end
end

function Conv1D(in_dim::Int, out_dim::Int, kernel_size::Int; bias::Bool = true, activation = identity,
    stride::Int = 1, pad::String = 'same', dilation::Int = 1, name::String="conv1d")
    if pad == 'same'
        pad_amt = kernel_size - 1
        pad = Tuple([cld(pad_amt, 2), fld(pad_amt,2)])
    return Conv1D(glorot_uniform(kernel_size, out_dim, in_dim), zeros(out_dim, 1), activation, stride, pad, dilation, name)
end

function build(layer::Conv1D)
    trainable_layer = []
    for field in fieldnames(layer)
        x = getfield(layer, field) 
        if x isa M{Float32}
            push!(trainable_layer, (layer.name * "." * field, x))
        end
    end
    return trainable_layer
end

function (layer::Conv1D)(x::A{T, 3}; training=false) where T <: AbstractFloat
    cdims = DenseConvDims(x, layer.weight; stride = layer.stride, padding = layer.pad, dilation = layer.dilation)
    return layer.activation.(conv(x, layer.weight, cdims) .+ layer.bias)
end

function gpu(layer::Conv1D)
    weight = CuArray(layer.weight)
    bias = CuArray(layer.bias)
    return Conv1D(weight, bias, layer.activation, layer.stride, layer.pad, layer.dilation, layer.name)
end

mutable struct ConvRBlock{F} <: Layer
    conv1::Conv1D{F}
    conv2::Conv1D{F}
    bn1::BatchNormalization
    bn2::BatchNormalization
    relu1::ReLU
    relu2::ReLU
    name::String
end

function ConvRBlock(in_dim::Int, out_dim::Int, kernel_size::Int; name::String="conv_r_block")
    conv1 = Conv1D(in_dim, out_dim, kernel_size; pad='same', name="conv1")
    conv2 = Conv1D(in_dim, out_dim, kernel_size; pad='same', name="conv2")
    bn1 = BatchNormalization(name="batch_norm1")
    bn2 = BatchNormalization(name="batch_norm2")
    relu1 = ReLU(name="relu1")
    relu2 = ReLU(name="relu2")
    return ConvRBlock(conv1, conv2, bn1, bn2, relu1, relu2, name)
end

function build(layer::ConvRBlock)
    trainable_layer = []
    for field in fieldnames(layer)
        x = getfield(layer, field) 
        if x isa Layer && !isempty(build(x))
            push!(trainable_layer, (layer.name, build(x)))
        end
    end
    return trainable_layer
end

function (layer::ConvRBlock)(x::A{T, 3}; training=false) where T <: AbstractFloat
    conv1_out = layer.conv1(x, training=training)
    bn1_out = layer.bn1(conv1_out, training=training)
    relu1_out = layer.relu1(bn1_out, training=training)

    conv2_out = layer.conv2(relu1_out, training=training)
    bn2_out = layer.bn2(conv2_out, training=training)
    relu2_out = layer.relu2(bn2_out, training=training)
    return x .+ relu2_out
end

function gpu(layer::ConvRBlock)
    member_list = []
    for field in fieldnames(layer)
        x = getfield(layer, field) 
        if x isa Layer
            push!(member_list, x |> gpu)
        end
    end
    return ConvRBlock(member_list..., layer.name)
end