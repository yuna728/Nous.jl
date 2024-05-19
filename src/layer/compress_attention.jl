mutable struct CompressAttention <: Layer
    num_areas::Int
    num_heads::Int
    depth::Int
    max_len::Int
    q_w::Dense
    k_w::Dense
    v_w::Dense
    out_w::Dense
    dropout::Dropout
    E::M{Float32}
    name::String
end

function CompressAttention(in_dim::Int, out_dim::Int, num_areas::Int, num_heads::Int;
    input_len = 100000, init = glorot_uniform, dropout_rate = 0.0, name::String="ca")

    out_dim % num_heads == 0 || throw(ArgumentError("out_dim = $(out_dim) should be divisible by nheads = $(num_heads)"))
    depth = div(out_dim, num_heads)

    q_w = Dense(in_dim, out_dim, name="q_w")
    k_w = Dense(in_dim, out_dim, name="k_w")
    v_w = Dense(in_dim, out_dim, name="v_w")
    out_w = Dense(out_dim, out_dim, name="out_w")
    dropout = Dropout(dropout_rate, name="dropout")

    max_len = 2 * input_len // num_areas - 1
    E = randn(depth, max_len)
    return CompressAttention(num_areas, num_heads, depth, max_len, q_w, k_w, v_w, out_w, dropout, E, name)
end

function build(layer::CompressAttention)
    trainable_layer = []
    for field in fieldnames(layer)
        x = getfield(layer, field) 
        if x isa Layer && !isempty(build(x))
            push!(trainable_layer, (layer.name, build(x)))
        elseif x isa M{Float32}
            push!(trainable_layer, (layer.name * "." * field, x))
        end
    end
    return trainable_layer
end

function (layer::CompressAttention)(x::A{T, 4}; training=false, mask=nothing) where T <: AbstractFloat
    inp_split = split_areas(x, layer.num_areas) # (in_dim, block_len, num_areas, batch_size)

    q = layer.q_w(inp_split, training=training)  # (out_dim, block_len, num_areas, batch_size)
    k = layer.k_w(inp_split, training=training) 
    v = layer.v_w(inp_split, training=training) 

    q = split_heads(q, layer.num_heads, layer.depth) # (depth, num_heads, block_len, num_areas, batch_size)
    k = split_heads(k, layer.num_heads, layer.depth) 
    v = split_heads(v, layer.num_heads, layer.depth) 

    Srel = relative_positional_encoding(q, layer.max_len, layer.E, layer.dropout)

    x, attentioin_weights = dot_product_attention_rpe(q, k, v, Srel, mask=mask) # (out_dim, l_q, num_areas, batch_size)
    x = layer.out_w(x, training=training) # (out_dim, l_q, num_areas, batch_size)
    return x, attention_weights
end

function gpu(layer::CompressAttention; amp::Bool=false)
    q_w = layer.q_w |> gpu(amp=amp)
    k_w = layer.k_w |> gpu(amp=amp)
    v_w = layer.v_w |> gpu(amp=amp)
    out_w = layer.out_w |> gpu(amp=amp)
    if amp
        E = CuArray{Float16}(layer.E)
    else
        E = CuArray(layer.E)
    return CompressAttention(layer.num_areas, layer.num_heads, layer.depth, layer.max_len,
                             q_w, k_w, v_w, out_w, layer.dropout, E, layer.name)
end

function split_heads(x::A{T, 4}, num_heads::Int, depth::Int) where T <: AbstractFloat
    return reshape(x, depth, num_heads, size(x)[2:end]...) # (depth, num_heads, block_len, num_areas, batch_size)
end

function split_areas(x::A{T, 4}, num_areas::Int) where T <: AbstractFloat
    return reshape(x, size(x,1), div(size(x,2),num_areas), num_areas, size(x,3)) # (d_model, block_len, num_areas, batch_size)
end

function relative_positional_encoding(q::A{T1, 5}, max_len::Int, E::M{T2}, dropout::Dropout) where T1 <: AbstractFloat, T2 <: AbstractFloat
    len_q = size(q, 3)
    E = T1.(get_left_embed(E, max_len, len_q)) # (depth, 2*l_q-1)
    E = dropout(E)
    Et = permutedims(E, (2, 1)) # (2*l_q-1, depth)
    qt = permutedims(q, (1, 3, 2, 4, 5)) # (depth, l_q, num_heads, num_areas, batch_size) 
    QE = batched_mul(E, q) # (2*l_q-1, l_q, num_heads, num_areas, batch_size) 
    Srel = relative_shift(QE) # (l_q, l_q, num_heads, num_areas, batch_size) 
end

function get_left_embed(E::M{Float32}, max_len::Int, len_q::Int)
    start_point = max(0, max_len - (2*len_q-1))
    e = E[:,start_point+1:]
    return e
end

function relative_shift(x::A{T, 5}) where T <: AbstractFloat # (2*l_q-1, l_q, num_heads, num_areas, batch_size) 
    to_pad = zeros(T, x[1:1,:,:,:,:]) # (1, l_q, num_heads, num_areas, batch_size) 
    x = cat(to_pad, x, dims=1) # (2*l_q, l_q, num_heads, num_areas, batch_size) 
    t2, t1, num_heads, num_areas, batch_size = size(x)
    x = reshape(x, t1, t2, num_heads, num_areas, batch_size)  # (l_q, 2*l_q, num_heads, num_areas, batch_size) 
    x = x[:,2:,:,:,:] # (l_q, 2*l_q-1, num_heads, num_areas, batch_size) 
    x = reshape(x, t2-1, t1, num_heads, num_areas, batch_size)  # (2*l_q-1, l_q, num_heads, num_areas, batch_size) 
    x = x[:div(t2//2),:,:,:,:] # (l_q, l_q, num_heads, num_areas, batch_size) 
    return x
end

function dot_product_attention_rpe(q::A{T, 5}, k::A{T, 5}, v::A{T, 5}, Srel::A{T, 5}; mask::Mask{T, 5}=nothing)
    kt = permutedims(k, (3, 1, 2, 4, 5)) # (l_k, depth, num_heads, num_areas, batch_size)
    qt = permutedims(q, (1, 3, 2, 4, 5)) # (depth, l_q, num_heads, num_areas, batch_size)
    logits = batched_mul(kt, qt) # (l_k, l_q, num_heads, num_areas, batch_size)
    logits = lgoits .+ Srel
    logits = logits ./ sqrt((size(q, 1)))

    logits = apply_attn_mask(logits, mask)

    attention_weights = softmax(logits, dims=1) # (l_k, l_q, num_heads, num_areas, batch_size)

    vt = permutedims(v, (1, 3, 2, 4, 5)) # (depth, l_v, num_heads, num_areas, batch_size)
    x = batched_mul(vt, attention_weights) # (depth, l_q, num_heads, num_areas, batch_size)

    x = permutedims(x, (1, 3, 2, 4, 5)) # (depth, num_heads, l_q, num_areas, batch_size)
    x = reshape(x, :, size(x)[3:end]...) # (out_dim, l_q, num_areas, batch_size)
    return x, attention_weights
end

function apply_attn_mask(logits::A{T, 5}, mask::Nothing) where T <: AbstractFloat = logits

function apply_attn_mask(logits::A{T, 5}, mask::A{T, 5}) where T <: AbstractFloat
    neginf = typemin(eltype(logits))
    return ifelse.(mask, logits, neginf)
end


mutable struct Compress <: Layer
    ca::CompressAttention
    ffn::FFN
    ln1::LayerNormalization
    ln2::LayerNormalization
    dropout1::Dropout
    dropout2::Dropout
    name::String
end

function Compress(in_dim::Int, d_model::Int, num_areas::Int, num_heads::Int, dff::Int; dropout_rate = 0.1, name::String="compress")
    ca = CompressAttention(in_dim, d_model, num_areas, num_heads, name="ca")
    ffn = FFN(in_dim, d_model, dff, name="ffn")

    layernorm1 = LayerNormalization(epsilon=1e-6, name="layer_norm1")
    layernorm2 = LayerNormalization(epsilon=1e-6, name="layer_norm2")

    dropout1 = Dropout(dropout_rate, name="dropout1")
    dropout2 = Dropout(dropout_rate, name="dropout2")
    return Compress(ca, ffn, layernorm1, layernorm2, dropout1, dropout2, name)
end

function build(layer::Compress)
    trainable_layer = []
    for field in fieldnames(layer)
        x = getfield(layer, field) 
        if x isa Layer && !isempty(build(x))
            push!(trainable_layer, (layer.name, build(x)))
        end
    end
    return trainable_layer
end

function (layer::Compress)(x::A{T, 3}; training::Bool=false, mask::Mask{T, 5}=nothing) where T <: AbstractFloat
    attn_output, attn_weights = layer.ca(x, training=training, mask=mask)  # (batch_size, input_seq_len, d_model)
    attn_output = layer.dropout1(attn_output, training=training)
    out1 = layer.ln1(x .+ attn_output, training=training)  # (batch_size, input_seq_len, d_model)

    ffn_output = layer.ffn(out1, training=training)  # (batch_size, input_seq_len, d_model)
    ffn_output = layer.dropout2(ffn_output, training=training)
    out2 = layer.ln2(x .+ out1 .+ ffn_output, training=training)  # (batch_size, input_seq_len, d_model)
    return out2, attn_weights
end

function gpu(layer::Compress)
    member_list = []
    for field in fieldnames(layer)
        x = getfield(layer, field) 
        if x isa Layer
            push!(member_list, x |> gpu)
        end
    end
    return Compress(member_list..., layer.name)
end