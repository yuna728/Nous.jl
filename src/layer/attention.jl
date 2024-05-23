mutable struct MultiHeadAttention <: Layer
    num_heads::Int
    depth::Int
    q_w::Dense
    k_w::Dense
    v_w::Dense
    out_w::Dense
    name::String
end

function MultiHeadAttention(in_dim::Int, out_dim::Int, num_heads::Int;
    init = glorot_uniform, dropout_rate::AbstractFloat = 0.0f0, name::String="mha")

    out_dim % num_heads == 0 || throw(ArgumentError("out_dim = $(out_dim) should be divisible by nheads = $(num_heads)"))

    q_w = Dense(in_dim, out_dim, name="q_w")
    k_w = Dense(in_dim, out_dim, name="k_w")
    v_w = Dense(in_dim, out_dim, name="v_w")
    out_w = Dense(out_dim, out_dim, name="out_w")
    return MultiHeadAttention(num_heads, div(out_dim, num_heads), q_w, k_w, v_w, out_w, name)
end

function build(layer::MultiHeadAttention)
    trainable_layer = []
    for field in fieldnames(typeof(layer))
        x = getfield(layer, field) 
        if x isa Layer
            for (name, weights) in build(x)
                push!(trainable_layer, (layer.name * "." * name, weights))
            end
        end
    end
    return trainable_layer
end

function (layer::MultiHeadAttention)(q::A{T, 3}, k::A{T, 3}, v::A{T, 3}; training::Bool=false, mask::Mask{4}=nothing) where T <: AbstractFloat
    q = layer.q_w(q, training=training)  # [q] = [qk_dim, q_len, batch_size]
    k = layer.k_w(k, training=training)  # [k] = [qk_dim, kv_len, batch_size] 
    v = layer.v_w(v, training=training)  # [v] = [v_dim, kv_len, batch_size]

    q = split_heads(q, layer.num_heads, layer.depth) 
    k = split_heads(k, layer.num_heads, layer.depth) 
    v = split_heads(v, layer.num_heads, layer.depth) 

    x, attention_weights = dot_product_attention(q, k, v, mask)
    x = layer.out_w(x, training=training) # (out_dim, q_len, batch_size)
    return x, attention_weights
end

function split_heads(x::A{T, 3}, num_heads::Int, depth::Int) where T <: AbstractFloat
    return reshape(x, depth, num_heads, size(x)[2:end]...) # (depth, num_heads, seq_len, batch_size)
end

function dot_product_attention(q::A{T, 4}, k::A{T, 4}, v::A{T, 4}, mask::Mask{4}) where T <: AbstractFloat
    # The following permutedims and batched_mul are equivalent to
    # @tullio logits[j, i, h, b] := q[d, h, i, b] * k[d, h, j, b] / √T(qk_dim)
    kt = permutedims(k, (3, 1, 2, 4)) # (k_len, depth, num_heads, batch_size)
    qt = permutedims(q, (1, 3, 2, 4)) ./ sqrt(T(size(q, 1))) # (depth, q_len, num_heads, batch_size)
    logits = batched_mul(kt, qt) # (k_len, q_len, num_heads, batch_size)

    logits = apply_attn_mask(logits, mask)

    attention_weights = softmax(logits, dim=1) # (k_len, q_len, num_heads, batch_size)

    vt = permutedims(v, (1, 3, 2, 4)) # (depth, v_len, num_heads, batch_size)
    x = batched_mul(vt, attention_weights) # (depth, q_len, num_heads, batch_size)
    x = permutedims(x, (1, 3, 2, 4)) # (depth, num_heads, q_len, batch_size)
    x = reshape(x, :, size(x)[3:end]...) # (out_dim, q_len, batch_size)
    return x, attention_weights
end

apply_attn_mask(logits::A{T, 4}, mask::Nothing) where T <: AbstractFloat = logits

function apply_attn_mask(logits::A{T, 4}, mask::A{Bool, 4}) where T <: AbstractFloat
    neginf = typemin(eltype(logits))
    return ifelse.(mask, neginf, logits)
end

function gpu(layer::MultiHeadAttention)
    member_list = []
    for field in fieldnames(typeof(layer))
        x = getfield(layer, field) 
        if x isa Layer
            push!(member_list, gpu(x))
        else
            push!(member_list, x)
        end
    end
    return MultiHeadAttention(member_list...)
end

mutable struct FFN <: Layer
    dense1::Dense
    dense2::Dense
    name::String
end

function FFN(in_dim::Int, d_model::Int, dff::Int; name::String="ffn")
    dense1 = Dense(in_dim, dff, activation=relu, name="dense1")
    dense2 = Dense(dff, d_model, name="dense2")
    return FFN(dense1, dense2, name)
end

function build(layer::FFN)
    trainable_layer = []
    for field in fieldnames(typeof(layer))
        x = getfield(layer, field) 
        if x isa Layer
            for (name, weights) in build(x)
                push!(trainable_layer, (layer.name * "." * name, weights))
            end
        end
    end
    return trainable_layer
end

function (layer::FFN)(x::A{T}; training=false) where T <: AbstractFloat
    dense1_out = layer.dense1(x, training=training)
    dense2_out = layer.dense2(dense1_out, training=training)
    return dense2_out
end

function gpu(layer::FFN)
    member_list = []
    for field in fieldnames(typeof(layer))
        x = getfield(layer, field) 
        if x isa Layer
            push!(member_list, x |> gpu)
        end
    end
    return FFN(member_list..., layer.name)
end

mutable struct Encoder <: Layer
    mha::MultiHeadAttention
    ffn::FFN
    ln1::LayerNormalization
    ln2::LayerNormalization
    dropout1::Dropout
    dropout2::Dropout
    name::String
end

function Encoder(in_dim::Int, d_model::Int, num_heads::Int, dff::Int; dropout_rate = 0.1, name::String="encoder")
    mha = MultiHeadAttention(in_dim, d_model, num_heads, name="mha")
    ffn = FFN(d_model, d_model, dff, name="ffn")

    layernorm1 = LayerNormalization(d_model, epsilon=1e-6, name="layer_norm1")
    layernorm2 = LayerNormalization(d_model, epsilon=1e-6, name="layer_norm2")

    dropout1 = Dropout(dropout_rate, name="dropout1")
    dropout2 = Dropout(dropout_rate, name="dropout2")
    return Encoder(mha, ffn, layernorm1, layernorm2, dropout1, dropout2, name)
end

function build(layer::Encoder)
    trainable_layer = []
    for field in fieldnames(typeof(layer))
        x = getfield(layer, field) 
        if x isa Layer
            for (name, weights) in build(x)
                push!(trainable_layer, (layer.name * "." * name, weights))
            end
        end
    end
    return trainable_layer
end

function (layer::Encoder)(x::A{T, 3}; training=false, mask::Mask{4}=nothing) where T <: AbstractFloat
    attn_output, attn_weights = layer.mha(x, x, x, training=training, mask=mask)  # (batch_size, input_seq_len, d_model)
    attn_output = layer.dropout1(attn_output, training=training)
    out1 = layer.ln1(x .+ attn_output, training=training)  # (batch_size, input_seq_len, d_model)

    ffn_output = layer.ffn(out1)  # (batch_size, input_seq_len, d_model)
    ffn_output = layer.dropout2(ffn_output, training=training)
    out2 = layer.ln2(x .+ out1 .+ ffn_output, training=training)  # (batch_size, input_seq_len, d_model)
    return out2, attn_weights
end

function gpu(layer::Encoder)
    member_list = []
    for field in fieldnames(typeof(layer))
        x = getfield(layer, field) 
        if x isa Layer
            push!(member_list, x |> gpu)
        end
    end
    return Encoder(member_list..., layer.name)
end