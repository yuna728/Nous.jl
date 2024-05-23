mutable struct EmbedBlock <: Layer
    embed::Dense
    out_dim::Int
    pos::A{Float32, 3}
    ln::LayerNormalization
    dropout::Dropout
    name::String
end

function EmbedBlock(in_dim::Int, out_dim::Int, pe_input::Int; name::String="embed_block")
    embedding = Dense(in_dim, out_dim, name="embed")
    pos_encoding = positional_encoding(pe_input, out_dim)
    layernorm = LayerNormalization(out_dim, epsilon=1e-6, name="layer_norm")
    dropout = Dropout(0.3f0, name="dropout")
    return EmbedBlock(embedding, out_dim, pos_encoding, layernorm, dropout, name)
end

function get_angles(pos::M{Int}, i::M{Int}, d_model::Int)
  angle_rates = Float32.(1 ./ (10000 .^ ((2 .* (div.(i,2))) ./ d_model))) # (out_dim, 1)
  return pos .* angle_rates # (out_dim, seq_len)
end

function positional_encoding(pos, d_model)
  angle_rads = get_angles(reshape(collect(0:pos-1), 1, pos),
                          reshape(collect(0:d_model-1), d_model, 1),
                          d_model)

  angle_rads[:, 1:2:end] .= sin.(angle_rads[:, 1:2:end])
  angle_rads[:, 2:2:end] .= cos.(angle_rads[:, 2:2:end])

  pos_encoding = reshape(angle_rads, d_model, pos, 1)

  return pos_encoding
end

function build(layer::EmbedBlock)
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

function (layer::EmbedBlock)(x::A{T, 3}; training=false) where T <: AbstractFloat
    inp_len = size(x, 2) # (in_dim, seq_len, batch_size)
    x = layer.embed(x, training=training) # (out_dim, seq_len, batch_size)
    x =  x .* T(sqrt(layer.out_dim))
    x =  x .+ T.(layer.pos[:, 1:inp_len, :]) # (out_dim, seq_len, batch_size)
    x = layer.ln(x, training=training)
    x = layer.dropout(x, training=training)
    return x
end

function gpu(layer::EmbedBlock)
  embed = layer.embed |> gpu
  pos = cu(layer.pos)
  ln = layer.ln |> gpu
  dropout = layer.dropout |> gpu
  return EmbedBlock(embed, layer.out_dim, pos, ln, dropout, layer.name)
end