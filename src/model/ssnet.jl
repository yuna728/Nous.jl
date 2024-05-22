

mutable struct SSNet <: Model
  conv1::Conv1D
  conv2::Conv1D
  conv3::Conv1D
  conv_blocks::Vector{ConvRBlock}
  embed1::EmbedBlock
  reshape_in_comp::Reshape
  comp_layers::Vector{Compress}
  reshape_out_comp::Reshape
  embed2::EmbedBlock
  enc_layers::Vector{Encoder}
  fc1::Dense
  reshape1::Reshape
  fc2::Dense
  reshape2::Reshape
  softmax::Softmax
  loss_da::Loss
  loss_ie::Loss
  optimizer::Optimizer
  trainable::Array
end

function SSNet(in_dim::Int, input_len::Int, conv_dim::Int, conv_kernel::Vector{Int}, num_c_layers::Int, num_t_layers::Int, d_comp::Int, d_model::Int,
    num_areas::Int, num_heads_comp::Int, num_heads::Int, dff_comp::Int, dff::Int, pe_input::Int; rate=0.1f0)

  # (in_dim, input_len, batch_size)

    ### Conv blocks ###
    conv1 = Conv1D(in_dim+1, conv_dim, 1, pad="same", name="conv1")
    conv2 = Conv1D(conv_dim, conv_dim, 1, pad="same", name="conv2")
    conv3 = Conv1D(conv_dim, conv_dim, 1, pad="same", name="conv3")
    conv_blocks = [ConvRBlock(conv_dim, conv_dim, k, name="conv_r_block" * string(i)) for (i, k) in enumerate(conv_kernel)]
    # (conv_dim, input_len, batch_size)

    ### Embedding & Pos Encoding 1 ###
    embed1 = EmbedBlock(conv_dim, d_comp, pe_input, name="embed1")
    # (d_comp, input_len, batch_size)

    ## reshape for compress layer ##
    reshape_in_comp = Reshape(d_comp, div(input_len, num_areas), num_areas, name="reshape_in_comp")
    # (d_comp, input_len//num_areas, num_areas, batch_size)

    ### Compress Length by Attention ###
    comp_layers = [Compress(d_comp, d_comp, num_areas, num_heads_comp, dff_comp; dropout_rate=rate, name="compress" * string(i))
                        for i in 1:num_c_layers]
  # (d_comp, input_len//num_areas, num_areas, batch_size)

    ### Encoder ###
    reshape_out_comp = Reshape(d_comp * div(input_len, num_areas), num_areas, name="reshape_out_comp")
  # (d_comp * input_len//num_areas, num_areas, batch_size)

    embed2 = EmbedBlock(d_comp * div(input_len,num_areas), d_model, num_areas, name="embed2")
    # (d_model, num_areas, batch_size)

    enc_layers = [Encoder(d_model, d_model, num_heads, dff; dropout_rate=rate, name="encoder" * string(i)) for i in 1:num_t_layers]
  # (d_model, num_areas, batch_size)

    ### Final Layer ###
  fc1 = Dense(d_model, 640, name="fc1")  # (640, num_areas, batch_size)
  reshape1 = Reshape(4, 160 * num_areas, name="reshape1") # (4, 160 * num_areas, batch_size)

  fc2 = Dense(d_model, 480, name="fc2") # (480, num_areas, batch_size)
  reshape2 = Reshape(3, 160 * num_areas, name="reshape2") # (3, 160 * num_areas, batch_size)
  
  softmax = Softmax(name="softmax")

  model = SSNet(conv1, conv2, conv3, conv_blocks, embed1, reshape_in_comp, comp_layers, reshape_out_comp,
                embed2, enc_layers, fc1, reshape1, fc2, reshape2, softmax, CrossEntropy(), CrossEntropy(), Adam(), [])
  build!(model)
  return model
end

function build!(model::SSNet)
  for field in fieldnames(typeof(model))
      x = getfield(model, field) 
      if x isa Layer
          for (name, weights) in build(x)
              push!(model.trainable, (name, weights))
          end
      elseif x isa Vector && eltype(x) <: Layer
          for i in 1:length(x)
              for (name, weights) in build(x[i])
                  push!(model.trainable, (name, weights))
              end
          end
      end
  end
end

function set_loss_da!(model::SSNet, loss::Loss)
  model.loss_da = loss
end

function set_loss_ie!(model::SSNet, loss::Loss)
  model.loss_ie = loss
end

function set_optimizer!(model::SSNet, optimizer::Optimizer)
  model.optimizer = optimizer
end

function (model::SSNet)(x::A{T, 3}; training::Bool=false) where T <: AbstractFloat
    # x = (in_dim, input_len, batch_size)
    batch = size(x, 3)
    const_0 = fill(T(0.), 1, 1, 1)
    const_1 = fill(T(1.), 1, 1, 1)
    mask = repeat(const_0, outer=[1, size(x, 2), batch]) # (1, input_len, batch_size)
    inp = cat(x, mask, dims=1) # (in_dim+1, input_len, batch_size)
    inp_mask = collect(dropdims(sum(inp .== T(0.), dims=1) .== size(inp, 1); dims=1)) # (input_len, batch_size)
    comp_mask, enc_mask = create_masks(inp_mask, model.comp_layers[1].ca.num_areas)

    ### Conv Tower ###
    x = inp # (in_dim+1, input_len, batch_size) # (in_dim+1, input_len, batch_size) # (kernel_size, in_dim+1, out_dim)
    x_conv1 = model.conv1(x, training=training)
    x = x_conv1
    for i in 1:length(model.conv_blocks)
      x = model.conv_blocks[i](x, training=training)
    end
    x_conv2 = model.conv2(x_conv1, training=training)
    x_conv3 = model.conv3(x, training=training)
    x = x_conv2 .+ x_conv3

    ### adding embedding and position encoding. ###
    inp_len1 = size(inp, 2)
    x = model.embed1(x, training=training) # B * L * d_comp

    ## reshape for compress layer ##
    x = model.reshape_in_comp(x) # (batch_size, num_areas, len//num_areas, d_comp)

    ### Compress Length by Attention ###
    local_attn_ave = nothing
    for i in 1:length(model.comp_layers)
      x, local_attn = model.comp_layers[i](x, training=training, mask=comp_mask) # (batch_size, num_areas, len//num_areas, d_comp)
      if i == 1
        local_attn_ave = local_attn
      else
        local_attn_ave = (i * local_attn_ave .+ local_attn) / (i + 1)
      end
    end

    ### adding embedding and position encoding. ###
    inp_len2 = size(x, 3) # num_areas
    x = model.reshape_out_comp(x, training=training)   # (batch_size, num_areas, len//num_areas * d_comp)
    x = model.embed2(x, training=training)

    ### Encoder ###
    global_attn_list = []
    for i in 1:length(model.enc_layers)
      x, global_attn = model.enc_layers[i](x, training=training, mask=enc_mask)
      global_attn_list = vcat(global_attn_list, [global_attn])  # Create a new list to avoid mutating
    end

    out_encoder = x

    ### Final Layer ###
    out1 = model.fc1(out_encoder, training=training) # B * 625 * 640
    out1 = model.reshape1(out1, training=training) # B * 100000 * 4

    out2 = model.fc2(out_encoder, training=training) # B * 625 * 480
    out2 = model.reshape2(out2, training=training) # B * 100000 * 3

    ### end ###

    output_da = model.softmax(out1, training=training)
    output_ie = model.softmax(out2, training=training)

    return output_da, output_ie, local_attn_ave, global_attn_list
end

function create_masks(inp::M{Bool}, num_areas::Int)
    seq_len, batch = size(inp)   # (input_len, batch_size)
    
    # Reshape the input
    mask = reshape(inp, div(seq_len, num_areas), num_areas, batch) # (seq_len/num_areas, num_areas, batch_size)

    # Create comp_mask
    comp_mask = reshape(mask, div(seq_len, num_areas), 1, 1, num_areas, batch) # (seq_len/num_areas, 1, 1, num_areas, batch_size)

    # Create enc_mask
    enc_mask = collect(dropdims(all(mask .!= 0, dims=1), dims=1))  # (num_areas, batch_size)
    enc_mask = reshape(enc_mask, num_areas, 1, 1, batch) # (num_areas, 1, 1, batch_size)
    
    return comp_mask, enc_mask
end

function gpu(model::SSNet)
  new_layer = []
  for field in fieldnames(typeof(layer))
      x = getfield(layer, field) 
      if x isa Layer
        push!(new_layer, gpu(x))
      elseif x isa Vector && eltype(x) <: Layer
        push!(new_layer, gpu.(x))
      end
  end
  return SSNet(new_layer..., model.loss, model.ooptimizer, model.trainable)
end
