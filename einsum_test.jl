using NNlib

# Example tensor definitions
depth = 3
l_q = 2
num_heads = 4
num_areas = 5
batch_size = 6

E = randn(Float32,  2 * l_q - 1, depth)  # (2*l_q-1, depth)
q = randn(Float32, depth, l_q, num_heads, num_areas, batch_size)  # (depth, l_q, num_heads, num_areas, batch_size)

# Method 1: Using batched_mul
function batched_mul_method(E, q)
    # Reshape for batched multiplication
    E_t_reshaped = repeat(E, outer=(1, 1,  num_heads, num_areas, batch_size))  # (2*l_q-1, depth, 1, 1, 1)
    q_t_reshaped = reshape(q, depth, l_q, num_heads, num_areas, batch_size)  # (depth, l_q, num_heads, num_areas, batch_size)

    # Perform batched multiplication
    QE = batched_mul(E_t_reshaped, q_t_reshaped)  # (2*l_q-1, l_q, num_heads, num_areas, batch_size)
    return QE
end

# Method 2: Using explicit loops
function explicit_loop_method(E, q)
    depth, l_q, num_heads, num_areas, batch_size = size(q)
    l_q2 = size(E, 1)  # 2*l_q - 1
    QE = zeros(Float32, l_q2, l_q, num_heads, num_areas, batch_size)

    for i in 1:l_q2
        for j in 1:l_q
            for k in 1:num_heads
                for l in 1:num_areas
                    for m in 1:batch_size
                        QE[i, j, k, l, m] = sum(E[i, :] .* q[:, j, k, l, m])
                    end
                end
            end
        end
    end
    return QE
end

function batched_mul_method2(E, q)
    # Reshape for batched multiplication
    q_t_reshaped = reshape(q, depth, l_q * num_heads * num_areas * batch_size)  # (depth, l_q*num_heads*num_areas*batch_size)
    
    # Perform batched multiplication
    QE_flat = batched_mul(E, q_t_reshaped)  # (2*l_q-1, l_q*num_heads*num_areas*batch_size)
    
    # Reshape result back to original dimensions
    QE = reshape(QE_flat, 2 * l_q -1, l_q, num_heads, num_areas, batch_size)  # (2*l_q-1, l_q, num_heads, num_areas, batch_size)
    return QE
end

# Compute results using both methods
result_batched_mul = batched_mul_method2(E, q)
result_explicit_loop = explicit_loop_method(E, q)

# Compare results
println("Results are equal: ", all(result_batched_mul .≈ result_explicit_loop))

# Display results for verification
println("Result using batched_mul method:")
println(result_batched_mul)

println("Result using explicit loop method:")
println(result_explicit_loop)