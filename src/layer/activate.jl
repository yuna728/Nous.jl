### Relu ###
struct Relu <: NLayer
end

function build(act_func::Relu)
    return []
end

function forward(act_func::Relu, x)
    return ifelse.(x.<0, zero(x), x)
end

### Softmax ###
struct Softmax <: NLayer
end

function build(act_func::Softmax)
    return []
end

function forward(act_func::Softmax, x)
    c = maximum(x, dims=2)
    exp_x = exp.(x .- c)
    sum_exp_x = sum(exp_x, dims=2)
    y = exp_x ./ sum_exp_x
    return y
end