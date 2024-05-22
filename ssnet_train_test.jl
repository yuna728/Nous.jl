using Nous
using CUDA

function test_32()
    model = SSNet(4, 100000, 64, [4, 16, 64], 2, 4, 64, 512, 625, 2, 8, 128, 512, 100000)
    loss_da = CrossEntropy()
    loss_ie = CrossEntropy()
    optimizer = Adam()
    set_loss_da!(model, loss_da)
    set_loss_ie!(model, loss_ie)
    set_optimizer!(model, optimizer)

    @time for iter in 1:100
        batch_x = randn(Float32, 4, 100000, 4)
        batch_y_da = randn(Float32, 4, 100000, 4)
        batch_y_ie = randn(Float32, 3, 100000, 4)
        finite, loss = train_loop!(model, (batch_x, batch_y_da, batch_y_ie))
    end
end

function test_16()
    model = SSNet(4, 100000, 64, [4, 16, 64], 2, 4, 64, 512, 625, 2, 8, 128, 512, 100000)
    loss_da = CrossEntropy()
    loss_ie = CrossEntropy()
    optimizer = Adam()
    set_loss_da!(model, loss_da)
    set_loss_ie!(model, loss_ie)
    optimizer_amp = DynamicLossScale(optimizer)
    set_optimizer!(model, optimizer_amp)

    @time for iter in 1:100
        batch_x = randn(Float32, 4, 100000, 4)
        batch_y_da = randn(Float32, 4, 100000, 4)
        batch_y_ie = randn(Float32, 3, 100000, 4)
        finite, loss = train_loop!(model, (batch_x, batch_y_da, batch_y_ie), amp=true)
    end
end

function test_gpu32()
    model = SSNet(4, 100000, 64, [4, 16, 64], 2, 4, 64, 512, 625, 2, 8, 128, 512, 100000)
    loss_da = CrossEntropy()
    loss_ie = CrossEntropy()
    optimizer = Adam()
    set_loss_da!(model, loss_da)
    set_loss_ie!(model, loss_ie)
    set_optimizer!(model, optimizer)

    gpu_model = gpu(model)

    @time for iter in 1:100
        batch_x = CuArray(randn(Float32, 4, 100000, 4))
        batch_y_da = CuArray(randn(Float32, 4, 100000, 4))
        batch_y_ie = CuArray(randn(Float32, 3, 100000, 4))
        finite, loss = train_loop!(gpu_model, (batch_x, batch_y_da, batch_y_ie))
    end
end

function test_gpu16()
    model = SSNet(4, 100000, 64, [4, 16, 64], 2, 4, 64, 512, 625, 2, 8, 128, 512, 100000)
    loss_da = CrossEntropy()
    loss_ie = CrossEntropy()
    optimizer = Adam()
    set_loss_da!(model, loss_da)
    set_loss_ie!(model, loss_ie)
    optimizer_amp = DynamicLossScale(optimizer)
    set_optimizer!(model, optimizer_amp)

    gpu_model = gpu(model)

    @time for iter in 1:100
        batch_x = CuArray(randn(Float32, 4, 100000, 4))
        batch_y_da = CuArray(randn(Float32, 4, 100000, 4))
        batch_y_ie = CuArray(randn(Float32, 3, 100000, 4))
        finite, loss = train_loop!(gpu_model, (batch_x, batch_y_da, batch_y_ie), amp=true)
    end
end

println("==========")
println("test cpu 32bit")
test_32()
println("==========")
println("test cpu 16bit")
test_16()
println("==========")
println("test gpu 32bit")
test_gpu32()
println("==========")
println("test gpu 16bit")
test_gpu16()
println("==========")