using Flux: onehotbatch
using Flux
using MLDatasets
using Nous

function main()
    x_train, y_train = MLDatasets.MNIST(split=:train)[:]
    x_train = Flux.flatten(x_train) # 784×60000
    y_train = onehotbatch(y_train, 0:9) # 10×60000
    x_train = transpose(x_train)
    y_train = transpose(y_train)

    model = MNISTModel()
    batch_size = 256
    data_size = 60000
    epochs = 10
    max_iters = div(data_size, batch_size)

    for epoch in 1:epochs
        indices = rand(1:data_size, data_size)
        rand_x = x_train[indices, :]
        rand_y = y_train[indices, :]
        total_loss = 0.
        total_acc = 0.
        count = 0
        @time for iter in 1:max_iters
            batch_x = rand_x[(iter-1)*batch_size+1:iter*batch_size,:]
            batch_y = rand_y[(iter-1)*batch_size+1:iter*batch_size,:]
            loss, acc = train_loop!(model, (batch_x, batch_y))
            total_loss += loss
            total_acc += acc
            count += 1
        end
        avg_loss = total_loss / count
        avg_acc = total_acc / count
        println("| epoch $(epoch) | loss $(round(avg_loss, digits=4)) | acc $(round(avg_acc, digits=4))")
        flush(stdout)
    end
end

main()