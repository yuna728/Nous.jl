function get_accuracy(y_true, y_pred)
    y_true = transpose(y_true)
    y_pred = transpose(y_pred)
    return mean(onecold(y_true) .== onecold(y_pred))
end