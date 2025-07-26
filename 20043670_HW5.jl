using Flux, MLDatasets, Statistics, Random, Plots, Images

import Pkg
Pkg.add("Flux")
Pkg.add("MLDatasets")
Pkg.add("Plots")
Pkg.add("Images")

function load_cifar10_dataset()
    x_train, y_train = CIFAR10.traindata()
    x_test, y_test = CIFAR10.testdata()

    # Normalize and reshape
    x_train = Float32.(permutedims(x_train, (4, 3, 2, 1))) ./ 255
    x_test = Float32.(permutedims(x_test, (4, 3, 2, 1))) ./ 255

    y_train = Flux.onehotbatch(y_train .+ 1, 1:10)
    y_test = Flux.onehotbatch(y_test .+ 1, 1:10)

    return (x_train, y_train), (x_test, y_test)
end

function create_lenet5_net()
    return Chain(
        Conv((5, 5), 3 => 6, relu), MaxPool((2, 2)),
        Conv((5, 5), 6 => 16, relu), MaxPool((2, 2)),
        flatten,
        Dense(400, 120, relu),
        Dense(120, 84, relu),
        Dense(84, 10),
        softmax
    )
end

function fit_net(net, x_train, y_train, x_test, y_test, eps, bs)
    optimizer = ADAM()
    loss_fn(x, y) = Flux.crossentropy(net(x), y)

    data = Flux.DataLoader((x_train, y_train), batchsize=bs, shuffle=true)

    for ep in 1:eps
        for (x, y) in data
            gs = gradient(Flux.params(net)) do
                loss_fn(x, y)
            end
            Flux.Optimise.update!(optimizer, Flux.params(net), gs)
        end
        println("Epoch $ep | Test Acc: ", compute_accuracy(net, x_test, y_test))
    end
    return net
end

function compute_accuracy(net, x, y)
    preds = net(x)
    mean(Flux.onecold(preds) .== Flux.onecold(y))
end

function run_sample_size_experiment()
    (x_train, y_train), (x_test, y_test) = load_cifar10_dataset()
    sample_sizes = [10000, 20000, 30000]
    eps = [6, 3, 2]
    test_accuracies = Float64[]

    for (sz, ep) in zip(sample_sizes, eps)
        net = create_lenet5_net()
        idx = randperm(size(x_train, 4))[1:sz]
        x_subset = x_train[:, :, :, idx]
        y_subset = y_train[:, idx]
        net = fit_net(net, x_subset, y_subset, x_test, y_test, ep, 128)
        push!(test_accuracies, compute_accuracy(net, x_test, y_test))
    end

    plot(sample_sizes, test_accuracies, xlabel="Training Examples", ylabel="Test Accuracy", title="Effect of More Unique Training Examples", marker=:circle)
end

function build_lenet_with_filter(filter_size)
    return Chain(
        Conv((filter_size, filter_size), 3 => 6, relu), MaxPool((2, 2)),
        Conv((filter_size, filter_size), 6 => 16, relu), MaxPool((2, 2)),
        flatten,
        Dense(400, 120, relu),
        Dense(120, 84, relu),
        Dense(84, 10),
        softmax
    )
end

function run_filter_variation_experiment()
    (x_train, y_train), (x_test, y_test) = load_cifar10_dataset()
    filter_sizes = [3, 5, 7]
    accuracies = Float64[]

    for f in filter_sizes
        net = build_lenet_with_filter(f)
        println("Training build_lenet_with_filter$f...")
        fit_net(net, x_train, y_train, x_test, y_test, 3, 128)
        push!(accuracies, compute_accuracy(net, x_test, y_test))
    end

    bar(string.("build_lenet_with_filter", filter_sizes), accuracies, xlabel="Architecture", ylabel="Test Accuracy", title="Effect of Filter Size")
end

function show_feature_maps(net, sample_img)
    sample_img = reshape(Float32.(sample_img) ./ 255, (32, 32, 3, 1))
    conv1 = net[1](sample_img)
    pool1 = net[2](conv1)
    conv2 = net[3](pool1)
    pool2 = net[4](conv2)

    heatmap(Gray.(dropdims(mean(conv1, dims=3)[:, :, 1, :], dims=4)), title="Conv1 Output")
    heatmap(Gray.(dropdims(mean(conv2, dims=3)[:, :, 1, :], dims=4)), title="Conv2 Output")
end

function train_and_display_features()
    (x_train, y_train), _ = load_cifar10_dataset()
    net = build_lenet_with_filter(3)
    fit_net(net, x_train, y_train, x_train, y_train, 1, 128)

    for i in 1:3
        sample_img = x_train[:, :, :, i]
        display(heatmap(Gray.(sample_img[:, :, 1])))
        show_feature_maps(net, sample_img)
    end
end
