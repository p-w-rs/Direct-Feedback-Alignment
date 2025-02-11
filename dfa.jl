using MLDatasets, OneHotArrays, LinearAlgebra, Random, ProgressMeter, Plots

# Set random seed for reproducibility
rng = Random.default_rng()
Random.seed!(rng, 0)
T = Float32

# Activation function and its derivative
σ(x) = 1 ./ (1 .+ exp.(-x))
σ′(x) = σ(x) .* (1 .- σ(x))

# Softmax function
softmax(x::AbstractMatrix) = exp.(x) ./ sum(exp.(x), dims=1)
softmax′(x::AbstractMatrix) = softmax(x) .* (1.0 .- softmax(x))

# Hyperparameters
batch_size = 256
η = 0.001f0
decay = 0.99f0

# Load and process MNIST data
x_train, y_train = MNIST(split=:train)[:]
x_test, y_test = MNIST(split=:test)[:]
x_train, y_train = Float32.(reshape(x_train, 28 * 28, :)), Float32.(onehotbatch(y_train, 0:9))
x_test, y_test = Float32.(reshape(x_test, 28 * 28, :)), Float32.(onehotbatch(y_test, 0:9))

### Define network weights ###
# We use glorot initialization for the weights
# by dividing the random weights by the square root of the number of input units
# 784 -> 64 -> 32 -> 10
W = [
    randn(rng, T, 64, 28 * 28) ./ sqrt(28 * 28),
    randn(rng, T, 32, 64) ./ sqrt(64),
    randn(rng, T, 10, 32) ./ sqrt(32)
]
b = [ones(T, 64), ones(T, 32), ones(T, 10)]

### Define direct feedback alignment matrices ###
B = [
    randn(rng, T, size(W[1], 1), 10),
    randn(rng, T, size(W[2], 1), 10),
    Matrix{T}(I, size(W[3], 1), 10)
]

# Initialize arrays to store accuracy and loss
train_accuracies = Float64[]
test_accuracies = Float64[]

for epoch in 1:200
    global W, B, b, η

    @showprogress for _ in 1:div(size(y_train, 2), batch_size)
        idxs = randperm(rng, size(x_train, 2))[1:batch_size]
        x, y = x_train[:, idxs], y_train[:, idxs]

        # Forward pass
        a1 = W[1] * x .+ b[1]
        h1 = σ(a1)

        a2 = W[2] * h1 .+ b[2]
        h2 = σ(a2)

        ay = W[3] * h2 .+ b[3]
        ŷ, ŷ′ = softmax(ay), softmax′(ay)

        # Error calculation
        e = ŷ .- y

        # Backward pass and weight update (using direct feedback alignment)
        δay = B[3] * e # B is the identity matrix so doesn't affect the error
        δa2 = B[2] * e .* σ′(a2)
        δa1 = B[1] * e .* σ′(a1)

        ΔW3 = δay * h2'
        ΔW2 = δa2 * h1'
        ΔW1 = δa1 * x'

        W[3] .-= η .* ΔW3
        W[2] .-= η .* ΔW2
        W[1] .-= η .* ΔW1
    end

    # Calculate train accuracy and loss
    train_preds = softmax(W[3] * σ.(W[2] * σ.(W[1] * x_train)))
    train_acc = sum(onecold(train_preds) .== onecold(y_train)) / size(y_train, 2)
    push!(train_accuracies, train_acc)

    # Calculate test accuracy
    test_preds = softmax(W[3] * σ.(W[2] * σ.(W[1] * x_test)))
    test_acc = sum(onecold(test_preds) .== onecold(y_test)) / size(y_test, 2)
    push!(test_accuracies, test_acc)

    println("Epoch $epoch: Train accuracy = $train_acc, Test accuracy = $test_acc")
    η *= decay
end

# Plot the results
plot(train_accuracies, label="Train Accuracy", title="Learning Progress", xlabel="Epoch", ylabel="Accuracy")
plot!(test_accuracies, label="Test Accuracy")
savefig("learning_progress.png")
