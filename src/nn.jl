using Random
using LinearAlgebra
include("../utils/activation.jl")

function initialise_model_weights(layer_dims, seed)
    params = Dict()

    for l=2:length(layer_dims)
        params[string("W_", (l-1))] = rand(StableRNG(seed), layer_dims[l], layer_dims[l-1]) * sqrt(2 / layer_dims[l-1])
        params[string("b_", (l-1))] = zeros(layer_dims[l], 1)
    end

    return params
end

function forward(A_prev, W, b, activation = "relu")
    @assert activation_function ∈ ("sigmoid", "relu")
    Z = (W * A_prev) .+ b
    linear_cache = (A_prev, W, b)

    @assert size(Z) == (size(W, 1), size(A_prev, 2))

    if activation_function == "sigmoid"
        A, activation_cache = sigmoid(Z)
    end

    if activation_function == "relu"
        A, activation_cache = relu(Z)
    end

    cache = (linear_step_cache=linear_cache, activation_step_cache=activation_cache)

    @assert size(A) == (size(W, 1), size(A_prev, 2))

    return A, cache
end
