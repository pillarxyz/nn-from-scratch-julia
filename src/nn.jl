module nn
    using Random
    using StableRNGs
    using LinearAlgebra
    include("../utils/activation.jl")
    using .activation

    function initialise_model_weights(layer_dims, seed)
        params = Dict()

        for l=2:length(layer_dims)
            params[string("W_", (l-1))] = rand(StableRNG(seed), layer_dims[l], layer_dims[l-1]) * sqrt(2 / layer_dims[l-1])
            params[string("b_", (l-1))] = zeros(layer_dims[l], 1)
        end

        return params
    end

    function forward(A_prev, W, b, f_activation = "relu")
        @assert f_activation ∈ ("sigmoid", "relu")
        Z = (W * A_prev) .+ b
        linear_cache = (A_prev, W, b)

        @assert size(Z) == (size(W, 1), size(A_prev, 2))

        if f_activation == "sigmoid"
            A, activation_cache = activation.sigmoid(Z)
        end

        if f_activation == "relu"
            A, activation_cache = activation.relu(Z)
        end

        cache = (linear_step_cache=linear_cache, activation_step_cache=activation_cache)

        @assert size(A) == (size(W, 1), size(A_prev, 2))

        return A, cache
    end

    function forward_propagate_model_weights(DMatrix, parameters)
        master_cache = []
        A = DMatrix
        L = Int(length(parameters) / 2)

        # Forward propagate until the last (output) layer
        for l = 1 : (L-1)
            A_prev = A
            A, cache = forward(A_prev,
                               parameters[string("W_", (l))],
                               parameters[string("b_", (l))],
                               "relu")
            push!(master_cache , cache)
        end

        # Make predictions in the output layer
        Ŷ, cache = forward(A,
                       parameters[string("W_", (L))],
                       parameters[string("b_", (L))],
                       "sigmoid")
        push!(master_cache, cache)

        return Ŷ, master_cache
    end

    function backward(∂Z, cache)
        # Unpack cache
        A_prev , W , b = cache
        m = size(A_prev, 2)

        # Partial derivates of each of the components
        ∂W = ∂Z * (A_prev') / m
        ∂b = sum(∂Z, dims = 2) / m
        ∂A_prev = (W') * ∂Z

        @assert (size(∂A_prev) == size(A_prev))
        @assert (size(∂W) == size(W))
        @assert (size(∂b) == size(b))

        return ∂W , ∂b , ∂A_prev
    end

    function linear_activation_backward(∂A, cache, f_activation="relu")
        @assert f_activation ∈ ("sigmoid", "relu")

        linear_cache , cache_activation = cache

        if (f_activation == "relu")

            ∂Z = activation.relu_backwards(∂A , cache_activation)
            ∂W , ∂b , ∂A_prev = backward(∂Z , linear_cache)

        elseif (f_activation == "sigmoid")

            ∂Z = activation.sigmoid_backwards(∂A , cache_activation)
            ∂W , ∂b , ∂A_prev = backward(∂Z , linear_cache)

        end

        return ∂W , ∂b , ∂A_prev
    end

    function back_propagate_model_weights(Ŷ, Y, master_cache)
        # Initiate the dictionary to store the gradients for all the components in each layer
        ∇ = Dict()

        L = length(master_cache)
        Y = reshape(Y , size(Ŷ))

        # Partial derivative of the output layer
        ∂Ŷ = (-(Y ./ Ŷ) .+ ((1 .- Y) ./ ( 1 .- Ŷ)))
        current_cache = master_cache[L]

        # Backpropagate on the layer preceeding the output layer
        ∇[string("∂W_", (L))], ∇[string("∂b_", (L))], ∇[string("∂A_", (L-1))] = linear_activation_backward(∂Ŷ, 
                                                                                                           current_cache, 
                                                                                                           "sigmoid")
        # Go backwards in the layers and compute the partial derivates of each component.
        for l=reverse(0:L-2)
            current_cache = master_cache[l+1]
            ∇[string("∂W_", (l+1))], ∇[string("∂b_", (l+1))], ∇[string("∂A_", (l))] = linear_activation_backward(∇[string("∂A_", (l+1))], 
                                                                                                                 current_cache, 
                                                                                                                 "relu")
        end

        # Return the gradients of the network
        return ∇
    end

    function update_model_weights(parameters, ∇, η)

        L = Int(length(parameters) / 2)

        # Update the parameters (weights and biases) for all the layers
        for l = 0: (L-1)
            parameters[string("W_", (l + 1))] -= η .* ∇[string("∂W_", (l + 1))]
            parameters[string("b_", (l + 1))] -= η .* ∇[string("∂b_", (l + 1))]
        end

        return parameters
    end

end
