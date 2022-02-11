module activation
    function sigmoid(X)
        sigma = 1 ./(1 .+ exp.(.-X))
        return sigma, X
    end

    function relu(X)
        rel = max.(0,X)
        return rel, X
    end

    function tanh_nn(X)
        result = (exp.(X).-exp.(.-X))./(exp.(X).+exp.(.-X))
        return result, X
    end

    function sigmoid_backwards(∂A, activated_cache)
        s = sigmoid(activated_cache)[1]
        ∂Z = ∂A .* s .* (1 .- s)

        @assert (size(∂Z) == size(activated_cache))
        return ∂Z
    end

    function relu_backwards(∂A, activated_cache)
        return ∂A .* (activated_cache .> 0)
    end
end
