
function log_loss(Ŷ, Y)
    m = size(Y, 2)
    epsilon = eps(1.0)

    # Deal with log(0) scenarios
    Ŷ_new = [max(i, epsilon) for i in Ŷ]
    Ŷ_new = [min(i, 1-epsilon) for i in Ŷ_new]

    cost = -sum(Y .* log.(Ŷ_new) + (1 .- Y) .* log.(1 .- Ŷ_new)) / m
    return cost
end
