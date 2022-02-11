module train
	include("nn.jl")
	include("../utils/loss.jl")
	using .nn
	using .loss
	function train_network(layer_dims , DMatrix, Y;  η=0.001, epochs=1000, seed=2020, verbose=true)
		# Initiate an empty container for cost, iterations, and accuracy at each iteration
		costs = []
		iters = []
		accuracy = []

		# Initialise random weights for the network
		params = nn.initialise_model_weights(layer_dims, seed)

		# Train the network
		for i = 1:epochs

		    Ŷ , caches  = nn.forward_propagate_model_weights(DMatrix, params)
		    cost = loss.log_loss(Ŷ, Y)
		    acc = loss.accuracy(Ŷ, Y)
		    ∇  = nn.back_propagate_model_weights(Ŷ, Y, caches)
		    params = nn.update_model_weights(params, ∇, η)

		    if verbose
		        println("Iteration -> $i, Cost -> $cost, Accuracy -> $acc")
		    end

		    # Update containers for cost, iterations, and accuracy at the current iteration (epoch)
		    push!(iters , i)
		    push!(costs , cost)
		    push!(accuracy , acc)
		end
		return (cost = costs, iterations = iters, accuracy = accuracy, parameters = params)
	    end
end
