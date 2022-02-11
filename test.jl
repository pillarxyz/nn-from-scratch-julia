using Plots
using MLJBase
using StableRNGs
include("src/train.jl")
using .train

plotly()

# Generate fake data
X, y = make_blobs(10_000, 3; centers=2, as_table=false, rng=2020);
X = Matrix(X');
y = reshape(y, (1, size(X, 2)));
f(x) =  x == 2 ? 0 : x
y2 = f.(y);

input_dim = size(X, 1);

layer_sizes = vcat(input_dim, [5, 3, 1])
layer_sizes = reshape(layer_sizes, (1, size(layer_sizes, 1)))

nn_results = train.train_network(layer_sizes, X, y2; η=0.01, epochs=50, seed=1, verbose=true);

p1 = plot(nn_results.accuracy,
         label="Accuracy",
         xlabel="Number of iterations",
         ylabel="Accuracy as %",
         title="Development of accuracy at each iteration");

p2 = plot(nn_results.cost,
         label="Cost",
         xlabel="Number of iterations",
         ylabel="Cost (J)",
         color="red",
         title="Development of cost at each iteration");

display(plot(p1, p2, layout = (2, 1), size = (800, 800)))
