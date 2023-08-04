using Statistics
using SpecialFunctions, Distributions
using Flux, Flux.Optimise
using Flux: params
using Base.Iterators: partition
using LinearAlgebra
using Turing, Random
using Metal
using CSV, DataFrames, AdvancedHMC

# Hide sampling progress.
Turing.setprogress!(false);

# data 
u1 = CSV.read("t0_Lorenz.csv", DataFrame)
du = CSV.read("dt0p5_Lorenz.csv", DataFrame)
xs = [collect(v) for v in eachrow(u1)]
ys = [collect(v) for v in eachrow(du)]

xs = xs[end-2:end]
ys = ys[end-2:end]
N = length(xs)

#---------------------------------------

# Architecture
# N = 8 # number of nodes in each hidden layer
# Nd = 7 # depth
# 𝒩 = Chain(
# Dense(3 => N, tanh),
# # Dense(N => N, tanh),
# # Dense(N => N, tanh),
# # Dense(N => N, tanh),
# # Dense(N => N, tanh),
# # Dense(N => N, tanh),
# Dense(N => 3)
# ) |> gpu

# Extract weights and a helper function to reconstruct NN from weights
parameters_initial, reconstruct = Flux.destructure(𝒩)
# bring in pre-trained weights as mean

length(parameters_initial) # number of parameters in NN

# Create a regularization term and a Gaussian prior variance term.
alpha = 1.0
sig = sqrt(1.0 / alpha)

k = 0
# Specify the probabilistic model.
@model function bayes_nn(xs, ys, nparameters, reconstruct)
    global k, N
    # Create the weight and bias vector.
    parameters ~ MvNormal(zeros(nparameters), sig .* ones(nparameters))
    # parameters ~ MvNormal(parameters_initial, sig .* ones(nparameters))

    # Construct NN from parameters
    nn = reconstruct(parameters)
    # Forward NN to make predictions
    preds = nn(xs)

    # Observe each prediction. 
    for i in 1:N #length(ys) isn't working??
        ys[:,i] ~ MvNormal(preds[:,i], sig .* ones(3))
    end
    k += 1
    println("hello × $(k)")
end;

# Perform inference.
cnt = 30
ch = sample(
    bayes_nn(hcat(xs...), hcat(ys...), length(parameters_initial), reconstruct), HMC(0.005, 4), # 0.05 is leapfrog step size, 4 is number of leapfrog steps
    cnt); # hcat( ...) is a good trick!

# Extract all weight and bias parameters.
theta = MCMCChains.group(ch, :parameters).value;

# Gaussian mixture models