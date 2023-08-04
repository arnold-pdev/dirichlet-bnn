using Statistics
using SpecialFunctions, Distributions
using Flux, Flux.Optimise
using Flux: params
using Base.Iterators: partition
using LinearAlgebra
using Turing, Random, ForwardDiff, ReverseDiff, Preferences
using Metal
using CSV, DataFrames, AdvancedHMC

include("GMMFunctions.jl")

# Hide sampling progress.
# Turing.setprogress!(false);
Turing.setadbackend(:reversediff)
# set_preferences!(ForwardDiff, "nansafe_mode" => true)
# Turing.setadbackend(:zygote)

# data 
u1 = CSV.read("t0_Lorenz.csv", DataFrame)
du = CSV.read("dt0p5_Lorenz.csv", DataFrame)
xs = [collect(v) for v in eachrow(u1)]
ys = [collect(v) for v in eachrow(du)]

xs = xs[end-5:end]
ys = ys[end-5:end]
L = length(xs)

#---------------------------------------

# Architecture
m = 2 # number of Gaussian modes
N = 16 # number of nodes in each hidden layer
Nd = 2 # depth
d = 3 # dimension of data

ℬ = Chain(
Dense(3 => N, tanh),
Dense(N => N, tanh),
# output layer = [w μ d ℓ]
Split( Chain(Dense(N => m), softmax), # weights, the mixture %
      Dense(N => d*m),        # μ, the mix means
      Dense(N => d*m, exp),   # D, the mix diags
      Dense(N => Int64(m*d*(d-1)/2)))# ℓ, the mix lower tri
) |> gpu

#---------------------------------------

# Extract weights and a helper function to reconstruct NN from weights
parameters_initial, reconstruct = Flux.destructure(ℬ)
# bring in pre-trained weights as mean

length(parameters_initial) # number of parameters in NN

# Create a regularization term and a Gaussian prior variance term.
alpha = 1.0
sig = sqrt(1.0 / alpha)

k = 0
# Specify the probabilistic model.
@model function bayes_nn(xs, ys, nparameters, reconstruct)
    global k, L
    # Create the weight and bias vector.
    parameters ~ DistributionsAD.MvNormal(zeros(nparameters), sig .* ones(nparameters))

    # Construct NN from parameters
    nn = reconstruct(parameters)
    # Forward NN to make predictions
    preds = nn.(xs)
    # Observe each prediction. 
    for i in 1:L
        println("======")
        println(preds[i][3])
        println("______")
        println(preds[i][4])
        ys[i] ~ ℳ2mm(preds[i])
        # ys[i] ~ MixtureModel(ℳ2gmm(preds[i]))
    end
    k += 1
    println("hello × $(k)")
end;

# Perform inference.
cnt = 30
# ch = sample(
#     bayes_nn(hcat(xs...), hcat(ys...), length(parameters_initial), reconstruct), HMC(0.005, 4), # 0.05 is leapfrog step size, 4 is number of leapfrog steps
#     cnt); # hcat( ...) is a good trick!
ch = sample(
    bayes_nn(xs, ys, length(parameters_initial), reconstruct), HMC(0.05, 10), # 0.05 is leapfrog step size, 4 is number of leapfrog steps
    cnt); # hcat( ...) is a good trick!

# Extract all weight and bias parameters.
theta = MCMCChains.group(ch, :parameters).value;

# produced non positive definite covar matrix?