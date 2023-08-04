using Statistics
using SpecialFunctions, Distributions, Random, Bijectors
using Flux, Flux.Optimise
using Flux: params
using Base.Iterators: partition
using LinearAlgebra, Turing, ReverseDiff
using CSV, DataFrames, BSON
using Dates, Plots, DataFrames

timeisnow = Dates.now()

# reverse diff should be more efficient than forward diff
Turing.setadbackend(:reversediff)

# load and prepare data:
# make sure to prepare the data by removing "< 0.01", etc
data = CSV.read("chemcam-data/ChemCam_PCA_composition.csv", DataFrame)
# replace 0.0 with 0.0001 in data_y
data_y = [ vcat([ maximum([data[i, j],0.001]) for j in 3:10 ], 
[maximum([0.001,100.0-sum([ data[i, j] for j in 3:10 ])])]) for i in 1:size(data)[1] ]
data_y = [ data_y[i]./sum(data_y[i]) for i in 1:size(data)[1] ]
M = 15 # number of PCA modes
data_x = [ [ data[i, j] for j in 11:(M+10) ] for i in 1:size(data)[1] ]

# data used for inference:
data_size = 2000
xs = data_x[1:data_size]
ys = data_y[1:data_size]

# partition the data
# xs = [ data_x[i] for i in partition(1:size(data)[1],10)] |> gpu
# ys = [ data_y[i] for i in partition(1:size(data)[1],10)] |> gpu
# xs = [data_x[i] for i in 1:2200]
# ys = [data_y[i] for i in 1:2200]
L = size(xs)[1]

# valset  = 2201:2557
# valX = data_x[valset] |> gpu
# valY = data_y[valset] |> gpu

# M = 15 # number of PCA modes (defined earlier)
𝒪 = 9 # number of output nodes
N = [M, 5, 5, 5, 𝒪] # number of nodes in each layer (including input and output)
ϵ = 1e-3 # small number to avoid log(0)
# what's going on with DistributionsAD Dirichlet?
# ---------------------------------------
# architecture
α = Chain(
Dense(N[1] => N[2], relu),
Dense(N[2] => N[3], relu),
Dense(N[3] => N[4], relu),
Dense(N[4] => N[5], sigmoid),
x->10*x .+1e-8
) |> gpu
#---------------------------------------

# Extract weights and a helper function to reconstruct NN from weights
parameters_initial, reconstruct = Flux.destructure(α)

# alternatively, bring in pre-trained weights as mean
# BSON.@load "dirichlet-model-relu.bson" model_state
# parameters_inital, reconstruct = Flux.destructure(model_state)
# parameters_start = vcat(asinh.(parameters_initial), ones(np+1))
np = length(parameters_initial) # number of parameters in NN

# define the transformation untilt(β,γ,δ)↦(θ,σ⁻¹,τ⁻¹)
# note: modifying the definition of untilt will not result in a consistent algorithm: one also needs to change the definition of logjacdet.and rand
function untilt(β,γ,δ)
    θ   = sinh.(β)
    σ⁻¹ = exp.(γ)
    τ⁻¹ = exp(δ)
    return (θ,σ⁻¹,τ⁻¹)
end
#...and its log jacobian determinant
function logjacdet(β,γ,δ)
    return sum(log.(cosh.(β))) + sum(γ) + δ
end

# define the horseshoe priors: https://turing.ml/v0.22/docs/using-turing/advanced
struct Horsehoof<: ContinuousMultivariateDistribution end
    #1. define the rand function:
    function horsehoof_rand(rng::AbstractRNG)
        σ = abs.(rand(rng,Cauchy(0,1),np))
        τ = abs(rand(rng,Cauchy(0,1)))
        θ = @. rand(rng,Normal(0,σ*τ))
        return vcat(asinh.(θ), -log.(σ), -log(τ)) # tilt
    end
    Distributions.rand(rng::AbstractRNG, d::Horsehoof) = horsehoof_rand(rng)

    #2. define the (unnormalized) logpdf function:
    function horsehoof_logpdf(β,γ,δ)
        (θ,σ⁻¹,τ⁻¹) = untilt(β,γ,δ)
        logJ = logjacdet(β,γ,δ)
        return sum(@. -0.5*(τ⁻¹*σ⁻¹*θ)^2 + log(σ⁻¹) - log1p(σ⁻¹^2)) + log(τ⁻¹) - log1p(τ⁻¹^2) + logJ
    end

    Distributions.length(d::Horsehoof) = 2*np+1
    Distributions.logpdf(d::Horsehoof, x::AbstractVector{<:Real}) = horsehoof_logpdf(x[1:Int64(floor(end/2))],x[Int64(floor(end/2))+1:end-1],x[end])
    Bijectors.bijector(d::Horsehoof) = identity

# Specify the probabilistic model.
@model function bayes_nn(xs, ys, np, reconstruct, L)
    variables ~ Horsehoof()
    parameters = untilt(variables[1:np],variables[np+1:2*np],variables[2*np+1])[1]
    # Construct NN from parameters
    nn = reconstruct(vec(parameters))
    # Forward NN to make predictions
    preds = nn.(xs)
    # Define likelihood
    for i in 1:L
        ys[i] ~ Dirichlet(preds[i])
    end
end

#---------------------------------------
# Perform inference.
n_samples = 100
leapfrog_stepsize = 0.0001
leapfrog_nsteps = 100
ch = sample(
    bayes_nn(xs, ys, np, reconstruct, L),
    HMC(leapfrog_stepsize,leapfrog_nsteps), n_samples, progress=true)

output = MCMCChains.group(ch, :variables).value;
untilt_output = zeros(size(output))

# make your header!
header = Vector{Symbol}(undef,2*np+1)
header[end] = Symbol("tau")
k = 1
layernumber = 1
while layernumber < length(N)
incomingnode = 1
    while incomingnode < N[layernumber] + 1
    layernode    = 1
        while layernode < N[layernumber+1] + 1
            header[k] = Symbol("theta[$(layernumber),$(incomingnode)in,$(layernode)]")
            header[k + np] = Symbol("sigma[$(layernumber),$(incomingnode)in,$(layernode)]")
            k += 1
            layernode += 1
        end
    incomingnode += 1
    end
for ℓ in 1:N[layernumber+1]
    header[k] = Symbol("theta[$(layernumber),bias,$(ℓ)]")
    header[k + np] = Symbol("sigma[$(layernumber),bias,$(ℓ)]")
    k += 1
end
layernumber += 1
end

for k in eachindex(output.data[1,1,:])
    for j in 1:n_samples
    global untilt_output
    a = untilt(
        output[j,1:np,k], 
        -output[j,np + 1:2*np,k],
        -output[j,2*np+1,k])
        # need negatives to go from precision (σ⁻¹ and τ⁻¹) to std (σ and τ)
    untilt_output[j,:,k] = vcat(a...)
    end
    df = DataFrames.DataFrame(untilt_output[:,:,k], header)

    CSV.write("chain=$(k)-samples=$(n_samples)-param=$(np)-data=$(data_size)-stepsize=$(leapfrog_stepsize)-nsteps=$(leapfrog_nsteps)_$(timeisnow).csv",df)

    lpost[k] = [horsehoof_logpdf(
        vec(output.data[i,1:np,1]),
        vec(output.data[i,np+1:2*np,1]),
        output.data[i,2*np+1,1]) 
        for i in eachindex(theta.data[:,1,1]) ]

    Plots.scatter!(-lpost[k], legend=false, color = :black, markersize = 2, xlabel = "sample", ylabel = "Negative log-posterior") # change the color to a colormap or something
end