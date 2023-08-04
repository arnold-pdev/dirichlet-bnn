using GaussianFilters, Flux, LinearAlgebra
using CSV 

# ts = CSV.read("ts_Lorenz.csv",DataFrame)
# ts = [collect(v) for v in eachrow(ts)]
# 

function learn_by_ukf(𝒩𝒩,ts::Array,σ²=1.0,priors="iid")
    # Advance the parameters of a neural network emulating an ODE solver
    # inputs:
    #        priors, specifications for the parameter priors
    #        ts, a pre-recorded time series
    #        𝒩𝒩, a Flux chain to be trained to emulate f(u)-u

    # PRIORS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Specify the priors and prepare for inference.

    # Extract weights and a helper function to reconstruct 𝒩𝒩 from weights
    parameters_initial, reconstruct = Flux.destructure(𝒩𝒩)
    nₚ = length(parameters_initial) # number of parameters in 𝒩𝒩

    # cast as Gaussian belief for GaussianFilters.jl
    # SUPER WEIRD: this doesn't work when μ = zeros!
    μ₀ = ones(nₚ)
    μ₀ = convert(Vector{Float64},parameters_initial)
    if priors == "iid"
        Σ₀ = Symmetric(Matrix(σ² .*I(nₚ)))
    elseif priors == "corr"
        θ = 0.01
        Σ₀ = Symmetric([σ² .*exp(-θ*abs.(i-j)) for i in 1:nₚ, j in 1:nₚ])
    end
    P = GaussianBelief(μ₀, Σ₀)
    # test correlated priors!
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # construct dynamics model (requires control vector)
    ## high values of paramnoise cause the model to forget faster 
    paramnoise = Symmetric(Matrix(0.00001 .*I(nₚ)))
    # ℱ = LinearDynamicsModel(1.0 .* I(nₚ), zeros(nₚ,3), paramnoise)
    ℱ = NonlinearDynamicsModel((p,u)->p, paramnoise)
    #----------
    # h needs to take in ts[k] as a control vector
    # I think this method uses ForwardDiff; ReverseDiff would be more efficient (UKF doesn't use derivatives, no?)
    # use the number of inputs in the NN to select how many u to feed in
    h = (p,u)->reconstruct(p)(u)
    # construct observation model
    ## high values of obsnoise cause the model to trust the data less
    obsnoise = Symmetric(Matrix(0.1 .*I(3))) # change for diff state space models
    𝒪 = NonlinearObservationModel(h, obsnoise)
    λ = 2.0
    α = 1.0
    β = 0.0
    filt = GaussianFilters.UnscentedKalmanFilter(ℱ, 𝒪, λ, α, β)
    # filt = GaussianFilters.ExtendedKalmanFilter(ℱ, 𝒪)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # training loop: ts is "action history" (control vector), 
    return run_filter(filt, P, ts[1:end-1], ts[2:end]-ts[1:end-1])
end  