using GaussianFilters, Flux, LinearAlgebra

# Advance the parameters of a neural network emulating an ODE solver
# inputs:
#        priors, specifications for the parameter priors
#        f,  a function that advances the state as the true dynamics
#        𝒩𝒩, a Flux chain to be trained to emulate f(u)-u

# PRIORS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Specify the priors and prepare for inference.

# Extract weights and a helper function to reconstruct 𝒩𝒩 from weights
parameters_initial, reconstruct = Flux.destructure(𝒩𝒩)
nₚ = length(parameters_initial) # number of parameters in 𝒩𝒩

# cast as Gaussian belief for GaussianFilters.jl
# expand to include state variables
μ₀ = zeros(nₚ)
Σ₀ = 1.0 .* I(nₚ)
P = GaussianBelief(μ₀, Σ₀)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# DYNAMICS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
σ = 10.0
β = 8/3
ρ = 28.0
Δt = 0.5
function lorenz!(du, u, p, t)
    du[1] = p[1] * (u[2] - u[1])
    du[2] = u[1] * (p[3] - u[3]) - u[2]
    du[3] = u[1] * u[2] - p[2] * u[3]
    nothing
end

function f!(u, Δt)
    prob = ODEProblem(lorenz!, u, (0.0, 10.0), [σ, β, ρ])
    nothing
end
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# specify h to be identity on the state variables
F = (u,p) -> (u .+ 𝒩𝒩(u),p)
# construct dynamics model 
ℱ = NonlinearDynamicsModel(F, noise)
H = (u,p) -> (u,0)
# construct observation model
𝒪 = LinearObservationModel(H, noise)
filt = GaussianFilters.UnscentedKalmanFilter(ℱ, 𝒪, 2.0, α, β)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~