using AdvancedHMC, ForwardDiff
using LogDensityProblems, LogDensityProblemsAD
using LinearAlgebra

# Define the target distribution using the `LogDensityProblem` interface
struct LogTargetDensity
    dim::Int
end
LogDensityProblems.logdensity(p::LogTargetDensity, θ) = -sum(abs2, θ) / 2  # standard multivariate normal
LogDensityProblems.dimension(p::LogTargetDensity) = p.dim
LogDensityProblems.capabilities(::Type{LogTargetDensity}) = LogDensityProblems.LogDensityOrder{0}()

# Choose parameter dimensionality and initial parameter value
D = 10; initial_θ = rand(D)
ℓπ = LogTargetDensity(D)

# Set the number of samples to draw and warmup iterations
n_samples, n_adapts = 2_000, 1_000

# Define a Hamiltonian system
metric = DiagEuclideanMetric(D)
hamiltonian = Hamiltonian(metric, ℓπ, ForwardDiff)

# Define a leapfrog solver, with initial step size chosen heuristically
initial_ϵ = find_good_stepsize(hamiltonian, initial_θ)
integrator = Leapfrog(initial_ϵ)

# Define an HMC sampler, with the following components
#   - multinomial sampling scheme,
#   - generalised No-U-Turn criteria, and
#   - windowed adaption for step-size and diagonal mass matrix
proposal = NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator)
adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))

# Run the sampler to draw samples from the specified Gaussian, where
#   - `samples` will store the samples
#   - `stats` will store diagnostic statistics for each sample
samples, stats = sample(hamiltonian, proposal, initial_θ, n_samples, adaptor, n_adapts; progress=true)

#---------

# # Ensure that julia was launched with appropriate number of threads
# println(Threads.nthreads())

# # Number of chains to sample
# nchains = 4

# # Cache to store the chains
# chains = Vector{Any}(undef, nchains)

# # The `samples` from each parallel chain is stored in the `chains` vector 
# # Adjust the `verbose` flag as per need
# Threads.@threads for i in 1:nchains
#   samples, stats = sample(hamiltonian, proposal, initial_θ, n_samples, adaptor, n_adapts; verbose=false)
#   chains[i] = samples
# end

# #---------

# function sample(
#     rng::Union{AbstractRNG, AbstractVector{<:AbstractRNG}},
#     h::Hamiltonian,
#     κ::HMCKernel,
#     θ::AbstractVector{<:AbstractFloat},
#     n_samples::Int,
#     adaptor::AbstractAdaptor=NoAdaptation(),
#     n_adapts::Int=min(div(n_samples, 10), 1_000);
#     drop_warmup=false,
#     verbose::Bool=true,
#     progress::Bool=false,
# )