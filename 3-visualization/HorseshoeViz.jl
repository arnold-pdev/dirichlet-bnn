# using GaussianRandomFields 
using StatsBase, Random, Turing, Distributions
using LinearAlgebra
using Plots

# generate a Gaussian random field with squared exponential covariance
# cov = CovarianceFunction(2, Gaussian(0.3))
# # generate a grid of points
# pts = range(0, stop=1, length=401)
# # generate a random field with this covariance
# grf = GaussianRandomField(cov, CirculantEmbedding(), pts, pts, minpadding=1000)

# z = GaussianRandomFields.sample(grf)

# use Turing to sample from a Horseshoe prior 
# @model function horseshoe()

# just fucking sample from Horseshoe prior
σ = rand(Cauchy(0,1),10_000_000)
θ1 = vcat(rand.(Normal.(0,σ.^2),1)...)
σ = rand(Cauchy(0,1),10_000_000)
θ2 = vcat(rand.(Normal.(0,σ.^2),1)...)
scatter(θ1[1:1_000_000],θ2[1:1_000_000],xlims = [-10,10],ylims = [-10,10], xaxis=:log, yaxis=:log, markersize=1,color=:black)
savefig("horseshoo.png")

# also make a posterior sample for a linear regression -- a percentage for "no correlation" or "no bias" is provided.
M = 25 # number of PCA modes
N = 5 # number of nodes in each hidden layer
𝒪 = 9 # number of output nodes
ϵ = 1e-3 # small number to avoid log(0)

α = Dense(1 => 1) |> gpu

# plot the field
surface(pts, pts, z, camera=(90,90), colorbar=false, size=(600,600), xlabel="x", ylabel="y", zlabel="z")
xflip!(true)

# sample the field by StatsBase 
grid = Iterators.product(pts,pts) |> collect
wv = ProbabilityWeights(vec(reshape(exp.(-z)/sum(exp.(-z)),:,1)))
p = StatsBase.sample(Random.GLOBAL_RNG,grid,wv,10000)

scatter(p, camera=(90,90), markersize = 1, color = :black, colorbar=false, size=(600,600), legend = false, grid = false, showaxis = false)

function posterior(x)
    # return exp(-x'*x /2) # Gaussian
    # return exp(-sqrt(x'*x))# weird dumb thing
    return exp(-sum(abs.(x))) # Laplace
end

# random walk MCMC 
L = 10000

q  = [0,0]
Q = zeros(2,L)
A = zeros(L)
σ = 1.0
for i in 1:L
    # how big of a step σ?
    q_ = rand(MvNormal(q,σ*I))
    a = minimum([1, posterior(q_)/posterior(q)])
    if rand() < a
        q = q_
        A[i] = 1
    end
    Q[:,i] = q
end

scatter(Q[1,:],Q[2,:],aspect_ratio = 1, color = :black, markersize = 1, legend = false, grid = false, showaxis = false)

# make an animation

# global-local MCMC 
# q = [0,0] 
# 