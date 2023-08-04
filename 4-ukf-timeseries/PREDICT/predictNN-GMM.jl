using Statistics
using SpecialFunctions, Distributions, GaussianMixtures
using Flux, Flux.Optimise
using Flux: params
using Base.Iterators: partition
using LinearAlgebra
using Metal
using CSV, BSON, DataFrames

include("GMMFunctions.jl")

# read in CSV
u1 = CSV.read("invariant-measure/t0_Lorenz.csv", DataFrame)
# u1 = CSV.read("ts_Lorenz.csv", DataFrame)
u1 = [collect(v) for v in eachrow(u1)]
du = CSV.read("invariant-measure/dt0p5_Lorenz.csv", DataFrame)
du = [collect(v) for v in eachrow(du)]
# du = u1[2:end] - u1[1:end-1]
# u1 = u1[1:end-1]
n = size(u1,1) - 500

# partition data 
train = ([(u1[i], du[i]) for i in partition(1:n,100)]) |> gpu
valset = (n+1):(n+500)
val_u1 = u1[valset] |> gpu
val_u2 = du[valset] |> gpu

# Architecture
m = 2 # number of Gaussian modes
N = 16 # number of nodes in each hidden layer
Nd = 2 # depth
d = 3 # dimension of data

struct Split{T}
    paths::T
end

Split(paths...) = Split(paths)

Flux.@functor Split

(m::Split)(x::AbstractArray) = map(f -> f(x), m.paths)

ℳ = Chain(
Dense(3 => N, tanh),
Dense(N => N, tanh),
# output layer = [w μ d ℓ]
Split( Chain(Dense(N => m), softmax), # weights, the mixture %
      Dense(N => d*m),        # μ, the mix means
      Dense(N => d*m, exp),   # D, the mix diags
      Dense(N => Int64(m*d*(d-1)/2)))# ℓ, the mix lower tri
) |> gpu
# structure: {p₁...pₘ,{μ}ₘ,{C₍₁₁₎...C₍₃₃}}
# Can the function just output a structure using functor?
# https://fluxml.ai/Flux.jl/stable/models/functors/
# https://fluxml.ai/Flux.jl/stable/models/advanced/

# BSON.@load "lorenzd-16x2-L=best-tanh-gmm.bson" model_state
# ℳ = model_state

# optimizer
opt = Adam()
# opt = Descent(0.01)

epochs = 1000 # number of epochs
# training loop
best = 1e6
for epoch = 1:epochs 
    for data in train 
        gs = gradient(Flux.params(ℳ)) do 
            l = Loss(data...)
        end 
        Optimise.update!(opt, Flux.params(ℳ), gs)
    end 
    # print accuracy wrt validation set
    @show Loss(val_u1, val_u2)
    if Loss(val_u1, val_u2) < best 
        # L = Int(floor(Loss(val_u1, val_u2)))
        BSON.@save "lorenzd-$(N)x$(Nd)-L=best-tanh-gmm.bson" model_state = ℳ
    end
end

# change naming scheme so that depth and width are automatically recorded
# L = Int(floor(Loss(val_u1, val_u2)))
# BSON.@save "lorenzd-$(N)x$(Nd)-L$(L)-tanh-gmm.bson" model_state = ℳ

# # plot results
# err = u2 .- ℳ.(u1)
# err = du .- ℳ.(u1)
# Plots.scatter(sort(norm.(err)),title="Sorted MSE of Model")
# savefig("errord-lin-$(N)x$(Nd)-L$(L)-tanh.png")

# Plots.scatter(sort(log.(norm.(err))),title="Sorted log MSE of Model")
# savefig("errord-log-$(N)x$(Nd)-L$(L)-tanh.png")

# pts1 = Point3f0.([u1[i][1] for i in eachindex(u1)],[u1[i][2] for i in eachindex(u1)],[u1[i][3] for i in eachindex(u1)])

# f = Figure()
# ax_xy = CairoMakie.Axis(f[1,1])
#     CairoMakie.scatter!(ax_xy,pts1, color=:black,  markersize = 1)
#     CairoMakie.scatter!(ax_xy,pts2, color=:red,  markersize = 1)
#     CairoMakie.scatter!(ax_xy,ptsu1, color=:green,  markersize = 1)#
#     CairoMakie.scatter!(ax_xy,ptsu2, color=:blue,  markersize = 1)#
# f

# U = Vector([u1])
# for i = 1:10
# push!(U,U[i].+ℳ.(U[i]))
# end