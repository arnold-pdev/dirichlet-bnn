using Statistics
using SpecialFunctions, Distributions
using Flux, Flux.Optimise
using Flux: params
using Base.Iterators: partition
using LinearAlgebra
using CUDA
using CSV, BSON, DataFrames
using Plots, CairoMakie

# read in CSV
# u1 = CSV.read("t0_Lorenz.csv", DataFrame)
u1 = CSV.read("ts_Lorenz.csv", DataFrame)
u1 = [collect(v) for v in eachrow(u1)]
# du = CSV.read("dt0p5_Lorenz.csv", DataFrame)
# du = [collect(v) for v in eachrow(du)]
du = u1[2:end] - u1[1:end-1]
u1 = u1[1:end-1]
n = size(u1,1) - 500

# partition data 
train = ([(u1[i], du[i]) for i in partition(1:n,100)]) |> gpu
valset = (n+1):(n+500)
val_u1 = u1[valset] |> gpu
val_u2 = du[valset] |> gpu

# Architecture
N = 16 # number of nodes in each hidden layer
Nd = 2 # depth
ℳ = Chain(
layer1 = Dense(3 => N, tanh),
# Dense(N => N, tanh),
# Dense(N => N, tanh),
# Dense(N => N, tanh),
# Dense(N => N, tanh),
# Dense(N => N, tanh),
output_layer = Dense(N => 3)
) |> gpu

# define loss function
ℓ(x1, x2) = (ℳ(x1) - x2)'*(ℳ(x1) - x2) # partial loss
Loss(x1, x2) = sum(ℓ.(x1,x2)) # total loss

# optimizer
opt = Adam()

epochs = 20000 # number of epochs
# training loop
for epoch = 1:epochs 
    for d in train 
        gs = gradient(params(ℳ)) do 
            l = Loss(d...)
        end 
        Optimise.update!(opt, params(ℳ), gs)
    end 
    # print accuracy wrt validation set
    @show Loss(val_u1, val_u2)
end

# change naming scheme so that depth and width are automatically recorded
L = Int(floor(Loss(val_u1, val_u2)))
BSON.@save "lorenzd-$(N)x$(Nd)-L$(L)-tanh.bson" model_state = ℳ

# # plot results
# err = u2 .- ℳ.(u1)
err = du .- ℳ.(u1)
Plots.scatter(sort(norm.(err)),title="Sorted MSE of Model")
savefig("errord-lin-$(N)x$(Nd)-L$(L)-tanh.png")

Plots.scatter(sort(log.(norm.(err))),title="Sorted log MSE of Model")
savefig("errord-log-$(N)x$(Nd)-L$(L)-tanh.png")

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