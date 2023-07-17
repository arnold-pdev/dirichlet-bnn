using Statistics
using SpecialFunctions, Distributions
using Flux, Flux.Optimise
using Flux: params
using Base.Iterators: partition
using LinearAlgebra
using CSV, DataFrames, BSON
using Plots

# load data
# make sure to prepare the data by removing "< 0.01", etc
data = CSV.read("chemcam-data/ChemCam_PCA_composition.csv", DataFrame)
# replace 0.0 with 0.0001 in data_y
data_y = [ vcat([ maximum([data[i, j],0.001]) for j in 3:10 ], 
[maximum([0.001,100.0-sum([ data[i, j] for j in 3:10 ])])]) for i in 1:size(data)[1] ]
data_y = [ data_y[i]./sum(data_y[i]) for i in 1:size(data)[1] ]
data_x = [ [ data[i, j] for j in 11:35 ] for i in 1:size(data)[1] ]

# partition the data
train = ([(data_x[i], data_y[i]) for i in partition(1:size(data)[1], 10)]) |> gpu
valset  = 91:100
valX = data_x[valset] |> gpu
valY = data_y[valset] |> gpu

M = 25 # number of PCA modes
N = 5 # number of nodes in each hidden layer
𝒪 = 9 # number of output nodes
ϵ = 1e-3 # small number to avoid log(0)

α = Chain(
Dense(M => N, relu),
Dense(N => N, relu),
Dense(N => N, relu),
Dense(N => N, relu),
Dense(N => N, relu),
Dense(N => 𝒪, exp)
) |> gpu

# define loss function
Log(x) = log.(x .+ ϵ)
loss(x, y) = -loglikelihood(Dirichlet(α(x)),[y]) # y is giving domain error problems??
Loss(x, y) = sum(loss.(x,y)) # total loss
opt = Adam()

# accuracy(x, y) = maximum(norm.(α.(x)./sum.(α.(x)).-y,1)) # accuracy function MAE (consider changing)
accuracy(x, y) = sum(abs.(norm.(α.(x)./sum.(α.(x)).-y,1)))

epochs = 1000 # number of epochs
# training loop
for epoch = 1:epochs 
    for d in train 
        gs = gradient(params(α)) do 
            l = Loss(d...)
        end 
        update!(opt, params(α), gs)
        # println( maximum([ maximum(abs.(params(α)[i])) for i in 1:length(params(α)) ]) )
    end 
    # print accuracy wrt validation set
    @show accuracy(valX, valY)
    @show Loss(valX, valY)
    if isnan(Loss(valX, valY))
        break
    else
        BSON.@save "dirichlet-model7.bson" model_state = α
    end
end

# # plot results
# xt = [ train_x[i][1] for i in eachindex(train_x) ]
# x = 0.0:0.01:1.0
# for j in 1:3
#     # plot mean generating training data
#     μ = [ map_xy(x[i])[j] for i in eachindex(x) ]./([ sum(map_xy(x[i])) for i in eachindex(x) ])
#     plot(x, μ, label = "μ$(j)")

#     # plot training data 
#     yt = [ train_y[i][j] for i in eachindex(train_x) ]
#     scatter!(xt, yt, label = "training data")

#     # plot model
#     a = [ α([x[i]])[j] for i in eachindex(x) ]./([ sum(α([x[i]])) for i in eachindex(x) ])
#     plot!(x, a, label = "model")

#     savefig("model_fit_$(j).png")
# end