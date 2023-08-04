using Statistics
using SpecialFunctions, Distributions
using Flux, Flux.Optimise
using Flux: params
using Base.Iterators: partition
using LinearAlgebra
using CUDA
using Plots

N = 5 # number of nodes in each hidden layer

α = Chain(
Dense(1 => N, relu),
Dense(N => N, relu),
Dense(N => N, relu),
Dense(N => N, relu),
Dense(N => N, relu),
Dense(N => 3, exp)
) |> gpu

# define loss function
loss(x, y) = sum(log.(gamma.(α(x)))) - log(gamma(sum(α(x)))) - (α(x) .- 1)' * log.(y) # partial loss
Loss(x, y) = sum(loss.(x,y)) # total loss
opt = Adam()

# accuracy(x, y) = maximum(norm.(α.(x)./sum.(α.(x)).-y,1)) # accuracy function MAE (consider changing)
accuracy(x, y) = sum(abs.(norm.(α.(x)./sum.(α.(x)).-y,1)))

epochs = 400 # number of epochs
# training loop
for epoch = 1:epochs 
    for d in train 
        gs = gradient(params(α)) do 
            l = Loss(d...)
        end 
        update!(opt, params(α), gs)
    end 
    # print accuracy wrt validation set
    @show accuracy(valX, valY)
    @show Loss(valX, valY)
end

# plot results
xt = [ train_x[i][1] for i in eachindex(train_x) ]
x = 0.0:0.01:1.0
for j in 1:3
    # plot mean generating training data
    μ = [ map_xy(x[i])[j] for i in eachindex(x) ]./([ sum(map_xy(x[i])) for i in eachindex(x) ])
    plot(x, μ, label = "μ$(j)")

    # plot training data 
    yt = [ train_y[i][j] for i in eachindex(train_x) ]
    scatter!(xt, yt, label = "training data")

    # plot model
    a = [ α([x[i]])[j] for i in eachindex(x) ]./([ sum(α([x[i]])) for i in eachindex(x) ])
    plot!(x, a, label = "model")

    savefig("model_fit_$(j).png")
end