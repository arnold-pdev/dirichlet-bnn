using Statistics
using SpecialFunctions, Distributions
using Flux, Flux.Optimise
using Flux: params
using Base.Iterators: partition
using LinearAlgebra
using CUDA
using Plots

# define sample function
map_xy(x) = [1.0, 0.0, 0.5]*x + [0.01, 0.5, 0.5]

# generate data of independent variable
train_x = [ rand(1) for i in 1:100 ]

# generate the Dirichlet parameters of the data
untrain_y = [map_xy(train_x[i][1]) for i in eachindex(train_x)]

# generate the actual y values by sampling from a Dirichlet distribution
train_y = [ rand(Dirichlet(30.0*untrain_y[i])) for i in eachindex(train_x) ]

# partition the data
train = ([(train_x[i], train_y[i]) for i in partition(1:90, 10)]) |> gpu
valset  = 91:100
valX = train_x[valset] |> gpu
valY = train_y[valset] |> gpu