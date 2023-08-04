using Statistics
using SpecialFunctions, Distributions, DistributionsAD, GaussianMixtures
using Flux, Flux.Optimise
using Flux: params
using Base.Iterators: partition
using LinearAlgebra
using Metal
using CSV, BSON, DataFrames

function ℳ2gmm(ℳout)
    # https://github.com/davidavdav/GaussianMixtures.jl
    # w, a Vector of length m with the weights of the mixtures
    # μ, a matrix of m by d means of the Gaussians
    # Σ, either a matrix of m by d variances of diagonal Gaussians, or a vector of m Triangular matrices of d by d, representing the Cholesky decomposition of the full covariance
    # hist a vector of History objects describing how the GMM was obtained. (The type History simply contains a time t and a comment string s)

    w = ℳout[1]
    m = length(w)
    d = Int64(length(ℳout[2])/m)
    μ = reshape(ℳout[2],(m, d))
    D = reshape(ℳout[3],(m, d))
    L = reshape(ℳout[4],(m,:))
    # Σ = [ UpperTriangular(zeros(d,d)) for i in 1:m ]
    N = Int64(d*(d-1)/2)
    Σ = [ UpperTriangular([ D[i,j]*(j==k) + L[i,relu(j-k-1+(k-1)*d-Int64(k*(k-1)/2))+1]*(j>k)    for k in 1:d, j in 1:d]) for i in 1:m ]

    return GMM(w, μ, Σ, [], 0)
end

function ℳ2mm(ℳout)
    # Σ doesn't agree with the above... but idk why. What I've done here seems accurate.
    w = ℳout[1]
    m = length(w)
    d = Int64(length(ℳout[2])/m)
    μ = reshape(ℳout[2],(m, d))
    D = reshape((ℳout[3]).^2,(m, d))
    L = reshape(ℳout[4],(m,:))
    Σ = [ UpperTriangular([ D[i,j]*(j==k) + 
    L[i,relu(j-k-1+(k-1)*d-Int64(k*(k-1)/2))+1]*(j>k)    for k in 1:d, j in 1:d]) for i in 1:m ]

    parameters = [(vec(μ[i,:]),(Σ[i]'*Σ[i]) + 1e-8*I) for i in 1:m]

    return DistributionsAD.MixtureModel(DistributionsAD.MvNormal,parameters,w)
end

function xμTΛxμ(x, μ, ciΣ)
    # good
    (nₓ, d) = size(x)
    (x - repeat(μ',nₓ)) * ciΣ'
end

function llpm(gmm::GMM, x::Matrix)
    (nₓ, d) = size(x)
    ng = gmm.n
    d == gmm.d || error("Inconsistent size gmm and x")
    
    normalization = [0.5d*log(2π) - sum(log.(diag((gmm.Σ[k])))) for k in 1:ng]
    Δ = [ xμTΛxμ(x, vec(gmm.μ[k,:]), gmm.Σ[k]) for k in 1:ng]

    # return [-0.5 * sum(abs2, Δ[k], dims=2) .- normalization[k] for k in 1:ng] #ll
    return hcat([-0.5 * sum(abs2, Δ[k], dims=2) .- normalization[k] for k in 1:ng]...) #ll
end

function logsumexpw(x::Matrix, w::Vector)
    y = x .+ log.(w)'
    return logsumexp(y)
end

function negloglike(gmm::GMM, y::AbstractArray)
    loglk = llpm(gmm,reshape(y,1,length(y)))
    # return -log((lk*gmm.w)[1])
    return -logsumexpw(loglk,gmm.w)
end

# define loss function
function Loss(x,y) 
    Γ =  ℳ2gmm.(ℳ.(x)) # create a GMM for each data point
    # Γ =  ℳ2mm.(ℳ.(x))
    # negative log-likelihood
    return mean(negloglike.(Γ,y))
    # return mean(-loglikelihood.(Γ,y))
end