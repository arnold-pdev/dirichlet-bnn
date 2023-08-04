using Flux, Flux.Optimise
using Flux: params
using CairoMakie, Observables
using LinearAlgebra, Statistics
using Metal
using Metal_LLVM_Tools_jll

# limit to comparisons, or change the video aspect ratio 

# depths of NNs
# input a structured NN and learn layers, parameters from that

# send in a tuple β of Gaussian beliefs, indexed by p; how to handle a single history of beliefs vs a tuple of histories?
# β = (b1,b2)
function movie_ukf(𝒩𝒩::Chain{<:Union{Tuple, AbstractVector, NamedTuple}}, b::AbstractVector, ts, Δt = 0.5)
    # Vector{GaussianBelief{<:AbstractVector{<:Number},<:Symmetric{<:Number}}} <- couldn't get this to work for b
    # function for non-vector inputs
    # change the Observables to vectors of Observables
    # layers = [3, 12, 3]
    # can we visualize fxn space pobabilities? Probably best shown as a video: start with two nearby points and draw samples, matched by color.
    # continuous fxn space priors don't make sense for Lorenz because of bifurcation

    parameters_initial, reconstruct = Flux.destructure(𝒩𝒩)
    nᵢ = size(𝒩𝒩[1].weight,2)
    layers = [nᵢ]
    for layer in 𝒩𝒩
        push!(layers,size(layer.weight,1))
    end
    param_cnt = [(layers[i]+1)*layers[i+1] for i in 1:length(layers)-1]
    layer_sum = cumsum(param_cnt)
    node_sum  = cumsum(vcat([repeat([layers[i]+1], layers[i+1]) for i in 1:length(layers)-1]...))
    input_sum = cumsum(vcat([repeat([layers[i+1]], layers[i]+1) for i in 1:length(layers)-1]...))

    # extract first nᵢ entries of ts
    tslice = [ts[i][1:nᵢ] for i in eachindex(ts)]

    N = sum(param_cnt)

    time = Δt.*(0:length(ts)-2)

    param = Observable(b[1].μ)
    covar = Observable(b[1].Σ)
    ℓ2 = Observable(Vector{Float64}(undef, length(ts)-1))
    mse = Observable(0.0)
    now = Observable(0.0)
    state = Observable(Point3f0.(ts[1][1], ts[1][2], ts[1][3]))

    function cmap(j)
        if j % 3 == 1
            return :Blues
        elseif j % 3 == 2
            return :Greens
        else
            return :Reds
        end
    end

    fig = Figure(resolution = (1600, 500), font = "CMU Serif")
    for i in 1:1
        # b = β[i]
        # plot the dynamics of the lorenz system.
        ax_lorenz = CairoMakie.Axis(fig[i,1], title = "Lorenz system", ylabel = "y", xlabel = "x", aspect = AxisAspect(1))
            pts = Point3f0.([ts[i][1] for i in eachindex(ts)],[ts[i][2] for i in eachindex(ts)],[ts[i][3] for i in eachindex(ts)])
            CairoMakie.scatter!(ax_lorenz, pts, color = :white, markersize = 5, strokewidth = 0.5, strokecolor = :black)
            CairoMakie.scatter!(ax_lorenz, state, color = :red, markersize = 5)

        # plot the parameters of the NN. bin the parameters by layer.
        ax_param = CairoMakie.Axis(fig[i,2], title = "Parameter means μ", ylabel = "parameter value", xlabel = "parameter index", aspect = AxisAspect(1))
        #for j in eachindex(layers)
        j = 1
            # would be very swell to include y labels
            CairoMakie.ylims!(ax_param, -100, 100)
            CairoMakie.xlims!(ax_param, 0, N)
            # turn off gridlines
            hidedecorations!(ax_param)
            # CairoMakie.scatter!( ax_param, param, color = 1:N,  markersize = 5, colormap = cmap(j))
            CairoMakie.scatter!( ax_param, param, color = 1:N,  markersize = 5, colormap = :rainbow)
            # CairoMakie.vlines!(ax_param, 0.5:1.0:N+0.5, color = :gray)
            CairoMakie.vlines!(ax_param, input_sum .+0.5, color = :gray, linewidth = 0.25)
            CairoMakie.vlines!(ax_param, layer_sum .+0.5, color = :black)
            # consider labelling
        #end

        # plot the error norm of the NN as a function of now step. plot a vertical line at the now step where the NN is retrained.
        # set y limit at 50
        ax_ℓ2 = CairoMakie.Axis(fig[i,3], title = "ℓ2 error", ylabel = "ℓ2 error", xlabel = "time", aspect = AxisAspect(1))
            CairoMakie.ylims!(ax_ℓ2, 0, 50)
            CairoMakie.xlims!(ax_ℓ2, minimum(time), maximum(time))
            CairoMakie.scatter!( ax_ℓ2, time, ℓ2, color = :blue,  markersize = 2) # change color 
            CairoMakie.vlines!(ax_ℓ2, now, color = :black)
            CairoMakie.hlines!(ax_ℓ2, mse, color = :orange)

        # plot the covariance matrix of the NN as a heatmap. could subdivide the heatmap by layer.
        ax_covar = CairoMakie.Axis(fig[i,4], title = "Parameter covariance Σ", aspect = AxisAspect(1))
            # log scaling on heatmap?
            # hm = CairoMakie.heatmap!(ax_covar, covar, colormap = :inferno, colorrange = (-0.05*maximum(b[1].Σ), maximum(b[1].Σ)), lowclip = :green, highclip = :white)
            hm = CairoMakie.heatmap!(ax_covar, covar, colormap = :inferno, colorrange = (-0.05, 1.0), lowclip = :green, highclip = :white)
            Colorbar(fig[:, 5], hm)
            # segment parameter groups
            CairoMakie.vlines!(ax_covar,  input_sum .+0.5, color = :gray, linewidth = 0.5)
            CairoMakie.hlines!(ax_covar,  input_sum .+0.5, color = :gray, linewidth = 0.5)
            # segment layers
            CairoMakie.vlines!(ax_covar, layer_sum .+0.5, color = :white)
            CairoMakie.hlines!(ax_covar, layer_sum .+0.5, color = :white)
    end    
        
    function animstep!(i)
        now[] += Δt
        param[] = b[i].μ
        covar[] = b[i].Σ
        ℓ2[] = norm.(reconstruct(b[i].μ).(tslice[1:end-1]) - ts[2:end] + ts[1:end-1])
        mse[] = mean(ℓ2[])
        state[] = Point3f0.(ts[i][1], ts[i][2], ts[i][3])
    end

    anim = CairoMakie.record(fig, "UKF-lorenz.mp4", 1:length(ts)-1,framerate = 15) do i 
        animstep!(i)
    end
end

function movie_ukf(models::AbstractVector, β::Vector{AbstractVector}, ts, Δt = 0.5)
    # function for vector inputs 
    # change the Observables to vectors of Observables
    # layers = [3, 12, 3]
    𝒩𝒩 = models[m]
    b = β[m]

    layers = [size(𝒩𝒩[1].weight,2)]
    for layer in 𝒩𝒩
        push!(layers,size(layer.weight,1))
    end
    param_cnt = [(layers[i]+1)*layers[i+1] for i in 1:length(layers)-1]
    layer_sum = cumsum(param_cnt)
    node_sum  = cumsum(vcat([repeat([layers[i]+1], layers[i+1]) for i in 1:length(layers)-1]...))
    input_sum = cumsum(vcat([repeat([layers[i+1]], layers[i]+1) for i in 1:length(layers)-1]...))

    N = sum(param_cnt)

    time = Δt.*(0:length(ts)-2)

    param = Observable(b[1].μ)
    covar = Observable(b[1].Σ)
    ℓ2 = Observable(Vector{Float64}(undef, length(ts)-1))
    mse = Observable(0.0)
    now = Observable(0.0)
    state = Observable(Point3f0.(ts[1][1], ts[1][2], ts[1][3]))

    function cmap(j)
        if j % 3 == 1
            return :Blues
        elseif j % 3 == 2
            return :Greens
        else
            return :Reds
        end
    end

    fig = Figure(resolution = (1600, 500*length(β)), font = "CMU Serif")
    for i in 1:1
        # b = β[i]
        # plot the dynamics of the lorenz system.
        ax_lorenz = CairoMakie.Axis(fig[i,1], title = "Lorenz system", ylabel = "y", xlabel = "x", aspect = AxisAspect(1))
            pts = Point3f0.([ts[i][1] for i in eachindex(ts)],[ts[i][2] for i in eachindex(ts)],[ts[i][3] for i in eachindex(ts)])
            CairoMakie.scatter!(ax_lorenz, pts, color = :white, markersize = 5, strokewidth = 0.5, strokecolor = :black)
            CairoMakie.scatter!(ax_lorenz, state, color = :red, markersize = 5)

        # plot the parameters of the NN. bin the parameters by layer.
        ax_param = CairoMakie.Axis(fig[i,2], title = "Parameter means μ", ylabel = "parameter value", xlabel = "parameter index", aspect = AxisAspect(1))
        #for j in eachindex(layers)
        j = 1
            # would be very swell to include y labels
            CairoMakie.ylims!(ax_param, -100, 100)
            CairoMakie.xlims!(ax_param, 0, N)
            # turn off gridlines
            hidedecorations!(ax_param)
            # CairoMakie.scatter!( ax_param, param, color = 1:N,  markersize = 5, colormap = cmap(j))
            CairoMakie.scatter!( ax_param, param, color = 1:N,  markersize = 5, colormap = :rainbow)
            # CairoMakie.vlines!(ax_param, 0.5:1.0:N+0.5, color = :gray)
            CairoMakie.vlines!(ax_param, input_sum .+0.5, color = :gray, linewidth = 0.25)
            CairoMakie.vlines!(ax_param, layer_sum .+0.5, color = :black)
            # consider labelling
        #end

        # plot the error norm of the NN as a function of now step. plot a vertical line at the now step where the NN is retrained.
        # set y limit at 50
        ax_ℓ2 = CairoMakie.Axis(fig[i,3], title = "ℓ2 error", ylabel = "ℓ2 error", xlabel = "time", aspect = AxisAspect(1))
            CairoMakie.ylims!(ax_ℓ2, 0, 50)
            CairoMakie.xlims!(ax_ℓ2, minimum(time), maximum(time))
            CairoMakie.scatter!( ax_ℓ2, time, ℓ2, color = :blue,  markersize = 2) # change color 
            CairoMakie.vlines!(ax_ℓ2, now, color = :black)
            CairoMakie.hlines!(ax_ℓ2, mse, color = :orange)

        # plot the covariance matrix of the NN as a heatmap. could subdivide the heatmap by layer.
        ax_covar = CairoMakie.Axis(fig[i,4], title = "Parameter covariance Σ", aspect = AxisAspect(1))
            hm = CairoMakie.heatmap!(ax_covar, covar, colormap = :inferno, colorrange = (-0.05*maximum(b[1].Σ), maximum(b[1].Σ)), lowclip = :green, highclip = :white)
            Colorbar(fig[:, 5], hm)
            # segment parameter groups
            CairoMakie.vlines!(ax_covar,  input_sum .+0.5, color = :gray, linewidth = 0.5)
            CairoMakie.hlines!(ax_covar,  input_sum .+0.5, color = :gray, linewidth = 0.5)
            # segment layers
            CairoMakie.vlines!(ax_covar, layer_sum .+0.5, color = :white)
            CairoMakie.hlines!(ax_covar, layer_sum .+0.5, color = :white)
    end    
        
    function animstep!(i)
        now[] += Δt
        param[] = b[i].μ
        covar[] = b[i].Σ
        ℓ2[] = norm.(reconstruct(b[i].μ).(ts[1:end-1]) - ts[2:end] + ts[1:end-1])
        mse[] = mean(ℓ2[])
        state[] = Point3f0.(ts[i][1], ts[i][2], ts[i][3])
    end

    anim = CairoMakie.record(fig, "UKF-lorenz.mp4", 1:length(ts)-1,framerate = 15) do i 
        animstep!(i)
    end
end