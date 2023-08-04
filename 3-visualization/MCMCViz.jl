# using GaussianRandomFields 
using StatsBase, Random, Turing, Distributions
using LinearAlgebra, SpecialFunctions
# using Plots
using CairoMakie

# would be dope to add an accept/reject ratio, maybe using poly

dist = "horseshoe"

# split up the role of the MCMC pdf and the contour pdf?
#--- contour pdf: gives the z values corresponding to x
function posterior(x, dist = "gaussian")
    if dist == "gaussian"
        return exp(-sum((x./2).^2)) # Gaussian
    elseif dist == "laplace"
        return exp(-sum(abs.(x))) # Laplace
    elseif dist == "horseshoe"
        # return exp(-0.5*(x./σ).^2)/(σ*(1+σ^2)) 
        return prod(@. exp(x^2/2)*gamma(0,x^2/2))
        # Horseshoe
    end
end

# hmc: show projection of trajectory with momentum giving speed to sample. Plot as "lines". Drawn in 3 seconds? Fading tails!

# random walk MCMC 
L = 3000

# q  = [0.1,0.1]
# Q = zeros(2,L)
# P = zeros(2,L)
# A = zeros(L)
# σ = 0.5
# for i in 1:L
#     global q, Q, P, A, σ
#     # how big of a step σ? reduce it
#     q_ = rand(MvNormal(q,σ*I))
#     a = minimum([1, posterior(q_,dist)/posterior(q,dist)])
#     if rand() < a
#         q = q_
#         A[i] = 1
#     end
#     Q[:,i] = q
#     P[:,i] = q_
# end

# Plots.scatter(Q[1,:],Q[2,:],aspect_ratio = 1, color = :black, markersize = 1.5, legend = false, grid = false, showaxis = false, ticks = false, xlims = [-10,10], ylims = [-10,10])

# add a point at a time, red or green for reject or accept

extent = 5
# ϵ = 0.25
x = -extent:0.1:extent
y = -extent:0.1:extent
grid = Iterators.product(x,y) |> collect

points = Observable(Point2f0[(0, 0)])
new_point = Observable(Point2f0[(0, 0)])
centroid = Observable(Point2f0[(0, 0)])
Σ = Observable(Matrix{Float32}(I, 2, 2))
color = Observable(:black)
arrowpos = Observable(Point2f0([NaN32, NaN32]))
arrowdir = Observable([0.0])

ratio = Observable([extent/2, extent])

ellipse = lift((Σ,centroid) -> [Point2f0(Σ*[cos(t), sin(t)]) + centroid[1] for t in range(0, stop = 2π, length = 100)], Σ, centroid)

fig = Figure(resolution = (600, 600), fontsize = 24)
# log-scale the contours, coincide with \sigma
ax = Axis(fig[1,1], aspect = 1, backgroundcolor = :transparent, legend = false, xlims = [-extent,extent], ylims = [-extent,extent])
    hidedecorations!(ax)
    hidespines!(ax)
    CairoMakie.poly!(ax, ellipse, color = :lemonchiffon)
    # light yellow ellipse for the empircal standard deviation
    if dist == "gaussian"
        CairoMakie.contour!(x, y, log.(posterior.(grid,dist)), linewidth = 4, levels = -((1:1:5)./2).^2)
    elseif dist == "laplace"
        CairoMakie.contour!(x, y, log.(posterior.(grid,dist)), linewidth = 4, levels = -(1:1:5))
    elseif dist == "horseshoe"
        CairoMakie.contour!(x, y, log.(posterior.(grid,dist)), linewidth = 4, levels = 1:-1:-3)
    end
    CairoMakie.scatter!(ax, arrowpos, color = color, marker = '↑', rotations = arrowdir, markersize = 40)

    CairoMakie.scatter!(ax, points, aspect_ratio = 1, color = :black, markersize = 8, legend = false, grid = false, showaxis = false, ticks = false, xlims = [-extent,extent], ylims = [-extent,extent])
    CairoMakie.scatter!(ax, new_point, color = color, markersize = 12)
    CairoMakie.scatter!(ax, centroid, color = :goldenrod1, marker = 'M', markersize = 40, strokewidth=.5, strokecolor=:black) # I think E might be more clear
    CairoMakie.lines!(ax, ellipse, color = :goldenrod1, linewidth = 4, linestyle = :dash)

    # acceptance percentage bar 
    CairoMakie.lines!(ax, [extent/2, extent], [-extent, -extent], linewidth = 20, color = :red)
    CairoMakie.lines!(ax, ratio, [-extent, -extent], linewidth = 20, color = :lawngreen)

frames = 1:L

record(fig, "random-walk_$(dist).mp4", frames;
        framerate = 15) do frame
    if frame>1
        points[] = push!(points[], Point2f0(vec(Q[:,frame-1])))
        centroid[] = [Point2f0(vec(mean(Q[:,1:frame-1], dims = 2)))]
    end

    if frame>2
        Σ[] = sqrt(cov(Q[:,1:frame-1]'))
    end

    new_point[] = [Point2f0(vec(P[:,frame]))] # green, red
    if A[frame] == 1
        color[] = :lawngreen
    else
        color[] = :red
    end

    ratio[] = extent/2 .+ (extent/2).*[0, sum(A[1:frame])/frame]

    # add in arrow
    if Q[1,frame] > extent
        arrowpos[] = Point2f0([extent, Q[2,frame]])
        if Q[1,frame] > Q[1,frame-1]
            arrowdir[] = [-pi/2]
        else
            arrowdir[] = [pi/2]
        end
    elseif Q[1,frame] < -extent
        arrowpos[] = Point2f0([-extent, Q[2,frame]])
        if Q[1,frame] > Q[1,frame-1]
            arrowdir[] = [-pi/2]
        else
            arrowdir[] = [pi/2]
        end
    elseif Q[2,frame] > extent
        arrowpos[] = Point2f0([Q[1,frame], extent])
        if Q[2,frame] > Q[2,frame-1]
            arrowdir[] = [0]
        else
            arrowdir[] = [pi]
        end
    elseif Q[2,frame] < -extent
        arrowpos[] = Point2f0([Q[1,frame], -extent])
        if Q[2,frame] > Q[2,frame-1]
            arrowdir[] = [0]
        else
            arrowdir[] = [pi]
        end
    else
        arrowpos[] = Point2f0([NaN32, NaN32])
    end
end

# batch importance sampling for BNN inference