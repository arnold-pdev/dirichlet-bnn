# using GaussianRandomFields 
using StatsBase, Random, Turing, Distributions
using LinearAlgebra, SpecialFunctions
using DifferentialEquations, DiffEqPhysics, OrdinaryDiffEq
# using Plots
using CairoMakie, ColorSchemes

dist = "horseshoe2"

function Log(x)
    if x > 1e-6
        return log(x)
    else
        return -Inf
    end
end

# split up the role of the MCMC pdf and the contour pdf?
#--- contour pdf: gives the z values corresponding to x
function posterior(x, dist = "gaussian")
    if dist == "gaussian"
        return exp(-sum((x./2).^2)) # Gaussian
    elseif dist == "laplace"
        return exp(-sum(abs.(x))) # Laplace
    elseif dist == "horseshoe"
        # matches numerical integration perfectly!
        return prod(@. exp(x^2/2)*gamma(0,x^2/2))
        # Horseshoe
    elseif dist == "horseshoe2"
        θ = x[1]
        σ = x[2]
        return exp(-0.5*(θ/σ)^2)/(abs(σ)*(1+σ^2))
    elseif dist == "horseshoe3"
        # add in log barrier to keep σ > 0
        β = 1e-5
        θ = x[1]
        σ = x[2]
        return exp(-0.5*(θ/σ)^2)/(abs(σ)*(1+σ^2)) + β*Log(σ)
    elseif dist == "horseshoe4"
        # add in log barrier to keep σ > 0
        β = 1e-5
        θ = x[1]
        p = x[2] # precision
        return exp(-0.5*(θ*p)^2)/((1+p^2))*abs(p^3) + β*Log(p)
    elseif dist == "precision"
        θ = x[1]
        p = x[2] # precision
        dp = 1.0
        return exp(-0.5*(θ*p)^2)*abs(p)/(1+p^2)*dp
    elseif dist == "horse-exp"
        θ = x[1]
        p = exp(-x[2])
        dp= p
        return exp(-0.5*(θ*p)^2)*abs(p)/(1+p^2)*dp
    elseif dist == "horse-exp2"
        θ = x[1]
        p = exp(-x[2]^2)
        dp= 2*abs(x[2])*p
        return exp(-0.5*(θ*p)^2)*abs(p)/(1+p^2)*dp
    elseif dist == "exp-tilt" # beautiful, convex, but asymmetric
        θ = exp(x[1])
        p = exp(-x[2])
        dp= p
        dθ= θ
        return exp(-0.5*(θ*p)^2)*abs(p)/(1+p^2)*dp*dθ
    elseif dist == "exp-tilt-otro" 
        θ = exp(x[1])
        p = exp(x[2])
        dp= p
        dθ= θ
        return exp(-0.5*(θ*p)^2)*abs(p)/(1+p^2)*dp*dθ
    elseif dist == "sinh-tilt" # this is the one
        θ = sinh(x[1])
        p = exp(-x[2])
        dp= p
        dθ= cosh(x[1])
        return exp(-0.5*(θ*p)^2)*abs(p)/(1+p^2)*dp*dθ
    elseif dist == "sinh-prec"
        θ = sinh(x[1])
        p = x[2]
        dp= 1
        dθ= cosh(x[1])
        return exp(-0.5*(θ*p)^2)*p/(1+p^2)*dp*dθ
    elseif dist == "sinh-tilt2"
        θ = sinh(x[1])
        p = exp(x[2]^2)
        dp= 2*x[2]*p
        dθ= cosh(x[1])
        return exp(-0.5*(θ*p)^2)*p/(1+p^2)*dp*dθ
    elseif dist == "psinh" # 
        θ = x[1]
        p = sinh(x[2])
        dθ= 1.0
        dp= cosh(x[2])
        return exp(-0.5*(θ*p)^2)*abs(p)/(1+p^2)*dp*dθ
    elseif dist == "dubsinh2" # lots of divergence
        θ = sinh(x[1])
        p = sinh(x[2])
        dθ= cosh(x[1])
        dp= cosh(x[2])
        return exp(-0.5*(θ*p)^2)*abs(p)/(1+p^2)*dp*dθ
    elseif dist == "dubsinh2n" # try removing singularity
        θ = sinh(x[1])
        p = sinh(x[2])
        dθ= cosh(x[1])
        dp= cosh(x[2])
        return exp(-0.5*(θ*p)^2)*p/(1+p^2)*dp*dθ
    elseif dist == "dubsinhsinh2" # idea is super-exponential tilting
        θ = sinh(sinh(x[1]))
        p = sinh(sinh(x[2]))
        dθ= cosh(sinh(x[1]))*cosh(x[1])
        dp= cosh(sinh(x[2]))*cosh(x[2])
        return exp(-0.5*(θ*p)^2)*p/(1+p^2)*dp*dθ
    elseif dist == "dubsinhfuck" # kinda works?
        θ = sinh(x[1]^3)
        p = cosh(x[2]^3)-1.0
        dθ= 3*x[1]^2*cosh(x[1]^3)
        dp= 3*x[2]^2*sinh(x[2]^3)
        return exp(-0.5*(θ*p)^2)*abs(p)/(1+p^2)*dp*dθ
    elseif dist == "dubsinhfucker" #horrendous, lol (all rejected)
        θ = sinh(x[1])
        p = cosh(x[2])-1.0
        dθ= cosh(x[1])
        dp= sinh(x[2])
        return exp(-0.5*(θ*p)^2)*abs(p)/(1+p^2)*dp*dθ
    end
end

# add in two views of the posterior for horseshoe2 and horseshoe3, so that we're comparing apples to apples (same number of parameters)

# hmc: show projection of trajectory with momentum giving speed to sample. Plot as "lines". Drawn in 4 steps? Fading tails!

# random walk MCMC 
L = 500

q0 = q  = [0.1,0.1]
Q = zeros(2,L)
P = zeros(2,L)
A = zeros(L)
σ = 0.5
nsteps = 1000

# defining the hamiltonian
T(p,q) = sum(p.^2)/2 # use mass matrix M
H(p,q,parameters) = T(p,q) -Log(posterior(q,dist))
τ = 10.0 # duration

traj = Vector{Vector{Point2f0}}(undef, L)

# E = Vector{Float32}(undef, L)
# E_BMFI= Vector{Float32}(undef, L)

for i in 1:L
    global q, Q, P, traj, A, σ, E, Ē, E_
    local q_, p
    # sample momentum 
    p = rand(MvNormal(zeros(2),I)) # this is disconnected from my def of KE
    # integrate Hamiltonian dynamics
    prob = HamiltonianProblem(H, p, q, (0.0, τ))
    sol = solve(prob, VerletLeapfrog(), dt = τ/nsteps, saveat = τ/nsteps)

    traj[i] = [ Point2f0(sol.u[i][3:4]) for i in eachindex(sol.u) ]
    q_ = sol.u[end][3:4]

    a = minimum([1, exp(-H(sol.u[end][1:2],sol.u[end][3:4],0) + H(sol.u[1][1:2],sol.u[1][3:4],0))])   
    
    # E[i] = H(sol.u[1][1:2],sol.u[1][3:4],0)
    # Ē = E[1:i] |> mean
    # if i==1
    #     E_BMFI[i] = 1
    # else
    #     E_BMFI[i] = [(E[k]-E[k-1])^2 for k in 2:i]/[(E[k]-Ē)^2 for k in 1:i] |> mean
    # end

    if rand() < a
        q = q_
        A[i] = 1 
    end
    # compute E-BFMI 

    Q[:,i] = q
    P[:,i] = q_
end

# add a point at a time, red or green for reject or accept

extent = 5
if dist == "horseshoe3" || dist == "horseshoe4"
    lowextent = 0
    highextent = extent
else
    lowextent = -extent
    highextent = extent
end
# ϵ = 0.25
x = -extent:0.1:extent
y = lowextent:0.1:highextent
grid = Iterators.product(x,y) |> collect

points = Observable([Point2f0(q0)])
# new_point = Observable(Point2f0[(0, 0)])
centroid = Observable(Point2f0[q0])
Σ = Observable(Float32.(ones(2,2)))

# arrowpos = Observable(Point2f0([NaN32, NaN32]))
# arrowdir = Observable([0.0])
# cmap = Observable(:OrRd_7)
cmap = Observable(:red)
tail = Observable(Point2f0[(NaN32, NaN32)])
ratio = Observable([extent/2, extent])

ellipse = lift((Σ,centroid) -> [Point2f0(Σ*[cos(t), sin(t)]) + centroid[1] for t in range(0, stop = 2π, length = 100)], Σ, centroid)

fig = Figure(resolution = (600, 600), fontsize = 24)
# log-scale the contours, coincide with \sigma

ax = Axis(fig[1,1], aspect = 1, backgroundcolor = :transparent, legend = false, xlims = [-extent,extent], ylims = [lowextent,highextent])
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
    elseif dist == "horseshoe2"
        CairoMakie.contour!(x, y, log.(posterior.(grid,dist)), linewidth = 4, levels = -1:-1:-5)
    else
        CairoMakie.contour!(x, y, Log.(posterior.(grid,dist)), linewidth = 4, levels = -1:-1:-5)
    end

    CairoMakie.scatter!(ax, points, aspect_ratio = 1, color = :black, markersize = 8, legend = false, grid = false, showaxis = false, ticks = false, xlims = [-extent,extent], ylims = [lowextent,highextent])
    CairoMakie.scatter!(ax, centroid, color = :goldenrod1, marker = 'M', markersize = 40, strokewidth=.5, strokecolor=:black) # I think E might be more clear
    CairoMakie.lines!(ax, ellipse, color = :goldenrod1, linewidth = 4, linestyle = :dash)

    # change to a color map if I feel compelled
    CairoMakie.lines!(ax, tail, color = cmap, linewidth = 4)

    # acceptance percentage bar 
    CairoMakie.lines!(ax, [extent/2, extent], [lowextent, lowextent], linewidth = 20, color = :red)
    CairoMakie.lines!(ax, ratio, [lowextent, lowextent], linewidth = 20, color = :lawngreen)
tailduration = 10
frames = 0:(L*tailduration - 1)

segments = div.((nsteps+1)*(1:tailduration),tailduration)
record(fig, "hmc_$(dist)-true.mp4", frames;
        framerate = 60) do frame
        
        # timing
        n = div(frame,tailduration)
        j = frame - n*tailduration

        if j == 0
            tail[] = Point2f0[(NaN32, NaN32)]
            if A[n+1] == 1
                # cmap[] = :algae
                cmap[] = :lawngreen
            else
                # cmap[] = :OrRd_7
                cmap[] = :red
            end
            if n>0
                points[] = push!(points[], Point2f0(vec(Q[:,n])))
                centroid[] = [Point2f0(vec(mean(Q[:,1:n], dims = 2)))]
            end
            if n>1
                Σ[] = sqrt(cov(Q[:,1:n]'))
            end
            ratio[] = extent/2 .+ (extent/2).*[0, sum(A[1:n])/n]
        else 
            tail[] = traj[n+1][1:minimum([segments[j],length(traj[n+1])])]
        end

end

# batch importance sampling for BNN inference