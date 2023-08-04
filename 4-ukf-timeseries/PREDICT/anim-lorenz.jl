using Flux, Flux.Optimise
using Flux: params
using CairoMakie, Observables
using DifferentialEquations.EnsembleAnalysis

u0 = sort(u1)
N = length(u1)
cmap = :rainbow

u = Observable(u0)
pts_u = lift(u->Point3f0.([u[i][1] for i in eachindex(u)],[u[i][2] for i in eachindex(u)],[u[i][3] for i in eachindex(u)]), u)

w = Observable(u0)
pts_w = lift(u->Point3f0.([u[i][1] for i in eachindex(u)],[u[i][2] for i in eachindex(u)],[u[i][3] for i in eachindex(u)]), w)

time = Observable(0.0)

fig = Figure()
ax_u3 = CairoMakie.Axis3(fig[1,1], title = "8×7 feedforward 𝒩𝒩")
    CairoMakie.scatter!( ax_u3, pts_u, color = 1:N,  markersize = 2, colormap = cmap)
ax_u2 = CairoMakie.Axis(fig[3,1])
    CairoMakie.scatter!( ax_u2, pts_u, color = 1:N,  markersize = 2, colormap = cmap)
ax_w3 = CairoMakie.Axis3(fig[1,2], title = "Tsit5 integration")
    CairoMakie.scatter!( ax_w3, pts_w, color = 1:N,  markersize = 2, colormap = cmap)
ax_w2 = CairoMakie.Axis(fig[3,2])
    CairoMakie.scatter!( ax_w2, pts_w, color = 1:N,  markersize = 2, colormap = cmap)
Label(fig[2,1:2],"t = $(time[])", justification = :center, lineheight = 1.1)
σ = 10.0
β = 8/3
ρ = 28.0
function lorenz!(du, u, p, t)
    du[1] = p[1] * (u[2] - u[1])
    du[2] = u[1] * (p[3] - u[3]) - u[2]
    du[3] = u[1] * u[2] - p[2] * u[3]
    nothing
end

Δt = 0.5
function animstep!(i)
    if i > 1
        # NN advance
        u[] += ℳ.(u[])

        # ODE advance
        prob = ODEProblem(lorenz!, [],  (0.0, Δt), [σ, β, ρ])
        ensprob = EnsembleProblem(prob;
                prob_func = (prob, i, repeat) -> remake(prob, u0 = w[][i]))
        sim = solve(ensprob, Tsit5(), EnsembleThreads(); trajectories = length(w[]), saveat=[Δt])
        wtemp  = componentwise_vectors_timestep(sim, 1)
        w[] = [[wtemp[1][j], wtemp[2][j], wtemp[3][j]] for j in 1:length(w[])]

        time[] += Δt
    end
end

CairoMakie.record(fig, "lorenz-NN.mp4", 1:80, framerate = 20) do i # L is the number of frames
    animstep!(i)
end