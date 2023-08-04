using DifferentialEquations, CairoMakie, CSV
using Statistics
using SpecialFunctions, Distributions
using Flux, Flux.Optimise
using Flux: params
using Base.Iterators: partition
using LinearAlgebra
using CUDA

# lorenz-63 with burn-in (wait to sample to make sure we're on attractor)
σ = 10.0
β = 8/3
ρ = 28.0

Δt = 0.5

function lorenz!(du, u, p, t)
    du[1] = p[1] * (u[2] - u[1])
    du[2] = u[1] * (p[3] - u[3]) - u[2]
    du[3] = u[1] * u[2] - p[2] * u[3]
    nothing
end

N = 5000
u0 = Vector{Vector{Float32}}(undef, N)
u1 = Vector{Vector{Float32}}(undef, N)
u2 = Vector{Vector{Float32}}(undef, N)
for i in 1:N
    global u0, u1, u2
    u0[i] = 20.0 .*rand(3) .- 10.0
    prob = ODEProblem(lorenz!, u0[i],  (0.0, 10.0 + Δt), [σ, β, ρ])
    sol = solve(prob, Tsit5(), saveat=[10.0, 10.0 + Δt])
    u1[i] = sol.u[1]
    u2[i] = sol.u[2]
end
pts0 = Point3f0.([u0[i][1] for i in eachindex(u0)],[u0[i][2] for i in eachindex(u0)],[u0[i][3] for i in eachindex(u0)])
pts1 = Point3f0.([u1[i][1] for i in eachindex(u1)],[u1[i][2] for i in eachindex(u1)],[u1[i][3] for i in eachindex(u1)])
pts2 = Point3f0.([u2[i][1] for i in eachindex(u2)],[u2[i][2] for i in eachindex(u2)],[u2[i][3] for i in eachindex(u2)])

ptsu1 = Point3f0.([um1[i][1] for i in eachindex(um1)],[um1[i][2] for i in eachindex(um1)],[um1[i][3] for i in eachindex(um1)])#
ptsu2 = Point3f0.([um2[i][1] for i in eachindex(um2)],[um2[i][2] for i in eachindex(um2)],[um2[i][3] for i in eachindex(um2)])#

f = Figure()
ax_xy = CairoMakie.Axis(f[1,1])
    CairoMakie.scatter!(ax_xy,pts1, color=:black,  markersize = 1)
    CairoMakie.scatter!(ax_xy,pts2, color=:red,  markersize = 1)
    CairoMakie.scatter!(ax_xy,ptsu1, color=:green,  markersize = 1)#
    CairoMakie.scatter!(ax_xy,ptsu2, color=:blue,  markersize = 1)#
ax_yz = CairoMakie.Axis(f[1,2])
    CairoMakie.scatter!(ax_yz,pts1, color=:black,  markersize = 1)
    CairoMakie.scatter!(ax_yz,pts2, color=:red,  markersize = 1)
ax_xz = CairoMakie.Axis(f[2,1])
    CairoMakie.scatter!(ax_xz,pts1, color=:black,  markersize = 1)
    CairoMakie.scatter!(ax_xz,pts2, color=:red,  markersize = 1)
ax_3d = CairoMakie.Axis3(f[2,2])
    CairoMakie.scatter!(ax_3d,pts1, color=:black,  markersize = 1)
    CairoMakie.scatter!(ax_3d,pts2, color=:red,  markersize = 1)
f

CSV.write("uni_Lorenz.csv", Table(x=[u0[i][1] for i in eachindex(u0)],y=[u0[i][2] for i in eachindex(u0)],z=[u0[i][3] for i in eachindex(u0)]))
CSV.write("t0_Lorenz.csv", Table(x=[u1[i][1] for i in eachindex(u1)],y=[u1[i][2] for i in eachindex(u1)],z=[u1[i][3] for i in eachindex(u1)]))
CSV.write("t$(Δt)_Lorenz.csv", Table(x=[u2[i][1] for i in eachindex(u2)],y=[u2[i][2] for i in eachindex(u2)],z=[u2[i][3] for i in eachindex(u2)]))
CSV.write("dt$(Δt)_Lorenz.csv", Table(x=[u2[i][1] - u1[i][1] for i in eachindex(u2)],y=[u2[i][2] - u1[i][2] for i in eachindex(u2)],z=[u2[i][3] - u1[i][3] for i in eachindex(u2)]))

# partition the data
n = N - 500
valset  = n+1:N
train = ([(u1[i], u2[i]) for i in partition(1:n,100)]) |> gpu
val_u1 = u1[valset] |> gpu
val_u2 = u2[valset] |> gpu

# du = u2 .- u1
# train = ([(u1[i], du[i]) for i in partition(1:n,100)]) |> gpu
# val_u1 = u1[valset] |> gpu
# val_u2 = du[valset] |> gpu