using DifferentialEquations, CSV
using Statistics
using SpecialFunctions, Distributions
using LinearAlgebra, TypedTables

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

prob = ODEProblem(lorenz!, 20.0 .*rand(3) .- 10.0,  (0.0, 500.0), [σ, β, ρ])
sol = solve(prob, Tsit5(), saveat=10.0:Δt:500.0)
u = sol.u

CSV.write("ts_Lorenz.csv", Table(x=[u[i][1] for i in eachindex(u)],y=[u[i][2] for i in eachindex(u)],z=[u[i][3] for i in eachindex(u)]))